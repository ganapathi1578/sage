"""Authenticated shared-network API and web application for Sage inference."""
from __future__ import annotations

import asyncio
import mimetypes
import secrets
import time
from contextlib import asynccontextmanager
import uuid
from pathlib import Path
from typing import Literal

import cv2
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

from services import settings
from services.catalog import DatasetCatalog
from services.runtime import ModelRuntime
from services.security import (
    check_login_rate_limit,
    configured as auth_configured,
    new_csrf_token,
    require_csrf,
    require_user,
)


catalog = DatasetCatalog()
runtime = ModelRuntime(catalog)

# A random fallback still signs cookies strongly, but sessions will not survive a
# restart. Login is disabled until administrators configure a stable secret key.
session_secret = settings.SESSION_SECRET if len(settings.SESSION_SECRET.encode("utf-8")) >= 32 else secrets.token_urlsafe(48)


def cleanup_orphaned_uploads() -> None:
    """Clear temporary uploads from prior processes; their in-memory IDs are gone."""
    settings.UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    for path in settings.UPLOAD_ROOT.iterdir():
        if not path.is_file():
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # A locked/temporarily unavailable file should not prevent startup.
            continue


async def _upload_janitor() -> None:
    while True:
        await asyncio.sleep(300)
        runtime.cleanup_uploads()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    cleanup_orphaned_uploads()
    task = asyncio.create_task(_upload_janitor())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Sage Model Playground", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=session_secret,
    session_cookie="sage_playground_session",
    max_age=settings.SESSION_MAX_AGE_SECONDS,
    same_site="lax",
    https_only=settings.COOKIE_SECURE,
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if request.url.path == "/api/uploads" and content_length:
        try:
            if int(content_length) > settings.MAX_UPLOAD_BYTES + 2 * 1024 * 1024:
                response = JSONResponse(status_code=413, content={"detail": "Upload request exceeds configured size limit"})
                response.headers.setdefault("X-Content-Type-Options", "nosniff")
                response.headers.setdefault("X-Frame-Options", "DENY")
                return response
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header"})
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; "
        "object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'",
    )
    return response


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=512)


class LoadModelRequest(BaseModel):
    run: str = Field(min_length=1, max_length=220)
    checkpoint: Literal["best_checkpoint.pt", "last_checkpoint.pt"]


class InferenceRequest(BaseModel):
    run: str = Field(min_length=1, max_length=220)
    checkpoint: Literal["best_checkpoint.pt", "last_checkpoint.pt"]
    scale: str = Field(min_length=1, max_length=40)
    split: Literal["train", "val", "test"] = "val"
    video_id: str | None = Field(default=None, max_length=500)
    upload_id: str | None = Field(default=None, max_length=80)
    example_id: int | None = None
    question_sentence_id: int | None = None
    option_sentence_ids: list[int] | None = None
    question_text: str = Field(min_length=1, max_length=settings.MAX_TEXT_LENGTH)
    options: list[str] = Field(min_length=2, max_length=settings.MAX_OPTIONS)
    expected_indices: list[int] = Field(default_factory=list, max_length=settings.MAX_OPTIONS)
    task_mode: Literal["single_choice", "multi_label"] = "single_choice"


def _verify_request_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    host = request.headers.get("host")
    if origin and host:
        from urllib.parse import urlsplit
        parsed = urlsplit(origin)
        if parsed.netloc.lower() != host.lower() or parsed.scheme not in {"http", "https"}:
            raise HTTPException(status_code=403, detail="Cross-origin requests are not allowed")


def _require_csrf(request: Request, username: str = Depends(require_user)) -> str:
    require_csrf(request, username)
    return username


def _scale_or_404(scale: str) -> None:
    if scale not in catalog.available_scales():
        raise HTTPException(status_code=404, detail=f"Dataset scale {scale!r} is not available on this server")


def _index_ready_or_progress(scale: str, split: str) -> dict:
    _scale_or_404(scale)
    status = catalog.index_status(scale, split, start=True)
    if status.get("status") == "failed":
        raise HTTPException(status_code=400, detail=status.get("error", "Dataset index failed"))
    return status


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(settings.FRONTEND_ROOT / "index.html")


@app.get("/assets/app.js", include_in_schema=False)
def app_js() -> FileResponse:
    return FileResponse(settings.FRONTEND_ROOT / "app.js", media_type="text/javascript")


@app.get("/assets/styles.css", include_in_schema=False)
def app_css() -> FileResponse:
    return FileResponse(settings.FRONTEND_ROOT / "styles.css", media_type="text/css")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "service": "sage-model-playground"}


@app.get("/api/session")
def session_state(request: Request) -> dict:
    username = request.session.get("username")
    is_authenticated = bool(username and username == settings.AUTH_USERNAME)
    return {
        "authenticated": is_authenticated,
        "username": username if is_authenticated else None,
        "csrf_token": request.session.get("csrf_token") if is_authenticated else None,
        "auth_configured": auth_configured(),
    }


@app.post("/api/login")
def login(body: LoginRequest, request: Request) -> dict:
    _verify_request_origin(request)
    client_ip = request.client.host if request.client else "unknown"
    check_login_rate_limit(client_ip)
    if not auth_configured():
        raise HTTPException(
            status_code=503,
            detail="Authentication is not configured. Set SAGE_PLAYGROUND_USERNAME, SAGE_PLAYGROUND_PASSWORD and a 32+ character SAGE_PLAYGROUND_SECRET_KEY.",
        )
    import hmac
    valid_username = hmac.compare_digest(body.username.encode("utf-8"), settings.AUTH_USERNAME.encode("utf-8"))
    valid_password = hmac.compare_digest(body.password.encode("utf-8"), settings.AUTH_PASSWORD.encode("utf-8"))
    if not (valid_username and valid_password):
        # Keep the same response for bad username/password.
        raise HTTPException(status_code=401, detail="Invalid username or password")
    request.session.clear()
    token = new_csrf_token()
    request.session["username"] = settings.AUTH_USERNAME
    request.session["csrf_token"] = token
    # Separate browser sessions even though this first deployment uses one shared
    # login. Temporary uploads are bound to this opaque session principal.
    request.session["principal_id"] = secrets.token_urlsafe(24)
    request.session["login_time"] = int(time.time())
    return {"authenticated": True, "username": settings.AUTH_USERNAME, "csrf_token": token}


@app.post("/api/logout")
def logout(request: Request, username: str = Depends(_require_csrf)) -> dict:
    request.session.clear()
    return {"ok": True}


@app.get("/api/status")
def status(_username: str = Depends(require_user)) -> dict:
    info = runtime.status()
    active_cfg = (runtime.active or {}).get("config", {})
    data_cfg = active_cfg.get("data", {}) if isinstance(active_cfg, dict) else {}
    if not data_cfg:
        try:
            data_cfg = catalog.data_cfg()
        except Exception:
            data_cfg = {}
    info["paths"] = {
        "checkpoint_root_ready": settings.OUTPUT_ROOT.is_dir(),
        "analysis_root_ready": settings.ANALYSIS_ROOT.is_dir(),
        "video_feature_store_ready": bool(info.get("video_store_ready", settings.VIDEO_STORE.exists())),
        "text_store_root_ready": any(
            settings.token_store_ready(scale, data_cfg) for scale in catalog.available_scales()
        ),
        "raw_video_roots_ready": [str(root) for root in settings.VIDEO_ROOTS if root.is_dir()],
    }
    return info


@app.get("/api/models")
def models(_username: str = Depends(require_user)) -> dict:
    return runtime.list_models()


@app.post("/api/model/load")
def load_model(body: LoadModelRequest, request: Request, _username: str = Depends(_require_csrf)) -> dict:
    _verify_request_origin(request)
    try:
        return runtime.load_model(body.run, body.checkpoint)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not load checkpoint: {exc}") from exc


@app.get("/api/datasets")
def datasets(_username: str = Depends(require_user)) -> dict:
    return catalog.list_datasets()


@app.get("/api/index-status")
def index_status(scale: str, split: Literal["train", "val", "test"] = "val", _username: str = Depends(require_user)) -> dict:
    _scale_or_404(scale)
    try:
        return catalog.index_status(scale, split, start=True)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/videos")
def videos(
    scale: str, split: Literal["train", "val", "test"] = "val",
    search: str = "", limit: int = 250, offset: int = 0,
    _username: str = Depends(require_user),
) -> dict:
    status = _index_ready_or_progress(scale, split)
    if status.get("status") != "complete":
        return JSONResponse(status_code=202, content=status)
    try:
        return catalog.get_indexed_videos(scale, split, search=search[:200], limit=limit, offset=offset)
    except RuntimeError:
        return JSONResponse(status_code=202, content=catalog.index_status(scale, split, start=True))


@app.get("/api/filters")
def filters(
    scale: str, split: Literal["train", "val", "test"] = "val", video_id: str | None = None,
    _username: str = Depends(require_user),
) -> dict:
    status = _index_ready_or_progress(scale, split)
    if status.get("status") != "complete":
        return JSONResponse(status_code=202, content=status)
    try:
        return catalog.list_filter_values(scale, split, video_id=video_id)
    except RuntimeError:
        return JSONResponse(status_code=202, content=catalog.index_status(scale, split, start=True))


@app.get("/api/questions")
def questions(
    scale: str, split: Literal["train", "val", "test"] = "val", video_id: str | None = None,
    task_type: str | None = None, reasoning_family: str | None = None,
    generator_family: str | None = None, limit: int = 10, offset: int = 0,
    _username: str = Depends(require_user),
) -> dict:
    status = _index_ready_or_progress(scale, split)
    if status.get("status") != "complete":
        return JSONResponse(status_code=202, content=status)
    try:
        result = catalog.query_examples(
            scale, split, video_id=video_id, task_type=task_type,
            reasoning_family=reasoning_family, generator_family=generator_family,
            limit=limit, offset=offset,
        )
        result["examples"] = catalog.add_texts(scale, result["examples"])
        return result
    except (RuntimeError, FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/video")
def video_file(video_id: str, _username: str = Depends(require_user)) -> FileResponse:
    path = catalog.resolve_video_path(video_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Original video file was not found in the configured video roots")
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream")


@app.post("/api/uploads")
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    username: str = Depends(_require_csrf),
) -> dict:
    _verify_request_origin(request)
    runtime.cleanup_uploads()
    owner_id = str(request.session.get("principal_id") or username)
    extension = Path(file.filename or "upload.mp4").suffix.lower()
    if extension not in settings.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported video extension {extension!r}")
    settings.UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    upload_id = uuid.uuid4().hex
    dest = settings.UPLOAD_ROOT / f"{upload_id}{extension}"
    written = 0
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > settings.MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f"Upload exceeds {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
                out.write(chunk)
        cap = cv2.VideoCapture(str(dest))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="This file could not be opened as a video")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if fps <= 0 or frames <= 0:
            raise HTTPException(status_code=400, detail="Video metadata is invalid or no frames can be decoded")
        duration = frames / fps
        if duration > settings.MAX_UPLOAD_DURATION_SECONDS:
            raise HTTPException(status_code=413, detail=f"Video duration exceeds {settings.MAX_UPLOAD_DURATION_SECONDS} second limit")
        runtime.upload_video(upload_id, owner_id, dest, Path(file.filename or "upload.mp4").name)
        return {
            "upload_id": upload_id, "filename": Path(file.filename or "upload.mp4").name,
            "size_bytes": written, "duration_seconds": round(duration, 2),
            "width": width, "height": height, "preview_url": f"/api/upload-video/{upload_id}",
        }
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@app.get("/api/upload-video/{upload_id}")
def uploaded_video_file(upload_id: str, request: Request, username: str = Depends(require_user)) -> FileResponse:
    owner_id = str(request.session.get("principal_id") or username)
    item = runtime.get_upload(upload_id, owner_id)
    if not item:
        raise HTTPException(status_code=404, detail="Uploaded video is missing or expired. Upload it again.")
    path = Path(item["path"])
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream")


@app.post("/api/inference")
def inference(body: InferenceRequest, request: Request, username: str = Depends(_require_csrf)) -> dict:
    _verify_request_origin(request)
    _scale_or_404(body.scale)
    if any(not item.strip() for item in body.options):
        raise HTTPException(status_code=422, detail="Answer options cannot be empty")
    if len(body.question_text.strip()) > settings.MAX_TEXT_LENGTH:
        raise HTTPException(status_code=422, detail="Question is too long")
    if any(len(item) > settings.MAX_TEXT_LENGTH for item in body.options):
        raise HTTPException(status_code=422, detail=f"Each answer option must be at most {settings.MAX_TEXT_LENGTH} characters")
    example = None
    if body.example_id is not None:
        idx_status = catalog.index_status(body.scale, body.split, start=False)
        if idx_status.get("status") != "complete":
            raise HTTPException(status_code=409, detail="Dataset index is not complete yet")
        try:
            example = catalog.get_example(body.scale, body.split, body.example_id)
        except (KeyError, RuntimeError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    owner_id = str(request.session.get("principal_id") or username)
    if body.upload_id:
        if not runtime.get_upload(body.upload_id, owner_id):
            raise HTTPException(status_code=404, detail="Uploaded video is missing or does not belong to your session")
    elif not body.video_id and example:
        body.video_id = example["video_id"]
    try:
        if example and not body.question_sentence_id:
            body.question_sentence_id = int(example["query_sentence_id"])
        if example and body.option_sentence_ids is None:
            body.option_sentence_ids = [int(x) for x in example["option_sentence_ids"]]
        result = runtime.predict(
            run=body.run, checkpoint=body.checkpoint, scale=body.scale, split=body.split,
            video_id=body.video_id, upload_id=body.upload_id, username=owner_id,
            question_text=body.question_text, options=body.options,
            question_sentence_id=body.question_sentence_id,
            option_sentence_ids=body.option_sentence_ids, example=example,
            expected_indices=body.expected_indices, task_mode=body.task_mode,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, KeyError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "services.app:app",
        host="0.0.0.0",
        port=int(__import__("os").environ.get("SAGE_PLAYGROUND_PORT", "8765")),
        reload=False,
    )
