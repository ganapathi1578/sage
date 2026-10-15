"""Environment-backed settings for the shared Sage Model Playground."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(os.environ.get("SAGE_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
SERVICES_ROOT = REPO_ROOT / "services"
FRONTEND_ROOT = SERVICES_ROOT / "frontend"


def resolve_repo_path(value: str | Path, default: str | Path) -> Path:
    raw = os.environ.get(str(value), None) if isinstance(value, str) else None
    selected = raw if raw is not None else str(default)
    path = Path(selected).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def configured_path(env_name: str, default: str | Path) -> Path:
    value = os.environ.get(env_name, str(default))
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def parse_path_list(raw: str | None, defaults: Iterable[Path]) -> list[Path]:
    if not raw:
        return [Path(item).expanduser().resolve() for item in defaults]
    # Linux server deployment is expected; pathsep is ':' on the target OS.
    return [Path(item.strip()).expanduser().resolve() for item in raw.split(os.pathsep) if item.strip()]


OUTPUT_ROOT = configured_path("SAGE_PLAYGROUND_OUTPUT_ROOT", "outputs/experiments")
ANALYSIS_ROOT = configured_path("SAGE_PLAYGROUND_ANALYSIS_ROOT", "UniProp/data/analysis")
TEXT_STORE_ROOT = configured_path("SAGE_PLAYGROUND_TEXT_STORE_ROOT", "data/embeddings/text/all-MiniLM-L6-v2-tokens")
VIDEO_STORE = configured_path("SAGE_PLAYGROUND_VIDEO_STORE", "data/embeddings/charades_v1_480/fastvithd_256_1fps.zarr")
VIDEO_ROOTS = parse_path_list(
    os.environ.get("SAGE_PLAYGROUND_VIDEO_ROOTS"),
    [REPO_ROOT / "data/datasets/charades/Charades_v1_480"],
)
INDEX_ROOT = configured_path("SAGE_PLAYGROUND_INDEX_ROOT", "services/cache/catalogs")
UPLOAD_ROOT = configured_path("SAGE_PLAYGROUND_UPLOAD_ROOT", "services/cache/uploads")
FASTVITHD_CONFIG = configured_path("SAGE_PLAYGROUND_FASTVITHD_CONFIG", "configs/embeddings/fastvithd_charades.yaml")
TRAINING_CONFIG = configured_path("SAGE_PLAYGROUND_TRAINING_CONFIG", "configs/training/experiments/llama_10k.yaml")
TEXT_MODEL_ID = os.environ.get("SAGE_PLAYGROUND_TEXT_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")



def repo_path(value: str | Path) -> Path:
    """Resolve a path from a Sage config relative to the repository root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _token_manifest_is_ready(root: Path, scale: str) -> bool:
    """Only report ragged token stores as ready; pooled sentence manifests are incompatible."""
    manifest_path = root / scale / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        import json
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        return (
            manifest.get("format") == "ragged_token_embeddings_v1"
            and manifest.get("complete") is True
            and isinstance(manifest.get("parts"), list)
        )
    except (OSError, ValueError, TypeError):
        return False


def token_store_candidates(data_cfg: dict | None = None) -> list[Path]:
    """Return deduplicated token-store roots, favoring an explicit env path or the model config."""
    roots: list[Path] = []
    if os.environ.get("SAGE_PLAYGROUND_TEXT_STORE_ROOT"):
        roots.append(TEXT_STORE_ROOT)
    configured = (data_cfg or {}).get("text_token_store_root") or (data_cfg or {}).get("text_store_root")
    if configured:
        roots.append(repo_path(configured))
    roots.extend([
        TEXT_STORE_ROOT,
        REPO_ROOT / "data/embeddings/text/all-MiniLM-L6-v2-tokens",
        REPO_ROOT / "data/embeddings/text/all-MiniLM-L6-v2",
    ])
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def resolve_token_store_root(scale: str, data_cfg: dict | None = None) -> Path | None:
    """Find the first compatible token store for this scale, else return None."""
    for root in token_store_candidates(data_cfg):
        if _token_manifest_is_ready(root, scale):
            return root
    return None


def token_store_ready(scale: str, data_cfg: dict | None = None) -> bool:
    return resolve_token_store_root(scale, data_cfg) is not None


MAX_UPLOAD_BYTES = int(os.environ.get("SAGE_PLAYGROUND_MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))
MAX_UPLOAD_DURATION_SECONDS = int(os.environ.get("SAGE_PLAYGROUND_MAX_UPLOAD_DURATION_SECONDS", "600"))
MAX_TEXT_LENGTH = int(os.environ.get("SAGE_PLAYGROUND_MAX_TEXT_LENGTH", "2000"))
MAX_OPTIONS = int(os.environ.get("SAGE_PLAYGROUND_MAX_OPTIONS", "16"))
MAX_VIDEO_FRAMES = max(1, int(os.environ.get("SAGE_PLAYGROUND_MAX_VIDEO_FRAMES", "32")))
SESSION_MAX_AGE_SECONDS = int(os.environ.get("SAGE_PLAYGROUND_SESSION_MAX_AGE_SECONDS", str(12 * 60 * 60)))
COOKIE_SECURE = os.environ.get("SAGE_PLAYGROUND_COOKIE_SECURE", "0").strip().lower() in {"1", "true", "yes"}
DEVICE = os.environ.get("SAGE_PLAYGROUND_DEVICE", "auto").strip().lower()
TEXT_DEVICE = os.environ.get("SAGE_PLAYGROUND_TEXT_DEVICE", "cpu").strip().lower()

AUTH_USERNAME = os.environ.get("SAGE_PLAYGROUND_USERNAME", "").strip()
AUTH_PASSWORD = os.environ.get("SAGE_PLAYGROUND_PASSWORD", "")
SESSION_SECRET = os.environ.get("SAGE_PLAYGROUND_SECRET_KEY", "")

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
SCALE_ORDER = ["10k", "100k", "1m", "10m", "100m", "full"]
SPLITS = ["train", "val", "test"]
