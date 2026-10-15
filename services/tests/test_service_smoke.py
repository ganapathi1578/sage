from __future__ import annotations

from pathlib import Path

import pytest

from services import settings
from services.app import app
from services.catalog import DatasetCatalog
from services.runtime import ModelRuntime


def test_health_endpoint_and_security_headers():
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_protected_model_endpoint_requires_authentication():
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.get("/api/models")
    assert response.status_code == 401


def test_checkpoint_name_cannot_traverse_output_root():
    runtime = ModelRuntime(DatasetCatalog())
    with pytest.raises(ValueError, match="Invalid experiment name"):
        runtime._checkpoint_path("../secrets", "last_checkpoint.pt")
    with pytest.raises(ValueError, match="Choose best_checkpoint.pt"):
        runtime._checkpoint_path("run-a", "secrets.pt")


def test_video_resolver_rejects_traversal_and_resolves_configured_files(tmp_path, monkeypatch):
    root = tmp_path / "videos"
    root.mkdir()
    valid = root / "clip.mp4"
    valid.write_bytes(b"not a real video; path validation only")
    monkeypatch.setattr(settings, "VIDEO_ROOTS", [root])
    catalog = DatasetCatalog()

    assert catalog.resolve_video_path("../clip.mp4") is None
    assert catalog.resolve_video_path("/etc/passwd") is None
    assert catalog.resolve_video_path("clip.mp4") == valid.resolve()


def test_login_csrf_and_logout_flow(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(settings, "AUTH_USERNAME", "test-researcher")
    monkeypatch.setattr(settings, "AUTH_PASSWORD", "test-only-password-not-for-deployment")
    monkeypatch.setattr(settings, "SESSION_SECRET", "unit-test-secret-that-is-long-enough-123456")
    with TestClient(app) as client:
        denied = client.get("/api/models")
        assert denied.status_code == 401

        login = client.post(
            "/api/login",
            json={"username": "test-researcher", "password": "test-only-password-not-for-deployment"},
        )
        assert login.status_code == 200, login.text
        csrf_token = login.json()["csrf_token"]

        allowed = client.get("/api/models")
        assert allowed.status_code == 200, allowed.text

        bad_csrf = client.post("/api/logout", json={})
        assert bad_csrf.status_code == 403
        logout = client.post("/api/logout", headers={"X-CSRF-Token": csrf_token}, json={})
        assert logout.status_code == 200, logout.text
        assert client.get("/api/models").status_code == 401


def test_token_store_discovery_rejects_pooled_manifests_and_finds_ragged_store(tmp_path, monkeypatch):
    import json

    monkeypatch.delenv("SAGE_PLAYGROUND_TEXT_STORE_ROOT", raising=False)
    monkeypatch.setattr(settings, "REPO_ROOT", tmp_path)
    pooled_root = tmp_path / "pooled"
    token_root = tmp_path / "token-level"
    for root in (pooled_root, token_root):
        (root / "10k").mkdir(parents=True)
    (pooled_root / "10k" / "manifest.json").write_text(
        json.dumps({"format": "pooled_sentence_embeddings_v1", "complete": True, "parts": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "TEXT_STORE_ROOT", pooled_root)
    assert settings.resolve_token_store_root("10k", {"text_token_store_root": str(pooled_root)}) is None

    (token_root / "10k" / "manifest.json").write_text(
        json.dumps({"format": "ragged_token_embeddings_v1", "complete": True, "parts": []}),
        encoding="utf-8",
    )
    chosen = settings.resolve_token_store_root("10k", {"text_token_store_root": str(token_root)})
    assert chosen == token_root.resolve()


def test_uploaded_feature_preparation_is_model_specific_and_does_not_mutate_cache():
    import numpy as np
    from services.runtime import ModelRuntime

    raw_features = np.arange(5 * 2 * 3, dtype=np.float32).reshape(5, 2, 3)
    original = raw_features.copy()
    cached = {
        "features": raw_features,
        "spatial_xy": np.asarray([[0.25, 0.5], [0.75, 0.5]], dtype=np.float32),
        "timestamps_sec": np.arange(5, dtype=np.float32),
        "temporal_positions": np.arange(5, dtype=np.float32),
    }
    prepared = ModelRuntime._prepare_uploaded_features(
        cached, {"max_video_frames": 32, "target_fps": 0.5}, 32
    )
    pooled = ModelRuntime._apply_spatial_pooling(prepared, "mean")

    assert pooled["features"].shape == (3, 1, 3)
    assert pooled["spatial_xy"].shape == (1, 2)
    assert pooled["timestamps_sec"].tolist() == [0.0, 2.0, 4.0]
    assert pooled["temporal_positions"].tolist() == [0.0, 1.0, 2.0]
    assert np.array_equal(cached["features"], original)
    assert cached["features"].shape == (5, 2, 3)
