"""Authentication, CSRF and small in-process login rate limiting."""
from __future__ import annotations

import secrets
import time
from collections import defaultdict, deque
from typing import Deque

from fastapi import HTTPException, Request

from services import settings


_attempts: dict[str, Deque[float]] = defaultdict(deque)
_WINDOW_SECONDS = 60.0
_MAX_ATTEMPTS = 8


def configured() -> bool:
    return bool(
        settings.AUTH_USERNAME
        and settings.AUTH_PASSWORD
        and len(settings.SESSION_SECRET.encode("utf-8")) >= 32
    )


def require_user(request: Request) -> str:
    username = request.session.get("username")
    if not username or username != settings.AUTH_USERNAME:
        raise HTTPException(status_code=401, detail="Authentication required")
    return str(username)


def require_csrf(request: Request, username: str) -> None:
    token = request.headers.get("x-csrf-token", "")
    expected = request.session.get("csrf_token", "")
    if not token or not expected or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=403, detail="CSRF token is missing or invalid")


def check_login_rate_limit(client_ip: str) -> None:
    now = time.monotonic()
    events = _attempts[client_ip]
    while events and now - events[0] > _WINDOW_SECONDS:
        events.popleft()
    if len(events) >= _MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many login attempts. Wait a minute and try again.")
    events.append(now)


def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)
