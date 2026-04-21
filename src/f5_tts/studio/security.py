from __future__ import annotations

import base64
import hmac
import os
from dataclasses import dataclass


DEFAULT_MAX_UPLOAD_MB = 64


@dataclass(frozen=True)
class StudioSecuritySettings:
    username: str
    password: str
    token: str
    max_upload_mb: int
    share_active: bool
    public_url: str
    bind_host: str

    @property
    def basic_auth_enabled(self) -> bool:
        return bool(self.username and self.password)

    @property
    def token_auth_enabled(self) -> bool:
        return bool(self.token)

    @property
    def auth_enabled(self) -> bool:
        return self.basic_auth_enabled or self.token_auth_enabled

    @property
    def auth_mode(self) -> str:
        if self.basic_auth_enabled and self.token_auth_enabled:
            return "basic+token"
        if self.basic_auth_enabled:
            return "basic"
        if self.token_auth_enabled:
            return "token"
        return "none"

    @property
    def public_surface(self) -> bool:
        host = (self.bind_host or "").strip().lower()
        non_loopback_host = host not in {"", "127.0.0.1", "localhost"}
        return self.share_active or bool(self.public_url) or non_loopback_host

    @property
    def sharing_warning(self) -> str | None:
        if not self.public_surface:
            return None
        if self.auth_enabled:
            return "This studio is exposed beyond localhost, but authentication is enabled."
        return "This studio is exposed beyond localhost without authentication. Protect it before sharing voice-cloning access."


def get_security_settings() -> StudioSecuritySettings:
    max_upload_raw = (os.environ.get("F5_TTS_MAX_UPLOAD_MB") or str(DEFAULT_MAX_UPLOAD_MB)).strip()
    try:
        max_upload_mb = max(1, int(max_upload_raw))
    except ValueError:
        max_upload_mb = DEFAULT_MAX_UPLOAD_MB
    return StudioSecuritySettings(
        username=(os.environ.get("F5_TTS_STUDIO_USERNAME") or "").strip(),
        password=(os.environ.get("F5_TTS_STUDIO_PASSWORD") or "").strip(),
        token=(os.environ.get("F5_TTS_STUDIO_TOKEN") or "").strip(),
        max_upload_mb=max_upload_mb,
        share_active=(os.environ.get("F5_TTS_STUDIO_SHARE_ACTIVE") or "0").strip() == "1",
        public_url=(os.environ.get("F5_TTS_STUDIO_PUBLIC_URL") or "").strip(),
        bind_host=(os.environ.get("F5_TTS_STUDIO_BIND_HOST") or "127.0.0.1").strip(),
    )


def max_upload_bytes(settings: StudioSecuritySettings | None = None) -> int:
    settings = settings or get_security_settings()
    return settings.max_upload_mb * 1024 * 1024


def ensure_upload_within_limit(size_bytes: int, settings: StudioSecuritySettings | None = None) -> None:
    limit = max_upload_bytes(settings)
    if size_bytes > limit:
        limit_mb = limit // (1024 * 1024)
        raise ValueError(f"Upload exceeds the configured {limit_mb} MB limit.")


def verify_token(candidate: str | None, settings: StudioSecuritySettings | None = None) -> bool:
    settings = settings or get_security_settings()
    if not settings.token_auth_enabled:
        return True
    if not candidate:
        return False
    return hmac.compare_digest(candidate.strip(), settings.token)


def verify_basic_header(authorization_header: str | None, settings: StudioSecuritySettings | None = None) -> bool:
    settings = settings or get_security_settings()
    if not settings.basic_auth_enabled:
        return True
    if not authorization_header or not authorization_header.lower().startswith("basic "):
        return False
    encoded = authorization_header.split(" ", 1)[1].strip()
    try:
        decoded = base64.b64decode(encoded).decode("utf-8")
    except Exception:
        return False
    username, _, password = decoded.partition(":")
    return hmac.compare_digest(username, settings.username) and hmac.compare_digest(password, settings.password)
