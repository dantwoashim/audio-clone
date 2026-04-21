from __future__ import annotations

import click
import os

from f5_tts.studio.runtime import get_service
from f5_tts.studio.security import get_security_settings, verify_token


_app = None


def _get_app():
    global _app
    if _app is None:
        from f5_tts.studio.app import build_studio_app

        _app = build_studio_app()
    return _app


@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option("--share", "-s", default=False, is_flag=True, help="Share the app via Gradio share link")
@click.option("--inbrowser", "-i", is_flag=True, default=False, help="Automatically open the app in your browser")
def main(port, host, share, inbrowser):
    service = get_service()
    settings = get_security_settings()
    from f5_tts.studio.app import APP_CSS, STUDIO_THEME, studio_allowed_paths

    os.environ["F5_TTS_STUDIO_BIND_HOST"] = host or "127.0.0.1"
    os.environ["F5_TTS_STUDIO_SHARE_ACTIVE"] = "1" if share else "0"

    auth_dependency = None
    if settings.token_auth_enabled and not settings.basic_auth_enabled:
        def auth_dependency(request):
            token = request.headers.get("x-f5-tts-token") or request.query_params.get("access_token")
            if verify_token(token, settings):
                return "token-user"
            return None

    _get_app().queue(default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=inbrowser,
        pwa=True,
        theme=STUDIO_THEME,
        css=APP_CSS,
        allowed_paths=studio_allowed_paths(service),
        auth=((settings.username, settings.password) if settings.basic_auth_enabled else None),
        auth_dependency=auth_dependency,
        max_file_size=f"{settings.max_upload_mb}mb",
        show_error=True,
    )


if __name__ == "__main__":
    main()
