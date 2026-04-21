from __future__ import annotations

import click

from f5_tts.studio.app import APP_CSS, STUDIO_THEME, build_studio_app
from f5_tts.studio.runtime import get_service


app = build_studio_app()


@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option("--share", "-s", default=False, is_flag=True, help="Share the app via Gradio share link")
@click.option("--inbrowser", "-i", is_flag=True, default=False, help="Automatically open the app in your browser")
def main(port, host, share, inbrowser):
    service = get_service()
    app.queue(default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=inbrowser,
        pwa=True,
        theme=STUDIO_THEME,
        css=APP_CSS,
        allowed_paths=[str(service.paths.root), str(service.paths.cache)],
        show_error=True,
    )


if __name__ == "__main__":
    main()
