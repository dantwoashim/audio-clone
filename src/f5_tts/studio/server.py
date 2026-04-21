from __future__ import annotations

import tempfile
from pathlib import Path

import click
import gradio as gr
import uvicorn
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

from f5_tts.infer.utils_infer import tempfile_kwargs
from f5_tts.studio.app import APP_CSS, STUDIO_THEME, build_studio_app
from f5_tts.studio.runtime import get_service
from f5_tts.studio.schemas import GenerationRequest, ProjectCreate, PronunciationRuleCreate


def _save_upload(service, upload: UploadFile) -> str:
    suffix = Path(upload.filename or "upload.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(dir=service.paths.incoming, suffix=suffix, **tempfile_kwargs) as handle:
        target = Path(handle.name)
    with target.open("wb") as output:
        output.write(upload.file.read())
    return str(target)


def create_api_router(service) -> APIRouter:
    router = APIRouter(prefix="/api/v1")

    @router.get("/system/profile")
    def get_system_profile():
        return service.system_profile()

    @router.get("/projects")
    def list_projects():
        return service.list_projects()

    @router.post("/projects")
    def create_project(payload: ProjectCreate):
        return service.create_project(payload.name, payload.description)

    @router.get("/projects/{project_id}")
    def get_project(project_id: int):
        try:
            return service.get_project_detail(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/pronunciation-rules")
    def save_pronunciation_rule(payload: PronunciationRuleCreate):
        return service.save_pronunciation_rule(payload.project_id, payload.source, payload.replacement)

    @router.post("/references/analyze")
    def analyze_reference(
        project_id: int = Form(...),
        name: str = Form(...),
        transcript: str = Form(""),
        kind: str = Form("reference"),
        context_notes: str = Form(""),
        gen_text: str = Form(""),
        audio: UploadFile = File(...),
    ):
        staged = _save_upload(service, audio)
        try:
            if kind == "style":
                saved, analysis = service.ingest_style(project_id, name, staged, transcript, context_notes, gen_text)
            else:
                saved, analysis = service.ingest_reference(project_id, name, staged, transcript)
            return {"asset": saved, "analysis": analysis}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/generations")
    def create_generation(payload: GenerationRequest):
        return service.enqueue_generation(payload)

    @router.get("/jobs/{job_id}")
    def get_job(job_id: int):
        try:
            return service.store.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: int):
        try:
            return service.cancel_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/assets/{asset_id}")
    def get_asset(asset_id: int):
        try:
            return service.store.get_audio_asset(asset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/assets/{asset_id}/download")
    def download_asset(asset_id: int):
        try:
            asset = service.store.get_audio_asset(asset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(asset["path"], filename=Path(asset["path"]).name)

    return router


def create_server_app(mount_studio: bool = True, service=None) -> FastAPI:
    service = service or get_service()
    app = FastAPI(title="F5-TTS Studio", docs_url="/docs", redoc_url="/redoc")
    app.include_router(create_api_router(service))

    @app.get("/")
    def root():
        return RedirectResponse(url="/app" if mount_studio else "/docs")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "profile": service.system_profile()}

    if mount_studio:
        studio_app = build_studio_app()
        app = gr.mount_gradio_app(
            app,
            studio_app,
            path="/app",
            allowed_paths=[str(service.paths.root), str(service.paths.cache)],
            footer_links=[],
            pwa=True,
            theme=STUDIO_THEME,
            css=APP_CSS,
        )

    return app

@click.command()
@click.option("--port", "-p", default=7862, type=int, help="Port to run the FastAPI + Gradio server on")
@click.option("--host", "-H", default="127.0.0.1", help="Host to bind the server to")
def main(port: int, host: str):
    uvicorn.run(create_server_app(), host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
