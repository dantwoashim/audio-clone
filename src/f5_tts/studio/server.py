from __future__ import annotations

import os
import tempfile
from pathlib import Path

import click
import uvicorn
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse

from f5_tts.infer.utils_infer import tempfile_kwargs
from f5_tts.studio.runtime import get_service
from f5_tts.studio.schemas import (
    EditRenderRequest,
    GenerationRequest,
    ProjectCreate,
    PronunciationRuleCreate,
    VoiceProfileCreate,
)
from f5_tts.studio.security import (
    ensure_upload_within_limit,
    get_security_settings,
    verify_basic_header,
    verify_token,
)


def _save_upload(service, upload: UploadFile) -> str:
    settings = get_security_settings()
    suffix = Path(upload.filename or "upload.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(dir=service.paths.incoming, suffix=suffix, **tempfile_kwargs) as handle:
        target = Path(handle.name)
    with target.open("wb") as output:
        total = 0
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            ensure_upload_within_limit(total, settings)
            output.write(chunk)
    return str(target)


def _is_authorized_request(request: Request) -> bool:
    settings = get_security_settings()
    if not settings.auth_enabled:
        return True

    candidate_token = request.headers.get("x-f5-tts-token") or request.query_params.get("access_token")
    if settings.token_auth_enabled:
        if verify_token(candidate_token, settings):
            return True
        if not settings.basic_auth_enabled:
            return False
    return verify_basic_header(request.headers.get("authorization"), settings)


def _unauthorized_response() -> PlainTextResponse:
    settings = get_security_settings()
    headers = {"WWW-Authenticate": "Basic"} if settings.basic_auth_enabled else {}
    return PlainTextResponse("Unauthorized", status_code=401, headers=headers)


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

    @router.get("/projects/{project_id}/voice-profiles")
    def list_voice_profiles(project_id: int):
        return service.list_voice_profiles(project_id)

    @router.post("/pronunciation-rules")
    def save_pronunciation_rule(payload: PronunciationRuleCreate):
        return service.save_pronunciation_rule(payload.project_id, payload.source, payload.replacement)

    @router.post("/voice-profiles")
    def save_voice_profile(payload: VoiceProfileCreate):
        return service.save_voice_profile(
            payload.project_id,
            payload.name,
            payload.reference_ids,
            description=payload.description,
            profile_id=payload.profile_id,
        )

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

    @router.post("/edit-sources/analyze")
    def analyze_edit_source(
        project_id: int = Form(...),
        name: str = Form(...),
        transcript: str = Form(""),
        audio: UploadFile = File(...),
    ):
        staged = _save_upload(service, audio)
        try:
            asset, metadata = service.ingest_edit_source(project_id, name, staged, transcript)
            return {"asset": asset, "analysis": metadata}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/generations")
    def create_generation(payload: GenerationRequest):
        return service.enqueue_generation(payload)

    @router.post("/edits/render")
    def create_edit_render(payload: EditRenderRequest):
        return service.render_edit_now(
            project_id=payload.project_id,
            source_asset_id=payload.source_asset_id,
            take_name=payload.name,
            target_text=payload.target_text,
            replacement_text=payload.replacement_text,
            occurrence=payload.occurrence,
            action=payload.action,
            preserve_timing=payload.preserve_timing,
            nfe_step=payload.nfe_step,
            render_spectrogram=payload.render_spectrogram,
        )

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
    settings = get_security_settings()
    app = FastAPI(title="F5-TTS Studio", docs_url="/docs", redoc_url="/redoc")

    @app.middleware("http")
    async def studio_auth_guard(request: Request, call_next):
        if request.url.path == "/healthz":
            return await call_next(request)
        if not _is_authorized_request(request):
            return _unauthorized_response()
        return await call_next(request)

    app.include_router(create_api_router(service))

    @app.get("/")
    def root():
        return RedirectResponse(url="/app" if mount_studio else "/docs")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "profile": service.system_profile()}

    if mount_studio:
        import gradio as gr

        from f5_tts.studio.app import APP_CSS, STUDIO_THEME, build_studio_app, studio_allowed_paths

        studio_app = build_studio_app()
        app = gr.mount_gradio_app(
            app,
            studio_app,
            path="/app",
            allowed_paths=studio_allowed_paths(service),
            footer_links=[],
            pwa=True,
            theme=STUDIO_THEME,
            css=APP_CSS,
            auth=((settings.username, settings.password) if settings.basic_auth_enabled else None),
            max_file_size=f"{settings.max_upload_mb}mb",
        )

    return app


@click.command()
@click.option("--port", "-p", default=7862, type=int, help="Port to run the FastAPI + Gradio server on")
@click.option("--host", "-H", default="127.0.0.1", help="Host to bind the server to")
def main(port: int, host: str):
    os.environ["F5_TTS_STUDIO_BIND_HOST"] = host
    os.environ["F5_TTS_STUDIO_SHARE_ACTIVE"] = "0"
    uvicorn.run(create_server_app(), host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
