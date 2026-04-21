from __future__ import annotations

import json
import re

import gradio as gr

from f5_tts.studio.profiles import PROFILE_MAP
from f5_tts.studio.runtime import format_duration, get_service
from f5_tts.studio.schemas import GenerationRequest


TITLE = "Voice Studio Pro"

APP_CSS = """
:root {
  --studio-ink: #16231b;
  --studio-muted: #5f6c63;
  --studio-paper: #f5f1e8;
  --studio-card: rgba(255, 252, 246, 0.9);
  --studio-line: rgba(22, 35, 27, 0.12);
  --studio-accent: #0f7c5b;
  --studio-accent-2: #d95b43;
  --studio-glow: rgba(15, 124, 91, 0.18);
}

.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(217, 91, 67, 0.16), transparent 28%),
    radial-gradient(circle at top right, rgba(15, 124, 91, 0.18), transparent 24%),
    linear-gradient(180deg, #f7f2e8 0%, #efe6d4 100%);
  color: var(--studio-ink);
}

.studio-shell {
  max-width: 1320px;
  margin: 0 auto;
  padding: 22px 18px 40px;
}

.studio-hero,
.studio-panel {
  border: 1px solid var(--studio-line);
  border-radius: 26px;
  background: var(--studio-card);
  box-shadow: 0 18px 55px rgba(60, 48, 28, 0.07);
  backdrop-filter: blur(10px);
}

.studio-hero {
  overflow: hidden;
  margin-bottom: 18px;
  padding: 28px;
}

.studio-hero h1 {
  margin: 0;
  font-size: clamp(2.6rem, 5vw, 4.4rem);
  line-height: 0.92;
  letter-spacing: -0.05em;
}

.studio-hero p {
  margin: 16px 0 0;
  max-width: 760px;
  color: var(--studio-muted);
  font-size: 1rem;
  line-height: 1.65;
}

.studio-panel {
  padding: 18px;
}

.studio-tab-note {
  color: var(--studio-muted);
  font-size: 0.95rem;
  margin: 0 0 10px;
}

.studio-primary button {
  min-height: 52px;
  border-radius: 18px !important;
  background: linear-gradient(135deg, #0f7c5b, #136d87) !important;
  box-shadow: 0 16px 35px var(--studio-glow);
}

.studio-secondary button {
  min-height: 48px;
  border-radius: 16px !important;
}
"""

HERO_HTML = f"""
<section class="studio-hero">
  <h1>{TITLE}</h1>
  <p>
    A local-first voice cloning studio built around F5-TTS, upgraded for real projects: saved references,
    reusable style prompts, queue-backed batch jobs, A/B take review, export bundles, and an M4-aware runtime profile
    so the whole machine does not get swallowed by one render.
  </p>
</section>
"""

SCRIPT_PRESETS = {
    "Product demo": "Welcome to Voice Studio Pro. This render uses a saved reference voice, reusable style guidance, and a quality-focused local workflow.",
    "Podcast intro": "Welcome back to the show. Today we are testing a cleaner local voice production workflow built for Apple Silicon.",
    "Support update": "Thanks for reaching out. We have your request and the next update is being prepared now. Please stay with us for a moment.",
}

STUDIO_THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="orange",
    neutral_hue="stone",
    radius_size=gr.themes.sizes.radius_lg,
    text_size=gr.themes.sizes.text_md,
)


def _choice_pairs(items: list[dict], label_key: str = "name") -> list[tuple[str, int]]:
    return [(item[label_key], int(item["id"])) for item in items]


def _table_rows(items: list[dict], columns: list[str]) -> list[list[object]]:
    rows: list[list[object]] = []
    for item in items:
        rows.append([item.get(column, "") for column in columns])
    return rows


def _project_markdown(project: dict | None) -> str:
    if not project:
        return "No project selected yet."
    return (
        f"### {project['name']}\n"
        f"{project['description'] or 'No description yet.'}\n\n"
        f"- References: {len(project['references'])}\n"
        f"- Styles: {len(project['styles'])}\n"
        f"- Takes: {len(project['assets'])}\n"
        f"- Jobs: {len(project['jobs'])}"
    )


def _reference_rows(items: list[dict]) -> list[list[object]]:
    rows = []
    for item in items:
        analysis = item["analysis"]
        rows.append(
            [
                item["id"],
                item["name"],
                format_duration(float(analysis.get("duration_seconds", 0.0))),
                analysis.get("backend", "manual"),
                " | ".join(analysis.get("warnings", [])[:2]),
            ]
        )
    return rows


def _style_rows(items: list[dict]) -> list[list[object]]:
    rows = []
    for item in items:
        analysis = item["analysis"]
        rows.append(
            [
                item["id"],
                item["name"],
                format_duration(float(analysis.get("duration_seconds", 0.0))),
                f"{float(analysis.get('recommended_speed', 1.0)):.2f}",
                ", ".join(analysis.get("matched_keywords", [])),
            ]
        )
    return rows


def _asset_rows(items: list[dict]) -> list[list[object]]:
    rows = []
    for item in items:
        rows.append(
            [
                item["id"],
                item["kind"],
                item["label"],
                format_duration(float(item.get("duration_seconds", 0.0) or 0.0)),
                item["created_at"],
            ]
        )
    return rows


def _job_rows(items: list[dict]) -> list[list[object]]:
    rows = []
    for item in items:
        rows.append(
            [
                item["id"],
                item["name"],
                item["status"],
                item["updated_at"],
                item.get("error_text") or "",
            ]
        )
    return rows


def _rule_rows(items: list[dict]) -> list[list[object]]:
    return [[item["id"], item["source"], item["replacement"], item["updated_at"]] for item in items]


def create_studio_app():
    service = get_service()

    def project_updates(selected_project_id: int | None = None):
        projects = service.list_projects()
        choices = _choice_pairs(projects)
        if not choices:
            project = service.create_project("Studio Sandbox", "Default local project for previews.")
            projects = service.list_projects()
            choices = _choice_pairs(projects)
            selected_project_id = project["id"]
        valid_ids = {value for _, value in choices}
        if selected_project_id not in valid_ids:
            selected_project_id = choices[0][1]

        project_detail = service.get_project_detail(int(selected_project_id))
        references = project_detail["references"]
        styles = project_detail["styles"]
        assets = project_detail["assets"]
        jobs = project_detail["jobs"]
        rules = project_detail["pronunciation_rules"]

        reference_choices = _choice_pairs(references)
        style_choices = [("No style prompt", 0)] + _choice_pairs(styles)
        asset_choices = _choice_pairs(assets, label_key="label")
        job_choices = [(f"{job['id']} - {job['name']}", int(job["id"])) for job in jobs if job["status"] in {"queued", "running"}]

        return (
            gr.update(choices=choices, value=selected_project_id),
            _project_markdown(project_detail),
            gr.update(choices=reference_choices, value=reference_choices[0][1] if reference_choices else None),
            gr.update(choices=style_choices, value=0),
            gr.update(choices=reference_choices, value=reference_choices[0][1] if reference_choices else None),
            gr.update(choices=style_choices, value=0),
            _reference_rows(references),
            _style_rows(styles),
            _asset_rows(assets),
            _job_rows(jobs),
            _job_rows(jobs),
            _rule_rows(rules),
            gr.update(choices=asset_choices, value=asset_choices[0][1] if asset_choices else None),
            gr.update(choices=asset_choices, value=asset_choices[1][1] if len(asset_choices) > 1 else (asset_choices[0][1] if asset_choices else None)),
            gr.update(choices=asset_choices, value=asset_choices[0][1] if asset_choices else None),
            gr.update(choices=job_choices, value=job_choices[0][1] if job_choices else None),
            json.dumps(service.system_profile(), indent=2),
        )

    def create_project(name: str, description: str, current_project_id: int | None):
        if not name.strip():
            raise gr.Error("Enter a project name first.")
        project = service.create_project(name, description)
        refresh = project_updates(project["id"])
        return refresh + ("Project created.",)

    def load_preset(preset_name: str):
        return SCRIPT_PRESETS[preset_name]

    def save_reference(project_id: int, consent: bool, name: str, ref_audio: str, ref_text: str):
        if not project_id:
            raise gr.Error("Select a project first.")
        if not consent:
            raise gr.Error("Confirm that you have the right to use this voice reference before saving it.")
        if not ref_audio:
            raise gr.Error("Upload or record a reference clip first.")
        staged = service.stage_upload(ref_audio)
        saved, analysis = service.ingest_reference(project_id, name or "Reference Voice", staged, ref_text)
        refresh = project_updates(project_id)
        summary = (
            f"Saved reference #{saved['id']}.\n"
            f"Transcript: {analysis['transcript']}\n"
            f"Warnings: {', '.join(analysis['warnings']) if analysis['warnings'] else 'None'}"
        )
        return (analysis["transcript"], summary) + refresh

    def save_style(project_id: int, name: str, style_audio: str, style_text: str, context_notes: str, gen_text: str):
        if not project_id:
            raise gr.Error("Select a project first.")
        if not style_audio:
            raise gr.Error("Upload or record a style prompt first.")
        staged = service.stage_upload(style_audio)
        saved, analysis = service.ingest_style(project_id, name or "Style Prompt", staged, style_text, context_notes, gen_text)
        refresh = project_updates(project_id)
        summary = (
            f"Saved style #{saved['id']}.\n"
            f"Transcript: {analysis['transcript']}\n"
            f"Recommended speed: {analysis['recommended_speed']:.2f}"
        )
        return (analysis["transcript"], summary) + refresh

    def build_request(
        project_id: int,
        reference_id: int,
        style_id: int,
        name: str,
        text: str,
        mode: str,
        use_style_prompt: bool,
        context_notes: str,
        speed: float,
        nfe_step: int,
        remove_silence: bool,
        render_spectrogram: bool,
    ) -> GenerationRequest:
        if not project_id:
            raise gr.Error("Select a project first.")
        if not reference_id:
            raise gr.Error("Save or choose a reference voice first.")
        if not text.strip():
            raise gr.Error("Enter text to generate.")
        return GenerationRequest(
            project_id=int(project_id),
            reference_id=int(reference_id),
            style_id=None if not style_id else int(style_id),
            name=name.strip() or f"{mode.title()} Render",
            text=text,
            mode=mode,
            use_style_prompt=use_style_prompt,
            context_notes=context_notes,
            speed=None if speed <= 0 else speed,
            nfe_step=None if nfe_step <= 0 else int(nfe_step),
            remove_silence=remove_silence,
            render_spectrogram=render_spectrogram,
        )

    def estimate_render(
        project_id: int,
        reference_id: int,
        style_id: int,
        name: str,
        text: str,
        mode: str,
        use_style_prompt: bool,
        context_notes: str,
        speed: float,
        nfe_step: int,
    ):
        request = build_request(
            project_id,
            reference_id,
            style_id,
            name,
            text,
            mode,
            use_style_prompt,
            context_notes,
            speed,
            nfe_step,
            False,
            False,
        )
        estimate = service.estimate(request)
        return (
            f"Estimated generation time: {format_duration(estimate['estimated_generation_seconds'])}\n"
            f"Predicted output length: {format_duration(estimate['predicted_output_seconds'])}\n"
            f"Chunks: {estimate['chunks']}\n"
            f"Effective speed: {estimate['effective_speed']:.2f}\n"
            f"Queue depth: {estimate['queue_depth']}\n"
            f"Estimate source: {estimate['confidence']}"
        )

    def render_now(
        project_id: int,
        reference_id: int,
        style_id: int,
        name: str,
        text: str,
        mode: str,
        use_style_prompt: bool,
        context_notes: str,
        speed: float,
        nfe_step: int,
        remove_silence: bool,
        render_spectrogram: bool,
    ):
        request = build_request(
            project_id,
            reference_id,
            style_id,
            name,
            text,
            mode,
            use_style_prompt,
            context_notes,
            speed,
            nfe_step,
            remove_silence,
            render_spectrogram,
        )
        job = service.render_now(request)
        if job["status"] != "completed":
            raise gr.Error(job.get("error_text") or "Render failed.")

        result = job["result"]
        refresh = project_updates(project_id)
        status = (
            f"{mode.title()} render complete.\n"
            f"Take #{result['asset_id']} saved to the project library.\n"
            f"Generation time: {format_duration(result['elapsed_seconds'])}\n"
            f"Output length: {format_duration(result['duration_seconds'])}"
        )
        return (
            result["audio_path"],
            result.get("spectrogram_path"),
            status,
        ) + refresh

    def render_preview(*args):
        return render_now(*args[:5], "preview", *args[5:])

    def render_final(*args):
        return render_now(*args[:5], "final", *args[5:])

    def submit_batch(
        project_id: int,
        reference_id: int,
        style_id: int,
        mode: str,
        use_style_prompt: bool,
        context_notes: str,
        batch_scripts: str,
    ):
        scripts = [chunk.strip() for chunk in re.split(r"\n\s*\n", batch_scripts) if chunk.strip()]
        if not scripts:
            raise gr.Error("Add one or more scripts separated by blank lines.")
        created = []
        for index, script in enumerate(scripts, start=1):
            request = build_request(
                project_id,
                reference_id,
                style_id,
                f"Batch {index:02d}",
                script,
                mode,
                use_style_prompt,
                context_notes,
                0.0,
                0,
                False,
                False,
            )
            created.append(service.enqueue_generation(request))
        refresh = project_updates(project_id)
        status = f"Queued {len(created)} jobs. The worker will process them one at a time."
        return (status,) + refresh

    def compare_takes(take_a_id: int | None, take_b_id: int | None):
        audio_a = None
        audio_b = None
        metadata_a = {}
        metadata_b = {}
        if take_a_id:
            asset_a = service.store.get_audio_asset(int(take_a_id))
            audio_a = asset_a["path"]
            metadata_a = asset_a
        if take_b_id:
            asset_b = service.store.get_audio_asset(int(take_b_id))
            audio_b = asset_b["path"]
            metadata_b = asset_b
        return audio_a, json.dumps(metadata_a, indent=2), audio_b, json.dumps(metadata_b, indent=2)

    def export_take(asset_id: int | None):
        if not asset_id:
            raise gr.Error("Choose a take from the library first.")
        bundle = service.export_asset_bundle(int(asset_id))
        return bundle, f"Export bundle created at {bundle}"

    def save_settings(profile_name: str, asr_backend: str, idle_unload_seconds: int):
        service.set_runtime_profile(profile_name)
        service.set_asr_backend(asr_backend)
        service.set_idle_unload_seconds(idle_unload_seconds)
        return json.dumps(service.system_profile(), indent=2), "Settings saved."

    def save_rule(project_id: int, source: str, replacement: str):
        if not project_id:
            raise gr.Error("Select a project first.")
        if not source.strip() or not replacement.strip():
            raise gr.Error("Enter both a source phrase and a replacement.")
        service.save_pronunciation_rule(project_id, source, replacement)
        refresh = project_updates(project_id)
        return ("Pronunciation rule saved.",) + refresh

    def cancel_job(project_id: int, job_id: int | None):
        if not job_id:
            raise gr.Error("Choose a queued job to cancel.")
        service.cancel_job(int(job_id))
        refresh = project_updates(project_id)
        return ("Cancellation requested.",) + refresh

    with gr.Blocks(
        title=TITLE,
        delete_cache=(3600, 7200),
        analytics_enabled=False,
    ) as app:
        with gr.Column(elem_classes=["studio-shell"]):
            gr.HTML(HERO_HTML)

            with gr.Row():
                with gr.Column(scale=7, elem_classes=["studio-panel"]):
                    gr.Markdown("### Active project")
                    active_project = gr.Dropdown(label="Project", choices=[], interactive=True)
                    project_summary = gr.Markdown()
                with gr.Column(scale=5, elem_classes=["studio-panel"]):
                    gr.Markdown("### New project")
                    project_name = gr.Textbox(label="Project name", placeholder="Client delivery pack")
                    project_description = gr.Textbox(label="Description", lines=3)
                    with gr.Row():
                        create_project_btn = gr.Button("Create project", elem_classes=["studio-secondary"])
                        refresh_btn = gr.Button("Refresh library", elem_classes=["studio-secondary"])
                    create_project_status = gr.Textbox(label="Project status", interactive=False)

            with gr.Tabs():
                with gr.Tab("Create"):
                    gr.Markdown("Build references, save styles, estimate runtime, then render a quick preview or a final take.", elem_classes=["studio-tab-note"])
                    with gr.Row():
                        with gr.Column(scale=5, elem_classes=["studio-panel"]):
                            gr.Markdown("### Reference voice")
                            reference_name = gr.Textbox(label="Reference name", value="Lead Voice")
                            reference_audio = gr.Audio(label="Reference audio", type="filepath", sources=["upload", "microphone"])
                            reference_text = gr.Textbox(label="Reference transcript", lines=4, placeholder="Leave blank to transcribe locally.")
                            reference_consent = gr.Checkbox(
                                label="I have the right to use this voice sample",
                                value=False,
                            )
                            save_reference_btn = gr.Button("Analyze and save reference", elem_classes=["studio-secondary"])
                            reference_status = gr.Textbox(label="Reference status", lines=5, interactive=False)
                            reference_choice = gr.Dropdown(label="Saved references", choices=[])

                        with gr.Column(scale=5, elem_classes=["studio-panel"]):
                            gr.Markdown("### Style prompt")
                            style_name = gr.Textbox(label="Style name", value="Calm Narrator")
                            style_audio = gr.Audio(label="Style prompt audio", type="filepath", sources=["upload", "microphone"])
                            style_text = gr.Textbox(label="Style transcript", lines=3, placeholder="Leave blank to transcribe locally.")
                            context_notes = gr.Textbox(
                                label="Delivery context",
                                lines=3,
                                placeholder="Examples: calm demo, energetic trailer, soft whisper, urgent support update.",
                            )
                            save_style_btn = gr.Button("Analyze and save style", elem_classes=["studio-secondary"])
                            style_status = gr.Textbox(label="Style status", lines=5, interactive=False)
                            style_choice = gr.Dropdown(label="Saved styles", choices=[])

                    with gr.Row():
                        with gr.Column(scale=7, elem_classes=["studio-panel"]):
                            gr.Markdown("### Script and render recipe")
                            preset_choice = gr.Dropdown(label="Starter preset", choices=list(SCRIPT_PRESETS.keys()), value="Product demo")
                            load_preset_btn = gr.Button("Load preset", elem_classes=["studio-secondary"])
                            render_name = gr.Textbox(label="Take name", value="Final Render")
                            render_text = gr.Textbox(label="Script", lines=10, value=SCRIPT_PRESETS["Product demo"])
                            use_style_prompt = gr.Checkbox(label="Use style prompt during generation", value=True)
                            render_mode = gr.Radio(label="Render mode", choices=["preview", "final"], value="final")
                            with gr.Accordion("Advanced controls", open=False):
                                speed = gr.Slider(label="Speed override", minimum=0.0, maximum=1.5, step=0.05, value=0.0)
                                nfe_step = gr.Slider(label="NFE override", minimum=0, maximum=64, step=2, value=0)
                                remove_silence = gr.Checkbox(label="Trim long silence in output", value=False)
                                render_spectrogram = gr.Checkbox(label="Render spectrogram", value=False)
                            with gr.Row():
                                estimate_btn = gr.Button("Estimate runtime", elem_classes=["studio-secondary"])
                                preview_btn = gr.Button("Quick preview", elem_classes=["studio-primary"])
                                final_btn = gr.Button("Final render", elem_classes=["studio-primary"])
                            render_status = gr.Textbox(
                                label="Render status",
                                lines=6,
                                interactive=False,
                                value="Select a project, save a clean reference, then generate a quick preview or a final render.",
                            )

                        with gr.Column(scale=5, elem_classes=["studio-panel"]):
                            gr.Markdown("### Latest output")
                            output_audio = gr.Audio(label="Generated take", type="filepath")
                            output_spectrogram = gr.Image(label="Spectrogram")
                            system_profile_create = gr.JSON(label="Runtime profile")

                with gr.Tab("Library"):
                    gr.Markdown("Review what is saved in the active project, compare takes, and export bundles for sharing.", elem_classes=["studio-tab-note"])
                    with gr.Row():
                        with gr.Column(scale=6, elem_classes=["studio-panel"]):
                            references_table = gr.Dataframe(
                                headers=["ID", "Name", "Duration", "ASR", "Warnings"],
                                datatype=["number", "str", "str", "str", "str"],
                                interactive=False,
                                label="Reference library",
                            )
                            styles_table = gr.Dataframe(
                                headers=["ID", "Name", "Duration", "Speed", "Keywords"],
                                datatype=["number", "str", "str", "str", "str"],
                                interactive=False,
                                label="Style library",
                            )
                        with gr.Column(scale=6, elem_classes=["studio-panel"]):
                            assets_table = gr.Dataframe(
                                headers=["ID", "Kind", "Label", "Duration", "Created"],
                                datatype=["number", "str", "str", "str", "str"],
                                interactive=False,
                                label="Saved takes",
                            )
                            jobs_table = gr.Dataframe(
                                headers=["ID", "Name", "Status", "Updated", "Notes"],
                                datatype=["number", "str", "str", "str", "str"],
                                interactive=False,
                                label="Recent jobs",
                            )
                            rules_table = gr.Dataframe(
                                headers=["ID", "Source", "Replacement", "Updated"],
                                datatype=["number", "str", "str", "str"],
                                interactive=False,
                                label="Pronunciation rules",
                            )

                    with gr.Row():
                        with gr.Column(scale=6, elem_classes=["studio-panel"]):
                            gr.Markdown("### A/B compare")
                            take_a = gr.Dropdown(label="Take A", choices=[])
                            take_b = gr.Dropdown(label="Take B", choices=[])
                            compare_btn = gr.Button("Load comparison", elem_classes=["studio-secondary"])
                            compare_audio_a = gr.Audio(label="Take A audio", type="filepath")
                            compare_meta_a = gr.Code(label="Take A metadata", language="json")
                        with gr.Column(scale=6, elem_classes=["studio-panel"]):
                            export_take_choice = gr.Dropdown(label="Export a take", choices=[])
                            export_btn = gr.Button("Create share bundle", elem_classes=["studio-secondary"])
                            export_status = gr.Textbox(label="Export status", interactive=False)
                            export_file = gr.File(label="Bundle index", interactive=False)
                            compare_audio_b = gr.Audio(label="Take B audio", type="filepath")
                            compare_meta_b = gr.Code(label="Take B metadata", language="json")

                with gr.Tab("Batch"):
                    gr.Markdown("Queue several scripts against the same voice setup. Jobs run one at a time to protect the M4 Air.", elem_classes=["studio-tab-note"])
                    with gr.Row():
                        with gr.Column(scale=6, elem_classes=["studio-panel"]):
                            batch_reference = gr.Dropdown(label="Batch reference", choices=[])
                            batch_style = gr.Dropdown(label="Batch style", choices=[])
                            batch_mode = gr.Radio(label="Batch mode", choices=["preview", "final"], value="final")
                            batch_use_style = gr.Checkbox(label="Use saved style prompt", value=True)
                            batch_context = gr.Textbox(label="Shared delivery context", lines=3)
                            batch_scripts = gr.Textbox(
                                label="Batch scripts",
                                lines=12,
                                placeholder="Add one script per block, separated by a blank line.",
                            )
                            batch_submit_btn = gr.Button("Queue batch", elem_classes=["studio-primary"])
                            batch_status = gr.Textbox(label="Batch status", interactive=False)
                        with gr.Column(scale=6, elem_classes=["studio-panel"]):
                            cancel_job_choice = gr.Dropdown(label="Cancelable jobs", choices=[])
                            cancel_job_btn = gr.Button("Cancel selected job", elem_classes=["studio-secondary"])
                            batch_jobs_table = gr.Dataframe(
                                headers=["ID", "Name", "Status", "Updated", "Notes"],
                                datatype=["number", "str", "str", "str", "str"],
                                interactive=False,
                                label="Queue monitor",
                            )

                with gr.Tab("Settings"):
                    gr.Markdown("Tune how aggressive the local runtime should be and add pronunciation rules for the active project.", elem_classes=["studio-tab-note"])
                    with gr.Row():
                        with gr.Column(scale=5, elem_classes=["studio-panel"]):
                            profile_choice = gr.Dropdown(
                                label="Runtime profile",
                                choices=[(profile.label, profile.name) for profile in PROFILE_MAP.values()],
                                value=service.get_runtime_profile_name(),
                            )
                            asr_backend = gr.Dropdown(
                                label="ASR backend",
                                choices=[("Auto", "auto"), ("MLX Whisper", "mlx_whisper"), ("Transformers Whisper", "transformers")],
                                value=service.get_asr_backend(),
                            )
                            idle_unload_seconds = gr.Number(
                                label="Idle unload seconds",
                                value=service.get_idle_unload_seconds(),
                                precision=0,
                            )
                            save_settings_btn = gr.Button("Save runtime settings", elem_classes=["studio-secondary"])
                            settings_status = gr.Textbox(label="Settings status", interactive=False)
                        with gr.Column(scale=7, elem_classes=["studio-panel"]):
                            gr.Markdown("### Pronunciation dictionary")
                            rule_source = gr.Textbox(label="Source phrase", placeholder="GPU")
                            rule_replacement = gr.Textbox(label="Replacement phrase", placeholder="gee pee you")
                            save_rule_btn = gr.Button("Save pronunciation rule", elem_classes=["studio-secondary"])
                            rule_status = gr.Textbox(label="Rule status", interactive=False)
                            system_profile_settings = gr.Code(label="System profile", language="json")
                            helper_markdown = gr.Markdown(
                                "Launch locally with `f5-tts_voice-studio`, or mount the API + studio together with `f5-tts_studio-server`."
                            )

            load_preset_btn.click(load_preset, inputs=[preset_choice], outputs=[render_text])
            refresh_btn.click(
                project_updates,
                inputs=[active_project],
                outputs=[
                    active_project,
                    project_summary,
                    reference_choice,
                    style_choice,
                    batch_reference,
                    batch_style,
                    references_table,
                    styles_table,
                    assets_table,
                    jobs_table,
                    batch_jobs_table,
                    rules_table,
                    take_a,
                    take_b,
                    export_take_choice,
                    cancel_job_choice,
                    system_profile_create,
                ],
            )
            active_project.change(
                project_updates,
                inputs=[active_project],
                outputs=[
                    active_project,
                    project_summary,
                    reference_choice,
                    style_choice,
                    batch_reference,
                    batch_style,
                    references_table,
                    styles_table,
                    assets_table,
                    jobs_table,
                    batch_jobs_table,
                    rules_table,
                    take_a,
                    take_b,
                    export_take_choice,
                    cancel_job_choice,
                    system_profile_create,
                ],
            )
            create_project_btn.click(
                create_project,
                inputs=[project_name, project_description, active_project],
                outputs=[
                    active_project,
                    project_summary,
                    reference_choice,
                    style_choice,
                    batch_reference,
                    batch_style,
                    references_table,
                    styles_table,
                    assets_table,
                    jobs_table,
                    batch_jobs_table,
                    rules_table,
                    take_a,
                    take_b,
                    export_take_choice,
                    cancel_job_choice,
                    system_profile_create,
                    create_project_status,
                ],
            )
            save_reference_btn.click(
                save_reference,
                inputs=[active_project, reference_consent, reference_name, reference_audio, reference_text],
                outputs=[
                    reference_text,
                    reference_status,
                    active_project,
                    project_summary,
                    reference_choice,
                    style_choice,
                    batch_reference,
                    batch_style,
                    references_table,
                    styles_table,
                    assets_table,
                    jobs_table,
                    batch_jobs_table,
                    rules_table,
                    take_a,
                    take_b,
                    export_take_choice,
                    cancel_job_choice,
                    system_profile_create,
                ],
                concurrency_limit=1,
                concurrency_id="studio_compute",
            )
            save_style_btn.click(
                save_style,
                inputs=[active_project, style_name, style_audio, style_text, context_notes, render_text],
                outputs=[
                    style_text,
                    style_status,
                    active_project,
                    project_summary,
                    reference_choice,
                    style_choice,
                    batch_reference,
                    batch_style,
                    references_table,
                    styles_table,
                    assets_table,
                    jobs_table,
                    batch_jobs_table,
                    rules_table,
                    take_a,
                    take_b,
                    export_take_choice,
                    cancel_job_choice,
                    system_profile_create,
                ],
                concurrency_limit=1,
                concurrency_id="studio_compute",
            )
            estimate_btn.click(
                estimate_render,
                inputs=[
                    active_project,
                    reference_choice,
                    style_choice,
                    render_name,
                    render_text,
                    render_mode,
                    use_style_prompt,
                    context_notes,
                    speed,
                    nfe_step,
                ],
                outputs=[render_status],
                concurrency_limit=1,
                concurrency_id="studio_compute",
            )
            preview_btn.click(
                render_preview,
                inputs=[
                    active_project,
                    reference_choice,
                    style_choice,
                    render_name,
                    render_text,
                    use_style_prompt,
                    context_notes,
                    speed,
                    nfe_step,
                    remove_silence,
                    render_spectrogram,
                ],
                outputs=[
                    output_audio,
                    output_spectrogram,
                    render_status,
                    active_project,
                    project_summary,
                    reference_choice,
                    style_choice,
                    batch_reference,
                    batch_style,
                    references_table,
                    styles_table,
                    assets_table,
                    jobs_table,
                    batch_jobs_table,
                    rules_table,
                    take_a,
                    take_b,
                    export_take_choice,
                    cancel_job_choice,
                    system_profile_create,
                ],
                concurrency_limit=1,
                concurrency_id="studio_compute",
            )
            final_btn.click(
                render_final,
                inputs=[
                    active_project,
                    reference_choice,
                    style_choice,
                    render_name,
                    render_text,
                    use_style_prompt,
                    context_notes,
                    speed,
                    nfe_step,
                    remove_silence,
                    render_spectrogram,
                ],
                outputs=[
                    output_audio,
                    output_spectrogram,
                    render_status,
                    active_project,
                    project_summary,
                    reference_choice,
                    style_choice,
                    batch_reference,
                    batch_style,
                    references_table,
                    styles_table,
                    assets_table,
                    jobs_table,
                    batch_jobs_table,
                    rules_table,
                    take_a,
                    take_b,
                    export_take_choice,
                    cancel_job_choice,
                    system_profile_create,
                ],
                concurrency_limit=1,
                concurrency_id="studio_compute",
            )
            batch_submit_btn.click(
                submit_batch,
                inputs=[active_project, batch_reference, batch_style, batch_mode, batch_use_style, batch_context, batch_scripts],
                outputs=[
                    batch_status,
                    active_project,
                    project_summary,
                    reference_choice,
                    style_choice,
                    batch_reference,
                    batch_style,
                    references_table,
                    styles_table,
                    assets_table,
                    jobs_table,
                    batch_jobs_table,
                    rules_table,
                    take_a,
                    take_b,
                    export_take_choice,
                    cancel_job_choice,
                    system_profile_create,
                ],
            )
            cancel_job_btn.click(
                cancel_job,
                inputs=[active_project, cancel_job_choice],
                outputs=[
                    batch_status,
                    active_project,
                    project_summary,
                    reference_choice,
                    style_choice,
                    batch_reference,
                    batch_style,
                    references_table,
                    styles_table,
                    assets_table,
                    jobs_table,
                    batch_jobs_table,
                    rules_table,
                    take_a,
                    take_b,
                    export_take_choice,
                    cancel_job_choice,
                    system_profile_create,
                ],
            )
            compare_btn.click(
                compare_takes,
                inputs=[take_a, take_b],
                outputs=[compare_audio_a, compare_meta_a, compare_audio_b, compare_meta_b],
            )
            export_btn.click(export_take, inputs=[export_take_choice], outputs=[export_file, export_status])
            save_settings_btn.click(
                save_settings,
                inputs=[profile_choice, asr_backend, idle_unload_seconds],
                outputs=[system_profile_settings, settings_status],
            )
            save_rule_btn.click(
                save_rule,
                inputs=[active_project, rule_source, rule_replacement],
                outputs=[
                    rule_status,
                    active_project,
                    project_summary,
                    reference_choice,
                    style_choice,
                    batch_reference,
                    batch_style,
                    references_table,
                    styles_table,
                    assets_table,
                    jobs_table,
                    batch_jobs_table,
                    rules_table,
                    take_a,
                    take_b,
                    export_take_choice,
                    cancel_job_choice,
                    system_profile_create,
                ],
            )
            app.load(
                project_updates,
                outputs=[
                    active_project,
                    project_summary,
                    reference_choice,
                    style_choice,
                    batch_reference,
                    batch_style,
                    references_table,
                    styles_table,
                    assets_table,
                    jobs_table,
                    batch_jobs_table,
                    rules_table,
                    take_a,
                    take_b,
                    export_take_choice,
                    cancel_job_choice,
                    system_profile_create,
                ],
            )
            app.load(service.warm_profile_if_needed, outputs=[render_status])
            app.load(lambda: json.dumps(service.system_profile(), indent=2), outputs=[system_profile_settings])
            app.unload(service.maybe_unload_idle_engine)

    return app


def build_studio_app():
    return create_studio_app()
