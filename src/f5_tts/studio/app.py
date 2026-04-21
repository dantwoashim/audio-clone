from __future__ import annotations

import html
import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

import gradio as gr

from f5_tts.studio.diagnostics import reference_quality_breakdown
from f5_tts.studio.profiles import PROFILE_MAP
from f5_tts.studio.runtime import format_duration, get_service
from f5_tts.studio.schemas import GenerationRequest


TITLE = "Voice Studio Pro"
VOICE_PROJECT_ROOT = Path(os.environ.get("VOICE_PROJECT_ROOT", Path.home() / "voice-project"))


def studio_allowed_paths(service=None) -> list[str]:
    allowed = set()
    if service is not None:
        allowed.add(str(service.paths.projects))
        allowed.add(str(service.paths.exports))
        allowed.add(str(service.paths.incoming))
    voice_output_root = VOICE_PROJECT_ROOT / "output"
    if voice_output_root.exists():
        allowed.add(str(voice_output_root))
    return sorted(allowed)

APP_CSS = """
:root {
  --studio-bg: #efe3cf;
  --studio-paper: rgba(252, 247, 239, 0.86);
  --studio-paper-strong: rgba(255, 251, 245, 0.96);
  --studio-ink: #161a1d;
  --studio-muted: #666155;
  --studio-line: rgba(22, 26, 29, 0.12);
  --studio-accent: #0f7b63;
  --studio-accent-2: #b04d2f;
  --studio-shadow: 0 28px 70px rgba(63, 47, 24, 0.09);
}

.gradio-container {
  color-scheme: light;
  --body-background-fill: transparent;
  --background-fill-primary: rgba(255, 250, 242, 0.88);
  --background-fill-secondary: rgba(255, 248, 240, 0.76);
  --block-background-fill: rgba(255, 250, 242, 0.9);
  --block-border-color: rgba(22, 26, 29, 0.1);
  --block-label-background-fill: rgba(255, 250, 242, 0.98);
  --block-label-border-color: rgba(22, 26, 29, 0.1);
  --block-label-text-color: var(--studio-accent);
  --block-title-text-color: var(--studio-ink);
  --input-background-fill: rgba(255, 250, 242, 0.96);
  --input-border-color: rgba(22, 26, 29, 0.12);
  --input-placeholder-color: #827b70;
  --body-text-color: var(--studio-ink);
  --body-text-color-subdued: var(--studio-muted);
  --button-secondary-background-fill: rgba(255, 251, 244, 0.9);
  --button-secondary-border-color: rgba(22, 26, 29, 0.12);
  --button-secondary-text-color: var(--studio-ink);
  --checkbox-label-text-color: var(--studio-ink);
  --link-text-color: var(--studio-accent);
  background:
    radial-gradient(circle at 0% 0%, rgba(176, 77, 47, 0.14), transparent 28%),
    radial-gradient(circle at 100% 0%, rgba(15, 123, 99, 0.16), transparent 24%),
    linear-gradient(180deg, #f4ecde 0%, #eadfca 100%);
  color: var(--studio-ink);
}

.gradio-container, .gradio-container * {
  font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
}

.studio-shell {
  max-width: 1440px;
  margin: 0 auto;
  padding: 24px 18px 54px;
}

.studio-hero,
.studio-panel,
.studio-page-intro,
.studio-overview,
.studio-trained-overview {
  border: 1px solid var(--studio-line);
  border-radius: 30px;
  background: var(--studio-paper);
  box-shadow: var(--studio-shadow);
  backdrop-filter: blur(12px);
}

.studio-hero {
  overflow: hidden;
  padding: 30px;
  margin-bottom: 18px;
  background:
    linear-gradient(135deg, rgba(251, 247, 240, 0.95), rgba(245, 234, 217, 0.82)),
    radial-gradient(circle at top right, rgba(15, 123, 99, 0.18), transparent 32%);
}

.studio-hero-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.3fr) minmax(320px, 0.7fr);
  gap: 18px;
  align-items: stretch;
}

.studio-eyebrow {
  margin: 0 0 10px;
  color: var(--studio-accent);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}

.studio-brand {
  margin: 0;
  font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
  font-size: clamp(3.2rem, 6vw, 5.4rem);
  line-height: 0.92;
  letter-spacing: -0.055em;
  color: var(--studio-ink);
  text-wrap: balance;
}

.studio-hero-copy {
  max-width: 760px;
  margin: 16px 0 0;
  color: var(--studio-muted);
  font-size: 1.02rem;
  line-height: 1.72;
}

.studio-hero-aside {
  display: grid;
  gap: 14px;
}

.studio-hero-card {
  min-height: 120px;
  border: 1px solid rgba(22, 26, 29, 0.1);
  border-radius: 24px;
  padding: 18px 20px;
  background: var(--studio-paper-strong);
}

.studio-hero-card strong {
  display: block;
  margin: 0 0 8px;
  font-size: 0.95rem;
  letter-spacing: 0.02em;
  color: var(--studio-ink);
}

.studio-hero-card p,
.studio-hero-card ul {
  margin: 0;
  color: var(--studio-muted);
  line-height: 1.55;
}

.studio-hero-card ul {
  padding-left: 18px;
}

.studio-panel {
  padding: 20px;
}

.studio-panel,
.studio-panel h3,
.studio-panel h4,
.studio-panel strong,
.studio-page-intro h3,
.studio-overview h2,
.studio-trained-overview h2 {
  color: var(--studio-ink);
}

.studio-panel h3,
.studio-panel h4,
.studio-panel p:last-child {
  margin-bottom: 0;
}

.studio-warning,
.studio-ok {
  margin: 14px 0 0;
  padding: 12px 14px;
  border-radius: 16px;
  font-weight: 600;
}

.studio-warning {
  color: #7f2917;
  background: rgba(176, 77, 47, 0.12);
  border: 1px solid rgba(176, 77, 47, 0.22);
}

.studio-ok {
  color: #0d5d4a;
  background: rgba(15, 123, 99, 0.1);
  border: 1px solid rgba(15, 123, 99, 0.2);
}

.studio-shell label,
.studio-shell legend,
.studio-shell .gradio-markdown p,
.studio-shell .gradio-markdown strong,
.studio-shell .gradio-markdown li {
  color: var(--studio-ink) !important;
}

.studio-shell input,
.studio-shell textarea,
.studio-shell select,
.studio-shell fieldset,
.studio-shell [data-testid="block"],
.studio-shell .block,
.studio-shell .icon-button-wrapper,
.studio-shell .form,
.studio-shell .gradio-dropdown,
.studio-shell .gradio-textbox,
.studio-shell .gradio-number,
.studio-shell .gradio-radio,
.studio-shell .gradio-audio,
.studio-shell .gradio-image,
.studio-shell .gradio-code,
.studio-shell .gradio-dataframe {
  color: var(--studio-ink) !important;
}

.studio-shell input,
.studio-shell textarea,
.studio-shell fieldset,
.studio-shell [data-testid="block"],
.studio-shell .block,
.studio-shell .icon-button-wrapper,
.studio-shell .form,
.studio-shell .gradio-dropdown,
.studio-shell .gradio-textbox,
.studio-shell .gradio-number,
.studio-shell .gradio-code {
  background: rgba(255, 250, 242, 0.88) !important;
  border-color: rgba(22, 26, 29, 0.1) !important;
}

.studio-control-stack,
.studio-stack {
  display: grid;
  gap: 14px;
}

.studio-overview,
.studio-trained-overview {
  padding: 22px;
}

.studio-overview h2,
.studio-trained-overview h2 {
  margin: 8px 0 10px;
  font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
  font-size: clamp(2rem, 3vw, 3rem);
  line-height: 0.98;
  letter-spacing: -0.04em;
}

.studio-overview-copy,
.studio-trained-copy {
  max-width: 760px;
}

.studio-overview-copy p,
.studio-trained-copy p {
  margin: 0;
  color: var(--studio-muted);
  line-height: 1.66;
}

.studio-stat-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 18px;
}

.studio-stat {
  border-radius: 22px;
  border: 1px solid rgba(22, 26, 29, 0.08);
  background: var(--studio-paper-strong);
  padding: 16px;
}

.studio-stat-label {
  display: block;
  color: var(--studio-muted);
  font-size: 0.76rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.studio-stat-value {
  display: block;
  margin-top: 7px;
  font-size: 1.18rem;
  font-weight: 700;
  line-height: 1.2;
}

.studio-mini-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 18px;
}

.studio-mini {
  border-top: 1px solid rgba(22, 26, 29, 0.08);
  padding-top: 12px;
}

.studio-mini strong {
  display: block;
  font-size: 0.82rem;
}

.studio-mini span {
  display: block;
  color: var(--studio-muted);
  font-size: 0.92rem;
  margin-top: 4px;
}

.studio-page-intro {
  padding: 20px 22px;
  margin-bottom: 14px;
}

.studio-page-intro h3 {
  margin: 6px 0 8px;
  font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
  font-size: clamp(1.7rem, 2.6vw, 2.5rem);
  line-height: 1;
  letter-spacing: -0.04em;
}

.studio-page-intro p {
  margin: 0;
  color: var(--studio-muted);
  line-height: 1.65;
}

.studio-tabs .tab-nav {
  gap: 10px;
  padding: 0 0 16px;
  background: transparent;
  justify-content: flex-start;
}

.studio-tabs .tab-nav button {
  border-radius: 999px !important;
  border: 1px solid rgba(22, 26, 29, 0.1) !important;
  background: rgba(255, 250, 242, 0.78) !important;
  padding: 12px 18px !important;
  color: var(--studio-ink) !important;
  font-weight: 700 !important;
  letter-spacing: 0.01em !important;
}

.studio-tabs .tab-nav button.selected {
  background: linear-gradient(135deg, var(--studio-accent), #155f8b) !important;
  color: #ffffff !important;
  box-shadow: 0 14px 30px rgba(15, 123, 99, 0.24);
}

.studio-primary button {
  min-height: 52px;
  border-radius: 18px !important;
  background: linear-gradient(135deg, var(--studio-accent), #155f8b) !important;
  color: #ffffff !important;
  box-shadow: 0 18px 34px rgba(15, 123, 99, 0.24);
}

.studio-secondary button {
  min-height: 48px;
  border-radius: 18px !important;
  background: rgba(255, 251, 244, 0.9) !important;
  border: 1px solid rgba(22, 26, 29, 0.12) !important;
  color: var(--studio-ink) !important;
}

.studio-audio-card audio {
  width: 100%;
}

.studio-muted-note {
  color: var(--studio-muted);
  font-size: 0.94rem;
  margin: 0;
}

.studio-panel .gradio-dataframe,
.studio-panel .gr-code,
.studio-panel .gr-json {
  border-radius: 20px;
  overflow: hidden;
}

.studio-shell .gradio-audio,
.studio-shell .gradio-image,
.studio-shell .gradio-code,
.studio-shell .gradio-dataframe {
  background: rgba(255, 250, 242, 0.78) !important;
  border: 1px solid rgba(22, 26, 29, 0.1) !important;
  border-radius: 20px !important;
}

.studio-shell .icon-button,
.studio-shell button[aria-label="Clear"] {
  opacity: 0.85;
}

.studio-shell label[data-testid="block-label"],
.studio-shell label[data-testid="block-label"] span {
  color: var(--studio-accent) !important;
}

.studio-shell table,
.studio-shell thead,
.studio-shell tbody,
.studio-shell tr,
.studio-shell th,
.studio-shell td {
  background: rgba(255, 250, 242, 0.96) !important;
  color: var(--studio-ink) !important;
  border-color: rgba(22, 26, 29, 0.1) !important;
}

.studio-shell .error,
.studio-shell .gradio-container .toast-wrap .toast-body {
  color: #fff;
}

.gradio-container footer,
.gradio-container .built-with-gradio,
.gradio-container .settings-trigger {
  display: none !important;
}

@media (max-width: 1100px) {
  .studio-hero-grid,
  .studio-stat-grid,
  .studio-mini-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 720px) {
  .studio-shell {
    padding: 18px 12px 34px;
  }

  .studio-hero,
  .studio-panel,
  .studio-page-intro,
  .studio-overview,
  .studio-trained-overview {
    border-radius: 24px;
  }

  .studio-brand {
    font-size: 2.7rem;
  }
}
"""

HERO_HTML = f"""
<section class="studio-hero">
  <div class="studio-hero-grid">
    <div>
      <p class="studio-eyebrow">Local-first cloned voice workflow</p>
      <h1 class="studio-brand">{TITLE}</h1>
      <p class="studio-hero-copy">
        Train, direct, and audition a voice on-device without turning the rest of your M4 Air into collateral damage.
        The studio is tuned for saved identity references, reusable delivery prompts, checkpoint-aware inference, and
        one clean workspace at a time.
      </p>
    </div>
    <div class="studio-hero-aside">
      <div class="studio-hero-card">
        <strong>One studio, multiple lanes</strong>
        <p>Create fresh takes, repair existing recordings, audition trained snapshots, diagnose quality, then package everything from one place instead of spelunking through raw forms.</p>
      </div>
      <div class="studio-hero-card">
        <strong>What changed</strong>
        <ul>
          <li>Clearer creation flow</li>
          <li>Alignment-first edit page</li>
          <li>Dedicated trained-voice page</li>
          <li>Reference coach and diagnostics</li>
          <li>Checkpoint-aware runtime controls</li>
          <li>Safer sharing and upload limits</li>
        </ul>
      </div>
    </div>
  </div>
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


def _dropdown_value(choices: list[tuple[str, int]], index: int = 0) -> int | None:
    return choices[index][1] if len(choices) > index else None


def _human_variant_name(name: str) -> str:
    labels = {
        "base": "Base model",
        "ft_noema": "Finetuned snapshot · no EMA",
        "ft_ema": "Finetuned snapshot · EMA",
    }
    return labels.get(name, name.replace("_", " ").strip().title())


def _current_voice_label(system_profile: dict) -> str:
    checkpoint_path = system_profile.get("checkpoint_path")
    if not checkpoint_path:
        return "Base model"
    checkpoint_name = Path(checkpoint_path).name
    lowered = checkpoint_name.lower()
    if "noema" in lowered:
        return "Finetuned snapshot · no EMA"
    if "ema" in lowered:
        return "Finetuned snapshot · EMA"
    return checkpoint_name


def _similarity_percent(expected: str, transcript: str) -> str:
    if not expected or not transcript:
        return "n/a"
    ratio = SequenceMatcher(None, expected.lower().strip(), transcript.lower().strip()).ratio()
    return f"{ratio * 100:.0f}%"


def _page_intro_html(kicker: str, title: str, body: str) -> str:
    return f"""
    <section class="studio-page-intro">
      <p class="studio-eyebrow">{html.escape(kicker)}</p>
      <h3>{html.escape(title)}</h3>
      <p>{html.escape(body)}</p>
    </section>
    """


def _project_overview_html(project: dict, system_profile: dict) -> str:
    description = project.get("description") or "No project description yet. Add one when this becomes a real delivery lane."
    checkpoint_label = _current_voice_label(system_profile)
    stats = [
        ("Saved references", str(len(project["references"]))),
        ("Style prompts", str(len(project["styles"]))),
        ("Recorded takes", str(len(project["assets"]))),
        ("Queued jobs", str(len([job for job in project["jobs"] if job["status"] in {'queued', 'running'}]))),
    ]
    stat_html = "".join(
        f"""
        <div class="studio-stat">
          <span class="studio-stat-label">{html.escape(label)}</span>
          <span class="studio-stat-value">{html.escape(value)}</span>
        </div>
        """
        for label, value in stats
    )
    return f"""
    <section class="studio-overview">
      <div class="studio-overview-copy">
        <p class="studio-eyebrow">Project cockpit</p>
        <h2>{html.escape(project['name'])}</h2>
        <p>{html.escape(description)}</p>
      </div>
      <div class="studio-stat-grid">{stat_html}</div>
      <div class="studio-mini-grid">
        <div class="studio-mini">
          <strong>Active voice</strong>
          <span>{html.escape(checkpoint_label)}</span>
        </div>
        <div class="studio-mini">
          <strong>Device</strong>
          <span>{html.escape(str(system_profile.get('device', 'unknown')).upper())}</span>
        </div>
        <div class="studio-mini">
          <strong>Runtime profile</strong>
          <span>{html.escape(system_profile.get('profile_label', 'Balanced'))}</span>
        </div>
      </div>
    </section>
    """


def _runtime_snapshot_html(system_profile: dict) -> str:
    checkpoint_label = _current_voice_label(system_profile)
    checkpoint_path = system_profile.get("checkpoint_path")
    checkpoint_note = Path(checkpoint_path).name if checkpoint_path else "Using the shipped base checkpoint."
    asr_model = system_profile.get("asr_model") or "auto"
    warm_note = (
        f"<div class=\"studio-mini\"><strong>Warm-up</strong><span>{html.escape(system_profile['last_warm_error'])}</span></div>"
        if system_profile.get("last_warm_error")
        else ""
    )
    return f"""
    <section class="studio-trained-overview">
      <div class="studio-trained-copy">
        <p class="studio-eyebrow">Render runtime</p>
        <h2>{html.escape(checkpoint_label)}</h2>
        <p>{html.escape(checkpoint_note)}</p>
      </div>
      <div class="studio-mini-grid">
        <div class="studio-mini">
          <strong>Device</strong>
          <span>{html.escape(str(system_profile.get('device', 'unknown')).upper())}</span>
        </div>
        <div class="studio-mini">
          <strong>Queue</strong>
          <span>{html.escape(str(system_profile.get('queue_depth', 0)))} waiting</span>
        </div>
        <div class="studio-mini">
          <strong>Engine</strong>
          <span>{"Warm" if system_profile.get("engine_loaded") else "Cold"}</span>
        </div>
        <div class="studio-mini">
          <strong>ASR</strong>
          <span>{html.escape(asr_model)}</span>
        </div>
        {warm_note}
        </div>
    </section>
    """


def _sharing_snapshot_html(system_profile: dict) -> str:
    warning = system_profile.get("sharing_warning")
    warning_html = (
        f"<p class=\"studio-warning\">{html.escape(warning)}</p>"
        if warning
        else "<p class=\"studio-ok\">Local-only mode. Nothing is exposed beyond this machine right now.</p>"
    )
    public_url = system_profile.get("public_url")
    public_url_html = (
        f"<div class=\"studio-mini\"><strong>Public URL</strong><span>{html.escape(public_url)}</span></div>"
        if public_url
        else ""
    )
    return f"""
    <section class="studio-trained-overview">
      <div class="studio-trained-copy">
        <p class="studio-eyebrow">Sharing safety</p>
        <h2>{'Protected share' if system_profile.get('auth_enabled') else 'Local by default'}</h2>
        <p>
          Upload limit: {html.escape(str(system_profile.get('upload_limit_mb', 64)))} MB.
          Auth mode: {html.escape(system_profile.get('auth_mode', 'none'))}.
        </p>
        {warning_html}
      </div>
      <div class="studio-mini-grid">
        <div class="studio-mini">
          <strong>Surface</strong>
          <span>{'Public' if system_profile.get('public_surface') else 'Loopback only'}</span>
        </div>
        <div class="studio-mini">
          <strong>Auth</strong>
          <span>{'Enabled' if system_profile.get('auth_enabled') else 'Disabled'}</span>
        </div>
        <div class="studio-mini">
          <strong>Upload cap</strong>
          <span>{html.escape(str(system_profile.get('upload_limit_mb', 64)))} MB</span>
        </div>
        {public_url_html}
      </div>
    </section>
    """


def _reference_rows(items: list[dict]) -> list[list[object]]:
    rows = []
    for item in items:
        analysis = item["analysis"]
        quality_score = analysis.get("quality_score")
        rows.append(
            [
                item["id"],
                item["name"],
                f"{float(quality_score):.1f}" if quality_score is not None else "n/a",
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


def _diagnostic_reference_rows(items: list[dict]) -> list[list[object]]:
    rows = []
    for item in items:
        quality = reference_quality_breakdown(item.get("analysis") or {})
        rows.append([item["id"], item["name"], f"{quality['score']:.1f}", quality["rating"], " / ".join((quality["strengths"] or quality["issues"])[:2])])
    rows.sort(key=lambda row: (-float(row[2]), str(row[1]).lower(), int(row[0])))
    return rows


def _checkpoint_choice_pairs(paths: list[str]) -> list[tuple[str, str]]:
    choices = [("Base model (shipped)", "")]
    for checkpoint_path in paths:
        checkpoint_name = Path(checkpoint_path).name
        lowered = checkpoint_name.lower()
        if "noema" in lowered:
            label = f"{checkpoint_name} · no EMA"
        elif "ema" in lowered:
            label = f"{checkpoint_name} · EMA"
        elif "model_last" in lowered:
            label = f"{checkpoint_name} · raw training checkpoint"
        else:
            label = checkpoint_name
        choices.append((label, checkpoint_path))
    return choices


def _serialize_voice_model(checkpoint_path: str | None, use_ema: bool) -> str:
    return json.dumps({"checkpoint_path": checkpoint_path or "", "use_ema": bool(use_ema)}, sort_keys=True)


def _deserialize_voice_model(value: str | None) -> tuple[str | None, bool | None]:
    if not value:
        return None, None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None, None
    checkpoint_path = str(payload.get("checkpoint_path", "") or "").strip() or ""
    use_ema = payload.get("use_ema")
    return checkpoint_path, None if use_ema is None else bool(use_ema)


def _voice_model_choices(system_profile: dict, checkpoint_choices: list[tuple[str, str]]) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(label: str, checkpoint_path: str | None, use_ema: bool) -> None:
        value = _serialize_voice_model(checkpoint_path, use_ema)
        if value in seen:
            return
        seen.add(value)
        choices.append((label, value))

    current_label = f"Current runtime · {_current_voice_label(system_profile)}"
    add(current_label, system_profile.get("checkpoint_path"), bool(system_profile.get("use_ema", True)))
    add("Base model · shipped", "", True)

    for label, checkpoint_path in checkpoint_choices:
        if not checkpoint_path:
            continue
        lowered = checkpoint_path.lower()
        inferred_use_ema = "noema" not in lowered
        add(f"Checkpoint · {label}", checkpoint_path, inferred_use_ema)

    return choices


def _discover_trained_payload(system_profile: dict, checkpoint_choices: list[tuple[str, str]]) -> dict[str, object]:
    output_root = VOICE_PROJECT_ROOT / "output"
    bakeoff_summaries = sorted(
        output_root.glob("bakeoff_*/summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    bakeoff_data: list[dict] = []
    if bakeoff_summaries:
        try:
            bakeoff_data = json.loads(bakeoff_summaries[0].read_text(encoding="utf-8"))
        except Exception:
            bakeoff_data = []

    variant_lookup = {item.get("name"): item for item in bakeoff_data if isinstance(item, dict)}
    rows = []
    for item in bakeoff_data:
        expected = item.get("expected_text", "")
        transcript = item.get("transcript", "")
        rows.append(
            [
                _human_variant_name(item.get("name", "unknown")),
                _similarity_percent(expected, transcript),
                transcript,
            ]
        )

    longform_candidates = sorted(
        output_root.glob("*/final_stitched.wav"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    longform_audio = str(longform_candidates[0]) if longform_candidates else None
    active_voice = _current_voice_label(system_profile)
    checkpoint_path = system_profile.get("checkpoint_path")
    checkpoint_name = Path(checkpoint_path).name if checkpoint_path else "Base checkpoint"
    latest_bakeoff = bakeoff_summaries[0].parent.name if bakeoff_summaries else "No bakeoff recordings yet"
    comparison_count = str(len(rows))
    overview_html = f"""
    <section class="studio-trained-overview">
      <div class="studio-trained-copy">
        <p class="studio-eyebrow">Trained voice destination</p>
        <h2>{html.escape(active_voice)}</h2>
        <p>
          This page is for auditioning the finetuned model, not just changing settings. Compare bakeoff takes,
          switch checkpoints, and keep one authoritative place for trained recordings.
        </p>
      </div>
      <div class="studio-stat-grid">
        <div class="studio-stat">
          <span class="studio-stat-label">Active checkpoint</span>
          <span class="studio-stat-value">{html.escape(checkpoint_name)}</span>
        </div>
        <div class="studio-stat">
          <span class="studio-stat-label">Bakeoff set</span>
          <span class="studio-stat-value">{html.escape(latest_bakeoff)}</span>
        </div>
        <div class="studio-stat">
          <span class="studio-stat-label">Compared variants</span>
          <span class="studio-stat-value">{html.escape(comparison_count)}</span>
        </div>
        <div class="studio-stat">
          <span class="studio-stat-label">Long-form sample</span>
          <span class="studio-stat-value">{'Ready' if longform_audio else 'Missing'}</span>
        </div>
      </div>
    </section>
    """
    return {
        "overview_html": overview_html,
        "rows": rows,
        "base_audio": variant_lookup.get("base", {}).get("audio_path"),
        "ft_noema_audio": variant_lookup.get("ft_noema", {}).get("audio_path"),
        "ft_ema_audio": variant_lookup.get("ft_ema", {}).get("audio_path"),
        "longform_audio": longform_audio,
        "checkpoint_choices": checkpoint_choices,
    }


def create_studio_app():
    service = get_service()

    def _preferred_project_id(projects: list[dict]) -> int | None:
        saved_project_id = (service.store.get_setting("active_project_id", "") or "").strip()
        if saved_project_id.isdigit():
            saved_id = int(saved_project_id)
            if any(int(project["id"]) == saved_id for project in projects):
                return saved_id

        richest_project_id = None
        richest_score = -1
        for project in projects:
            detail = service.get_project_detail(int(project["id"]))
            score = (
                len(detail["references"]) * 100
                + len(detail["styles"]) * 10
                + len(detail["assets"]) * 5
                + len(detail["jobs"])
            )
            if score > richest_score:
                richest_score = score
                richest_project_id = int(project["id"])
        return richest_project_id

    def _project_state(selected_project_id: int | None = None) -> dict[str, object]:
        projects = service.list_projects()
        project_choices = _choice_pairs(projects)
        if not project_choices:
            project = service.create_project("Studio Sandbox", "Default local project for previews.")
            projects = service.list_projects()
            project_choices = _choice_pairs(projects)
            selected_project_id = project["id"]

        valid_ids = {value for _, value in project_choices}
        if selected_project_id not in valid_ids:
            selected_project_id = _preferred_project_id(projects) or project_choices[0][1]

        service.store.set_setting("active_project_id", str(selected_project_id))

        project_detail = service.get_project_detail(int(selected_project_id))
        references = project_detail["references"]
        styles = project_detail["styles"]
        assets = project_detail["assets"]
        sources = [asset for asset in assets if asset["kind"] == "source"]
        jobs = project_detail["jobs"]
        rules = project_detail["pronunciation_rules"]
        system_profile = service.system_profile()
        recommended_references = service.recommend_references(int(selected_project_id))

        reference_choices = _choice_pairs(references)
        style_choices = [("No style prompt", 0)] + _choice_pairs(styles)
        source_choices = _choice_pairs(sources, label_key="label")
        asset_choices = _choice_pairs(assets, label_key="label")
        job_choices = [(f"{job['id']} · {job['name']}", int(job["id"])) for job in jobs if job["status"] in {"queued", "running"}]
        checkpoint_choices = _checkpoint_choice_pairs(service.list_checkpoint_candidates())
        voice_model_choices = _voice_model_choices(system_profile, checkpoint_choices)
        trained_payload = _discover_trained_payload(system_profile, checkpoint_choices)
        current_checkpoint = system_profile.get("checkpoint_path") or ""
        current_use_ema = bool(system_profile.get("use_ema", True))

        return {
            "project_choices": project_choices,
            "selected_project_id": selected_project_id,
            "project_overview": _project_overview_html(project_detail, system_profile),
            "reference_choices": reference_choices,
            "reference_value": (recommended_references[0]["id"] if recommended_references else _dropdown_value(reference_choices)),
            "style_choices": style_choices,
            "style_value": 0,
            "source_choices": source_choices,
            "source_value": _dropdown_value(source_choices),
            "voice_model_choices": voice_model_choices,
            "voice_model_value": voice_model_choices[0][1] if voice_model_choices else _serialize_voice_model("", True),
            "reference_rows": _reference_rows(references),
            "style_rows": _style_rows(styles),
            "asset_rows": _asset_rows(assets),
            "job_rows": _job_rows(jobs),
            "rule_rows": _rule_rows(rules),
            "diagnostic_reference_rows": _diagnostic_reference_rows(references),
            "asset_choices": asset_choices,
            "take_a_value": _dropdown_value(asset_choices),
            "take_b_value": _dropdown_value(asset_choices, 1) or _dropdown_value(asset_choices),
            "export_value": _dropdown_value(asset_choices),
            "job_choices": job_choices,
            "job_value": _dropdown_value(job_choices),
            "runtime_snapshot": _runtime_snapshot_html(system_profile),
            "sharing_snapshot": _sharing_snapshot_html(system_profile),
            "system_profile_json": json.dumps(system_profile, indent=2),
            "current_checkpoint": current_checkpoint,
            "current_use_ema": current_use_ema,
            "trained_overview": trained_payload["overview_html"],
            "trained_rows": trained_payload["rows"],
            "trained_base_audio": trained_payload["base_audio"],
            "trained_ft_noema_audio": trained_payload["ft_noema_audio"],
            "trained_ft_ema_audio": trained_payload["ft_ema_audio"],
            "trained_longform_audio": trained_payload["longform_audio"],
            "checkpoint_choices": checkpoint_choices,
        }

    def project_updates(selected_project_id: int | None = None):
        state = _project_state(selected_project_id)

        return (
            gr.update(choices=state["project_choices"], value=state["selected_project_id"]),
            state["project_overview"],
            gr.update(choices=state["reference_choices"], value=state["reference_value"]),
            gr.update(choices=state["style_choices"], value=state["style_value"]),
            gr.update(choices=state["reference_choices"], value=state["reference_value"]),
            gr.update(choices=state["style_choices"], value=state["style_value"]),
            gr.update(choices=state["reference_choices"], value=state["reference_value"]),
            gr.update(choices=state["style_choices"], value=state["style_value"]),
            gr.update(choices=state["source_choices"], value=state["source_value"]),
            gr.update(choices=state["voice_model_choices"], value=state["voice_model_value"]),
            state["reference_rows"],
            state["style_rows"],
            state["asset_rows"],
            state["job_rows"],
            state["job_rows"],
            state["rule_rows"],
            state["diagnostic_reference_rows"],
            gr.update(choices=state["asset_choices"], value=state["export_value"]),
            gr.update(choices=state["reference_choices"], value=state["reference_value"]),
            gr.update(choices=state["asset_choices"], value=state["take_a_value"]),
            gr.update(choices=state["asset_choices"], value=state["take_b_value"]),
            gr.update(choices=state["asset_choices"], value=state["export_value"]),
            gr.update(choices=state["job_choices"], value=state["job_value"]),
            state["runtime_snapshot"],
            state["sharing_snapshot"],
            state["system_profile_json"],
            gr.update(value=state["current_checkpoint"]),
            gr.update(value=state["current_use_ema"]),
            state["trained_overview"],
            state["trained_rows"],
            state["trained_base_audio"],
            state["trained_ft_noema_audio"],
            state["trained_ft_ema_audio"],
            state["trained_longform_audio"],
            gr.update(choices=state["checkpoint_choices"], value=state["current_checkpoint"]),
            gr.update(value=state["current_use_ema"]),
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
            f"Quality: {analysis.get('quality_score', 'n/a')} ({analysis.get('quality_rating', 'unrated')})\n"
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

    def save_edit_source(project_id: int, name: str, source_audio: str, source_text: str):
        if not project_id:
            raise gr.Error("Select a project first.")
        if not source_audio:
            raise gr.Error("Upload the source audio you want to edit first.")
        staged = service.stage_upload(source_audio)
        saved, metadata = service.ingest_edit_source(project_id, name or "Editable Source", staged, source_text)
        refresh = project_updates(project_id)
        summary = (
            f"Saved editable source #{saved['id']}.\n"
            f"Transcript: {metadata['transcript']}\n"
            f"Alignment backend: {metadata['alignment_backend']}\n"
            f"Warnings: {', '.join(metadata['warnings']) if metadata['warnings'] else 'None'}"
        )
        return metadata["transcript"], summary, json.dumps(metadata["alignment"][:48], indent=2) + ("\n..." if len(metadata["alignment"]) > 48 else ""), *refresh

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
        seed: int,
        checkpoint_override: str | None = None,
        use_ema_override: bool | None = None,
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
            seed=None if seed <= 0 else int(seed),
            checkpoint_path=checkpoint_override,
            use_ema=use_ema_override,
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
        seed: int,
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
            seed,
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
        seed: int,
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
            seed,
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
            f"Output length: {format_duration(result['duration_seconds'])}\n"
            f"Seed: {result.get('seed', 'random')}"
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

    def _build_voice_request(
        project_id: int,
        reference_id: int,
        style_id: int,
        model_choice: str,
        name: str,
        text: str,
        mode: str,
        use_style_prompt: bool,
        context_notes: str,
        speed: float,
        nfe_step: int,
        remove_silence: bool,
        render_spectrogram: bool,
        seed: int,
    ) -> GenerationRequest:
        checkpoint_override, use_ema_override = _deserialize_voice_model(model_choice)
        return build_request(
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
            seed,
            checkpoint_override=checkpoint_override,
            use_ema_override=use_ema_override,
        )

    def estimate_voice_render(
        project_id: int,
        reference_id: int,
        style_id: int,
        model_choice: str,
        name: str,
        text: str,
        use_style_prompt: bool,
        context_notes: str,
        speed: float,
        nfe_step: int,
        seed: int,
    ):
        request = _build_voice_request(
            project_id,
            reference_id,
            style_id,
            model_choice,
            name,
            text,
            "final",
            use_style_prompt,
            context_notes,
            speed,
            nfe_step,
            False,
            False,
            seed,
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

    def render_voice_now(
        project_id: int,
        reference_id: int,
        style_id: int,
        model_choice: str,
        name: str,
        text: str,
        mode: str,
        use_style_prompt: bool,
        context_notes: str,
        speed: float,
        nfe_step: int,
        remove_silence: bool,
        render_spectrogram: bool,
        seed: int,
    ):
        request = _build_voice_request(
            project_id,
            reference_id,
            style_id,
            model_choice,
            name,
            text,
            mode,
            use_style_prompt,
            context_notes,
            speed,
            nfe_step,
            remove_silence,
            render_spectrogram,
            seed,
        )
        job = service.render_now(request)
        if job["status"] != "completed":
            raise gr.Error(job.get("error_text") or "Render failed.")

        result = job["result"]
        refresh = project_updates(project_id)
        status = (
            f"Voice page {mode} render complete.\n"
            f"Take #{result['asset_id']} saved to the project library.\n"
            f"Generation time: {format_duration(result['elapsed_seconds'])}\n"
            f"Output length: {format_duration(result['duration_seconds'])}\n"
            f"Seed: {result.get('seed', 'random')}"
        )
        return (
            result["audio_path"],
            result.get("spectrogram_path"),
            status,
        ) + refresh

    def render_voice_preview(*args):
        return render_voice_now(*args[:6], "preview", *args[6:])

    def render_voice_final(*args):
        return render_voice_now(*args[:6], "final", *args[6:])

    def load_edit_source_preview(source_asset_id: int | None):
        if not source_asset_id:
            return "", "", None
        asset = service.store.get_audio_asset(int(source_asset_id))
        metadata = asset.get("metadata", {})
        alignment = metadata.get("alignment") or []
        preview = json.dumps(alignment[:48], indent=2)
        if len(alignment) > 48:
            preview += "\n..."
        return metadata.get("transcript", ""), preview, asset["path"]

    def render_edit_now(
        project_id: int,
        source_asset_id: int,
        take_name: str,
        target_text: str,
        replacement_text: str,
        occurrence: int,
        action: str,
        preserve_timing: bool,
        mode: str,
        nfe_step: int,
        render_spectrogram: bool,
    ):
        if not project_id:
            raise gr.Error("Select a project first.")
        if not source_asset_id:
            raise gr.Error("Analyze and save an editable source first.")
        if not target_text.strip():
            raise gr.Error("Enter the anchor phrase from the transcript.")
        if action in {"replace", "insert_before", "insert_after"} and not replacement_text.strip():
            raise gr.Error("Enter the replacement or inserted text for this edit action.")
        effective_nfe = 0 if nfe_step <= 0 else int(nfe_step)
        if mode == "preview" and effective_nfe <= 0:
            effective_nfe = PROFILE_MAP[service.get_runtime_profile_name()].preview_nfe_step
        elif effective_nfe <= 0:
            effective_nfe = PROFILE_MAP[service.get_runtime_profile_name()].final_nfe_step

        result = service.render_edit_now(
            project_id=int(project_id),
            source_asset_id=int(source_asset_id),
            take_name=take_name.strip() or "Speech Edit",
            target_text=target_text,
            replacement_text=replacement_text,
            occurrence=max(int(occurrence), 1),
            action=action,
            preserve_timing=preserve_timing,
            nfe_step=effective_nfe,
            render_spectrogram=render_spectrogram,
        )
        refresh = project_updates(project_id)
        status = (
            f"Edit render complete.\n"
            f"Asset #{result['asset_id']} saved to the project library.\n"
            f"Edit action: {result['plan']['action']}\n"
            f"Edited phrase: {result['plan']['target_text']} -> {result['plan']['replacement_text'] or '[deleted]'}\n"
            f"Output length: {format_duration(result['duration_seconds'])}"
        )
        return result["audio_path"], result.get("spectrogram_path"), json.dumps(result["plan"], indent=2), status, *refresh

    def render_edit_preview(*args):
        return render_edit_now(*args[:8], "preview", *args[8:])

    def render_edit_final(*args):
        return render_edit_now(*args[:8], "final", *args[8:])

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
                0,
            )
            service.enqueue_generation(request)
        refresh = project_updates(project_id)
        status = f"Queued {len(scripts)} jobs. The worker will process them one at a time."
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

    def diagnose_take(project_id: int, asset_id: int | None, reference_id: int | None, expected_text: str):
        if not project_id:
            raise gr.Error("Select a project first.")
        if not asset_id:
            raise gr.Error("Choose a saved take first.")
        report = service.diagnose_asset(
            int(project_id),
            int(asset_id),
            reference_id=None if not reference_id else int(reference_id),
            expected_text=expected_text,
        )
        status = (
            f"Diagnostics complete for take #{report['asset_id']}.\n"
            f"Transcript backend: {report['transcript_backend']}\n"
        )
        if report.get("word_error_rate") is not None:
            status += f"WER: {report['word_error_rate']:.2%}\n"
        if report.get("char_error_rate") is not None:
            status += f"CER: {report['char_error_rate']:.2%}\n"
        if report.get("voice_similarity_proxy") is not None:
            status += f"Similarity proxy: {report['voice_similarity_proxy']:.2f}\n"
        status += f"Warnings: {', '.join(report['warnings']) if report['warnings'] else 'None'}"
        return status, json.dumps(report, indent=2)

    def detect_checkpoint_widgets():
        detected = service.detect_latest_checkpoint()
        choices = _checkpoint_choice_pairs(service.list_checkpoint_candidates())
        if not detected:
            status = "No local finetune checkpoint was detected."
            return "", gr.update(choices=choices, value=""), status, status
        status = f"Detected latest checkpoint: {detected}"
        return detected, gr.update(choices=choices, value=detected), status, status

    def save_settings(
        project_id: int,
        profile_name: str,
        asr_backend: str,
        idle_unload_seconds: int,
        checkpoint_path_value: str,
        use_ema_value: bool,
    ):
        service.set_runtime_profile(profile_name)
        service.set_asr_backend(asr_backend)
        service.set_idle_unload_seconds(idle_unload_seconds)
        service.set_checkpoint_path(checkpoint_path_value)
        service.set_use_ema(use_ema_value)
        refresh = project_updates(project_id)
        return refresh + ("Settings saved.", "Settings saved.")

    def warm_engine(project_id: int | None):
        message = service.warm_engine_now()
        refresh = project_updates(project_id)
        return refresh + (message, message)

    def apply_trained_checkpoint(project_id: int, checkpoint_value: str, use_ema_value: bool):
        service.set_checkpoint_path(checkpoint_value)
        service.set_use_ema(use_ema_value)
        refresh = project_updates(project_id)
        message = (
            f"Trained voice updated. Active checkpoint: {checkpoint_value}"
            if checkpoint_value
            else "Switched back to the shipped base model."
        )
        return refresh + (message, message)

    def switch_to_base_model(project_id: int):
        service.set_checkpoint_path("")
        service.set_use_ema(True)
        refresh = project_updates(project_id)
        message = "Switched back to the shipped base model."
        return refresh + (message, message)

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

    initial_state = _project_state()

    with gr.Blocks(
        title=TITLE,
        delete_cache=(3600, 7200),
        analytics_enabled=False,
    ) as app:
        with gr.Column(elem_classes=["studio-shell"]):
            gr.HTML(HERO_HTML)

            with gr.Row():
                with gr.Column(scale=4, elem_classes=["studio-panel", "studio-control-stack"]):
                    gr.Markdown("### Project rail")
                    gr.Markdown(
                        "Choose the active project, create a new one, and keep one delivery lane open at a time.",
                        elem_classes=["studio-muted-note"],
                    )
                    active_project = gr.Dropdown(
                        label="Active project",
                        choices=initial_state["project_choices"],
                        value=initial_state["selected_project_id"],
                        interactive=True,
                    )
                    project_name = gr.Textbox(label="New project name", placeholder="Narrative Audio")
                    project_description = gr.Textbox(label="Description", lines=3, placeholder="Client delivery pack, in-house narration, character study...")
                    with gr.Row():
                        create_project_btn = gr.Button("Create project", elem_classes=["studio-secondary"])
                        refresh_btn = gr.Button("Refresh", elem_classes=["studio-secondary"])
                    create_project_status = gr.Textbox(label="Project status", interactive=False)

                with gr.Column(scale=8):
                    project_overview = gr.HTML(value=initial_state["project_overview"])

            with gr.Tabs(elem_classes=["studio-tabs"]):
                with gr.Tab("Create"):
                    gr.HTML(
                        _page_intro_html(
                            "Creation",
                            "Build the take, not the clutter.",
                            "Identity on the left, script in the middle, live playback on the right. The page should feel like direction, not paperwork.",
                        )
                    )
                    with gr.Row():
                        with gr.Column(scale=4, elem_classes=["studio-panel", "studio-stack"]):
                            gr.Markdown("### Identity")
                            reference_name = gr.Textbox(label="Reference name", value="Lead Voice")
                            reference_audio = gr.Audio(label="Reference audio", type="filepath", sources=["upload", "microphone"])
                            reference_text = gr.Textbox(label="Reference transcript", lines=4, placeholder="Leave blank to transcribe locally.")
                            reference_consent = gr.Checkbox(
                                label="I have the right to use this voice sample",
                                value=False,
                            )
                            save_reference_btn = gr.Button("Analyze and save reference", elem_classes=["studio-secondary"])
                            reference_status = gr.Textbox(label="Reference status", lines=5, interactive=False)
                            reference_choice = gr.Dropdown(
                                label="Saved reference",
                                choices=initial_state["reference_choices"],
                                value=initial_state["reference_value"],
                            )
                            gr.Markdown("### Delivery prompt")
                            style_name = gr.Textbox(label="Style name", value="Calm Narrator")
                            style_audio = gr.Audio(label="Style prompt audio", type="filepath", sources=["upload", "microphone"])
                            style_text = gr.Textbox(label="Style transcript", lines=3, placeholder="Leave blank to transcribe locally.")
                            context_notes = gr.Textbox(
                                label="Delivery context",
                                lines=3,
                                placeholder="Examples: calm narrator, energetic trailer, soft whisper, urgent support update.",
                            )
                            save_style_btn = gr.Button("Analyze and save style", elem_classes=["studio-secondary"])
                            style_status = gr.Textbox(label="Style status", lines=5, interactive=False)
                            style_choice = gr.Dropdown(
                                label="Saved style",
                                choices=initial_state["style_choices"],
                                value=initial_state["style_value"],
                            )

                        with gr.Column(scale=5, elem_classes=["studio-panel", "studio-stack"]):
                            gr.Markdown("### Script workspace")
                            preset_choice = gr.Dropdown(label="Starter preset", choices=list(SCRIPT_PRESETS.keys()), value="Product demo")
                            load_preset_btn = gr.Button("Load preset", elem_classes=["studio-secondary"])
                            render_name = gr.Textbox(label="Take name", value="Final Render")
                            render_text = gr.Textbox(label="Script", lines=14, value=SCRIPT_PRESETS["Product demo"])
                            use_style_prompt = gr.Checkbox(label="Use saved style prompt during generation", value=True)
                            render_mode = gr.Radio(label="Render mode", choices=["preview", "final"], value="final")
                            with gr.Accordion("Advanced controls", open=False):
                                speed = gr.Slider(label="Speed override", minimum=0.0, maximum=1.5, step=0.05, value=0.0)
                                nfe_step = gr.Slider(label="NFE override", minimum=0, maximum=64, step=2, value=0)
                                seed = gr.Number(label="Seed lock (0 = random)", value=0, precision=0)
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
                                value="Select a project, save a clean reference, and render a first take.",
                            )

                        with gr.Column(scale=3, elem_classes=["studio-panel", "studio-stack"]):
                            runtime_snapshot = gr.HTML(value=initial_state["runtime_snapshot"])
                            output_audio = gr.Audio(label="Latest output", type="filepath", elem_classes=["studio-audio-card"])
                            output_spectrogram = gr.Image(label="Spectrogram")

                with gr.Tab("Edit"):
                    gr.HTML(
                        _page_intro_html(
                            "Editing",
                            "Alignment-first speech editing.",
                            "Import an existing recording, confirm the transcript, choose the words to replace by text, then render a localized edit instead of hand-trimming time ranges.",
                        )
                    )
                    with gr.Row():
                        with gr.Column(scale=4, elem_classes=["studio-panel", "studio-stack"]):
                            gr.Markdown("### Editable source")
                            edit_source_name = gr.Textbox(label="Source name", value="Editable Source")
                            edit_source_audio = gr.Audio(label="Source audio", type="filepath", sources=["upload", "microphone"])
                            edit_source_text = gr.Textbox(
                                label="Confirmed transcript",
                                lines=6,
                                placeholder="Paste the exact transcript when you know it. Leave blank only when you need local alignment help.",
                            )
                            save_edit_source_btn = gr.Button("Analyze and save source", elem_classes=["studio-secondary"])
                            edit_source_status = gr.Textbox(label="Source status", lines=6, interactive=False)
                            edit_source_choice = gr.Dropdown(
                                label="Saved editable source",
                                choices=initial_state["source_choices"],
                                value=initial_state["source_value"],
                            )
                            edit_source_audio_preview = gr.Audio(
                                label="Selected source",
                                type="filepath",
                                elem_classes=["studio-audio-card"],
                            )

                        with gr.Column(scale=5, elem_classes=["studio-panel", "studio-stack"]):
                            gr.Markdown("### Edit plan")
                            edit_take_name = gr.Textbox(label="Edited take name", value="Speech Edit")
                            edit_target_text = gr.Textbox(
                                label="Replace this phrase",
                                placeholder="Exact words from the transcript to replace",
                            )
                            edit_replacement_text = gr.Textbox(
                                label="Replacement / inserted text",
                                placeholder="Replacement text, or leave blank for delete",
                            )
                            with gr.Row():
                                edit_occurrence = gr.Number(label="Occurrence", value=1, precision=0)
                                edit_action = gr.Radio(
                                    label="Edit action",
                                    choices=["replace", "delete", "insert_before", "insert_after"],
                                    value="replace",
                                )
                                edit_preserve_timing = gr.Checkbox(label="Preserve original timing", value=True)
                            with gr.Accordion("Advanced controls", open=False):
                                edit_nfe_step = gr.Slider(label="NFE override", minimum=0, maximum=64, step=2, value=0)
                                edit_render_spectrogram = gr.Checkbox(label="Render spectrogram", value=False)
                            with gr.Row():
                                edit_preview_btn = gr.Button("Quick edit preview", elem_classes=["studio-primary"])
                                edit_final_btn = gr.Button("Final edit render", elem_classes=["studio-primary"])
                            edit_status = gr.Textbox(
                                label="Edit status",
                                lines=6,
                                interactive=False,
                                value="Save a source, choose the phrase to replace, and render a localized edit.",
                            )

                        with gr.Column(scale=3, elem_classes=["studio-panel", "studio-stack"]):
                            gr.Markdown("### Alignment and result")
                            edit_alignment_preview = gr.Code(label="Word alignment preview", language="json")
                            edit_output_audio = gr.Audio(label="Edited output", type="filepath", elem_classes=["studio-audio-card"])
                            edit_output_spectrogram = gr.Image(label="Spectrogram")
                            edit_plan_preview = gr.Code(label="Resolved edit plan", language="json")

                with gr.Tab("Voices"):
                    gr.HTML(
                        _page_intro_html(
                            "Voices",
                            "Direct text-to-speech for saved and trained voices.",
                            "Pick a saved reference, decide whether it speaks through the base model or a finetuned checkpoint, then render without bouncing through runtime settings.",
                        )
                    )
                    with gr.Row():
                        with gr.Column(scale=4, elem_classes=["studio-panel", "studio-stack"]):
                            gr.Markdown("### Voice route")
                            voice_reference_choice = gr.Dropdown(
                                label="Reference voice",
                                choices=initial_state["reference_choices"],
                                value=initial_state["reference_value"],
                            )
                            voice_model_choice = gr.Dropdown(
                                label="Voice engine",
                                choices=initial_state["voice_model_choices"],
                                value=initial_state["voice_model_value"],
                            )
                            voice_style_choice = gr.Dropdown(
                                label="Optional style prompt",
                                choices=initial_state["style_choices"],
                                value=initial_state["style_value"],
                            )
                            voice_use_style_prompt = gr.Checkbox(
                                label="Use saved style prompt during generation",
                                value=True,
                            )
                            voice_context_notes = gr.Textbox(
                                label="Delivery context",
                                lines=3,
                                placeholder="Examples: intimate, calm narrator, urgent, trailer read, soft whisper.",
                            )
                            voice_take_name = gr.Textbox(label="Take name", value="Voice Page Render")
                            voice_status = gr.Textbox(
                                label="Voice status",
                                interactive=False,
                                lines=6,
                                value="Choose a reference and voice engine, then render directly from this page.",
                            )

                        with gr.Column(scale=5, elem_classes=["studio-panel", "studio-stack"]):
                            gr.Markdown("### Text to speech")
                            voice_preset_choice = gr.Dropdown(
                                label="Starter preset",
                                choices=list(SCRIPT_PRESETS.keys()),
                                value="Product demo",
                            )
                            voice_load_preset_btn = gr.Button("Load preset", elem_classes=["studio-secondary"])
                            voice_text = gr.Textbox(
                                label="Script",
                                lines=14,
                                value=SCRIPT_PRESETS["Product demo"],
                            )
                            with gr.Accordion("Advanced controls", open=False):
                                voice_speed = gr.Slider(label="Speed override", minimum=0.0, maximum=1.5, step=0.05, value=0.0)
                                voice_nfe_step = gr.Slider(label="NFE override", minimum=0, maximum=64, step=2, value=0)
                                voice_seed = gr.Number(label="Seed lock (0 = random)", value=0, precision=0)
                                voice_remove_silence = gr.Checkbox(label="Trim long silence in output", value=False)
                                voice_render_spectrogram = gr.Checkbox(label="Render spectrogram", value=False)
                            with gr.Row():
                                voice_estimate_btn = gr.Button("Estimate runtime", elem_classes=["studio-secondary"])
                                voice_preview_btn = gr.Button("Quick preview", elem_classes=["studio-primary"])
                                voice_final_btn = gr.Button("Final render", elem_classes=["studio-primary"])

                        with gr.Column(scale=3, elem_classes=["studio-panel", "studio-stack"]):
                            gr.Markdown("### Latest voice output")
                            voice_output_audio = gr.Audio(
                                label="Direct TTS output",
                                type="filepath",
                                elem_classes=["studio-audio-card"],
                            )
                            voice_output_spectrogram = gr.Image(label="Spectrogram")
                            gr.Markdown(
                                "This page is the fast lane: reference voice plus checkpoint plus text, with no detour through global settings.",
                                elem_classes=["studio-muted-note"],
                            )

                with gr.Tab("Trained Voice"):
                    gr.HTML(
                        _page_intro_html(
                            "Trained voice",
                            "A separate destination for the finetuned model.",
                            "This page is for checkpoint choice, bakeoff review, and trained recordings. You should not need to hunt through generic library pages to hear what the training actually did.",
                        )
                    )
                    with gr.Row():
                        with gr.Column(scale=5, elem_classes=["studio-panel", "studio-stack"]):
                            trained_overview = gr.HTML(value=initial_state["trained_overview"])
                            trained_checkpoint_choice = gr.Dropdown(
                                label="Checkpoint to audition",
                                choices=initial_state["checkpoint_choices"],
                                value=initial_state["current_checkpoint"],
                            )
                            trained_use_ema = gr.Checkbox(
                                label="Use EMA weights for the selected checkpoint",
                                value=initial_state["current_use_ema"],
                            )
                            with gr.Row():
                                trained_detect_checkpoint_btn = gr.Button("Detect latest checkpoint", elem_classes=["studio-secondary"])
                                trained_apply_btn = gr.Button("Use selected checkpoint", elem_classes=["studio-primary"])
                            trained_base_btn = gr.Button("Switch back to base model", elem_classes=["studio-secondary"])
                            trained_page_status = gr.Textbox(
                                label="Trained voice status",
                                interactive=False,
                                lines=4,
                                value="The current trained checkpoint and bakeoff recordings will appear here.",
                            )

                        with gr.Column(scale=7, elem_classes=["studio-panel", "studio-stack"]):
                            trained_bakeoff_table = gr.Dataframe(
                                headers=["Variant", "Readback score", "Transcript"],
                                datatype=["str", "str", "str"],
                                interactive=False,
                                label="Bakeoff recordings",
                                value=initial_state["trained_rows"],
                            )
                            trained_longform_audio = gr.Audio(
                                label="Latest long-form trained render",
                                type="filepath",
                                elem_classes=["studio-audio-card"],
                                value=initial_state["trained_longform_audio"],
                            )

                    with gr.Row():
                        with gr.Column(scale=4, elem_classes=["studio-panel", "studio-stack"]):
                            gr.Markdown("### Base model")
                            trained_base_audio = gr.Audio(
                                label="Bakeoff · base",
                                type="filepath",
                                elem_classes=["studio-audio-card"],
                                value=initial_state["trained_base_audio"],
                            )
                        with gr.Column(scale=4, elem_classes=["studio-panel", "studio-stack"]):
                            gr.Markdown("### Finetuned snapshot · no EMA")
                            trained_ft_noema_audio = gr.Audio(
                                label="Bakeoff · no EMA",
                                type="filepath",
                                elem_classes=["studio-audio-card"],
                                value=initial_state["trained_ft_noema_audio"],
                            )
                        with gr.Column(scale=4, elem_classes=["studio-panel", "studio-stack"]):
                            gr.Markdown("### Finetuned snapshot · EMA")
                            trained_ft_ema_audio = gr.Audio(
                                label="Bakeoff · EMA",
                                type="filepath",
                                elem_classes=["studio-audio-card"],
                                value=initial_state["trained_ft_ema_audio"],
                            )

                with gr.Tab("Library"):
                    gr.HTML(
                        _page_intro_html(
                            "Library",
                            "Everything saved in the active project.",
                            "References, style prompts, takes, and exports live here. The goal is fast auditioning and retrieval, not an endless card wall.",
                        )
                    )
                    with gr.Row():
                        with gr.Column(scale=6, elem_classes=["studio-panel", "studio-stack"]):
                            references_table = gr.Dataframe(
                                headers=["ID", "Name", "Quality", "Duration", "ASR", "Warnings"],
                                datatype=["number", "str", "str", "str", "str", "str"],
                                interactive=False,
                                label="Reference library",
                                value=initial_state["reference_rows"],
                            )
                            styles_table = gr.Dataframe(
                                headers=["ID", "Name", "Duration", "Speed", "Keywords"],
                                datatype=["number", "str", "str", "str", "str"],
                                interactive=False,
                                label="Style library",
                                value=initial_state["style_rows"],
                            )
                        with gr.Column(scale=6, elem_classes=["studio-panel", "studio-stack"]):
                            assets_table = gr.Dataframe(
                                headers=["ID", "Kind", "Label", "Duration", "Created"],
                                datatype=["number", "str", "str", "str", "str"],
                                interactive=False,
                                label="Saved takes",
                                value=initial_state["asset_rows"],
                            )
                            jobs_table = gr.Dataframe(
                                headers=["ID", "Name", "Status", "Updated", "Notes"],
                                datatype=["number", "str", "str", "str", "str"],
                                interactive=False,
                                label="Recent jobs",
                                value=initial_state["job_rows"],
                            )
                            rules_table = gr.Dataframe(
                                headers=["ID", "Source", "Replacement", "Updated"],
                                datatype=["number", "str", "str", "str"],
                                interactive=False,
                                label="Pronunciation rules",
                                value=initial_state["rule_rows"],
                            )

                    with gr.Row():
                        with gr.Column(scale=6, elem_classes=["studio-panel", "studio-stack"]):
                            gr.Markdown("### A/B compare")
                            take_a = gr.Dropdown(label="Take A", choices=initial_state["asset_choices"], value=initial_state["take_a_value"])
                            take_b = gr.Dropdown(label="Take B", choices=initial_state["asset_choices"], value=initial_state["take_b_value"])
                            compare_btn = gr.Button("Load comparison", elem_classes=["studio-secondary"])
                            compare_audio_a = gr.Audio(label="Take A audio", type="filepath", elem_classes=["studio-audio-card"])
                            compare_meta_a = gr.Code(label="Take A metadata", language="json")
                        with gr.Column(scale=6, elem_classes=["studio-panel", "studio-stack"]):
                            gr.Markdown("### Export")
                            export_take_choice = gr.Dropdown(
                                label="Export a take",
                                choices=initial_state["asset_choices"],
                                value=initial_state["export_value"],
                            )
                            export_btn = gr.Button("Create share bundle", elem_classes=["studio-secondary"])
                            export_status = gr.Textbox(label="Export status", interactive=False)
                            export_file = gr.File(label="Bundle index", interactive=False)
                            compare_audio_b = gr.Audio(label="Take B audio", type="filepath", elem_classes=["studio-audio-card"])
                            compare_meta_b = gr.Code(label="Take B metadata", language="json")

                with gr.Tab("Diagnostics"):
                    gr.HTML(
                        _page_intro_html(
                            "Diagnostics",
                            "Measure quality, don’t just guess.",
                            "Use the reference coach to pick the strongest identity clip, then diagnose saved takes against expected text and reference timbre before you ship them.",
                        )
                    )
                    with gr.Row():
                        with gr.Column(scale=5, elem_classes=["studio-panel", "studio-stack"]):
                            diagnostic_references_table = gr.Dataframe(
                                headers=["ID", "Name", "Score", "Rating", "Why"],
                                datatype=["number", "str", "str", "str", "str"],
                                interactive=False,
                                label="Reference coach",
                                value=initial_state["diagnostic_reference_rows"],
                            )
                            gr.Markdown(
                                "Higher scores usually mean cleaner identity anchors: stable loudness, enough speech, headroom, and a safer 6-12 second window.",
                                elem_classes=["studio-muted-note"],
                            )
                        with gr.Column(scale=7, elem_classes=["studio-panel", "studio-stack"]):
                            diagnostic_asset_choice = gr.Dropdown(
                                label="Saved take to diagnose",
                                choices=initial_state["asset_choices"],
                                value=initial_state["export_value"],
                            )
                            diagnostic_reference_choice = gr.Dropdown(
                                label="Reference voice for similarity check",
                                choices=initial_state["reference_choices"],
                                value=initial_state["reference_value"],
                            )
                            diagnostic_expected_text = gr.Textbox(
                                label="Expected text override",
                                lines=5,
                                placeholder="Leave blank to use the saved render recipe or edited transcript when available.",
                            )
                            diagnostic_run_btn = gr.Button("Run diagnostics", elem_classes=["studio-primary"])
                            diagnostic_status = gr.Textbox(
                                label="Diagnostics status",
                                interactive=False,
                                lines=6,
                                value="Pick a saved take and run a local QA pass.",
                            )
                            diagnostic_report = gr.Code(label="Diagnostics report", language="json")

                with gr.Tab("Queue"):
                    gr.HTML(
                        _page_intro_html(
                            "Queue",
                            "Batch jobs without cooking the machine.",
                            "Use one shared voice setup, queue several blocks, and let the worker process them one at a time.",
                        )
                    )
                    with gr.Row():
                        with gr.Column(scale=6, elem_classes=["studio-panel", "studio-stack"]):
                            batch_reference = gr.Dropdown(
                                label="Batch reference",
                                choices=initial_state["reference_choices"],
                                value=initial_state["reference_value"],
                            )
                            batch_style = gr.Dropdown(
                                label="Batch style",
                                choices=initial_state["style_choices"],
                                value=initial_state["style_value"],
                            )
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
                        with gr.Column(scale=6, elem_classes=["studio-panel", "studio-stack"]):
                            cancel_job_choice = gr.Dropdown(
                                label="Cancelable jobs",
                                choices=initial_state["job_choices"],
                                value=initial_state["job_value"],
                            )
                            cancel_job_btn = gr.Button("Cancel selected job", elem_classes=["studio-secondary"])
                            batch_jobs_table = gr.Dataframe(
                                headers=["ID", "Name", "Status", "Updated", "Notes"],
                                datatype=["number", "str", "str", "str", "str"],
                                interactive=False,
                                label="Queue monitor",
                                value=initial_state["job_rows"],
                            )

                with gr.Tab("Runtime"):
                    gr.HTML(
                        _page_intro_html(
                            "Runtime",
                            "System controls and pronunciation rules.",
                            "This is the operations page: runtime profile, checkpoint override, ASR backend, and the per-project pronunciation dictionary.",
                        )
                    )
                    with gr.Row():
                        with gr.Column(scale=5, elem_classes=["studio-panel", "studio-stack"]):
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
                            checkpoint_path = gr.Textbox(
                                label="Inference checkpoint override",
                                value=initial_state["current_checkpoint"],
                                lines=2,
                                placeholder="Leave blank to use the shipped base model.",
                            )
                            use_ema = gr.Checkbox(label="Use EMA weights", value=initial_state["current_use_ema"])
                            with gr.Row():
                                detect_checkpoint_btn = gr.Button("Detect latest checkpoint", elem_classes=["studio-secondary"])
                                warm_engine_btn = gr.Button("Warm engine now", elem_classes=["studio-secondary"])
                                save_settings_btn = gr.Button("Save runtime settings", elem_classes=["studio-primary"])
                            settings_status = gr.Textbox(label="Settings status", interactive=False)
                        with gr.Column(scale=7, elem_classes=["studio-panel", "studio-stack"]):
                            gr.Markdown("### Pronunciation dictionary")
                            rule_source = gr.Textbox(label="Source phrase", placeholder="bhauju")
                            rule_replacement = gr.Textbox(label="Replacement phrase", placeholder="bah-joo")
                            save_rule_btn = gr.Button("Save pronunciation rule", elem_classes=["studio-secondary"])
                            rule_status = gr.Textbox(label="Rule status", interactive=False)
                            sharing_snapshot = gr.HTML(value=initial_state["sharing_snapshot"])
                            system_profile_settings = gr.Code(
                                label="System profile",
                                language="json",
                                value=initial_state["system_profile_json"],
                            )
                            gr.Markdown(
                                "Launch locally with `f5-tts_voice-studio`, or mount the API and UI together with `f5-tts_studio-server`.",
                                elem_classes=["studio-muted-note"],
                            )

            refresh_outputs = [
                active_project,
                project_overview,
                reference_choice,
                style_choice,
                batch_reference,
                batch_style,
                voice_reference_choice,
                voice_style_choice,
                edit_source_choice,
                voice_model_choice,
                references_table,
                styles_table,
                assets_table,
                jobs_table,
                batch_jobs_table,
                rules_table,
                diagnostic_references_table,
                diagnostic_asset_choice,
                diagnostic_reference_choice,
                take_a,
                take_b,
                export_take_choice,
                cancel_job_choice,
                runtime_snapshot,
                sharing_snapshot,
                system_profile_settings,
                checkpoint_path,
                use_ema,
                trained_overview,
                trained_bakeoff_table,
                trained_base_audio,
                trained_ft_noema_audio,
                trained_ft_ema_audio,
                trained_longform_audio,
                trained_checkpoint_choice,
                trained_use_ema,
            ]

            load_preset_btn.click(load_preset, inputs=[preset_choice], outputs=[render_text])
            voice_load_preset_btn.click(load_preset, inputs=[voice_preset_choice], outputs=[voice_text])

            refresh_btn.click(project_updates, inputs=[active_project], outputs=refresh_outputs)
            active_project.change(project_updates, inputs=[active_project], outputs=refresh_outputs)

            create_project_btn.click(
                create_project,
                inputs=[project_name, project_description, active_project],
                outputs=refresh_outputs + [create_project_status],
            )

            save_reference_btn.click(
                save_reference,
                inputs=[active_project, reference_consent, reference_name, reference_audio, reference_text],
                outputs=[reference_text, reference_status] + refresh_outputs,
                concurrency_limit=1,
                concurrency_id="studio_compute",
            )

            save_style_btn.click(
                save_style,
                inputs=[active_project, style_name, style_audio, style_text, context_notes, render_text],
                outputs=[style_text, style_status] + refresh_outputs,
                concurrency_limit=1,
                concurrency_id="studio_compute",
            )

            save_edit_source_btn.click(
                save_edit_source,
                inputs=[active_project, edit_source_name, edit_source_audio, edit_source_text],
                outputs=[edit_source_text, edit_source_status, edit_alignment_preview] + refresh_outputs,
                concurrency_limit=1,
                concurrency_id="studio_compute",
            )

            edit_source_choice.change(
                load_edit_source_preview,
                inputs=[edit_source_choice],
                outputs=[edit_source_text, edit_alignment_preview, edit_source_audio_preview],
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
                    seed,
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
                    seed,
                ],
                outputs=[output_audio, output_spectrogram, render_status] + refresh_outputs,
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
                    seed,
                ],
                outputs=[output_audio, output_spectrogram, render_status] + refresh_outputs,
                concurrency_limit=1,
                concurrency_id="studio_compute",
            )

            voice_estimate_btn.click(
                estimate_voice_render,
                inputs=[
                    active_project,
                    voice_reference_choice,
                    voice_style_choice,
                    voice_model_choice,
                    voice_take_name,
                    voice_text,
                    voice_use_style_prompt,
                    voice_context_notes,
                    voice_speed,
                    voice_nfe_step,
                    voice_seed,
                ],
                outputs=[voice_status],
                concurrency_limit=1,
                concurrency_id="studio_compute",
            )

            voice_preview_btn.click(
                render_voice_preview,
                inputs=[
                    active_project,
                    voice_reference_choice,
                    voice_style_choice,
                    voice_model_choice,
                    voice_take_name,
                    voice_text,
                    voice_use_style_prompt,
                    voice_context_notes,
                    voice_speed,
                    voice_nfe_step,
                    voice_remove_silence,
                    voice_render_spectrogram,
                    voice_seed,
                ],
                outputs=[voice_output_audio, voice_output_spectrogram, voice_status] + refresh_outputs,
                concurrency_limit=1,
                concurrency_id="studio_compute",
            )

            voice_final_btn.click(
                render_voice_final,
                inputs=[
                    active_project,
                    voice_reference_choice,
                    voice_style_choice,
                    voice_model_choice,
                    voice_take_name,
                    voice_text,
                    voice_use_style_prompt,
                    voice_context_notes,
                    voice_speed,
                    voice_nfe_step,
                    voice_remove_silence,
                    voice_render_spectrogram,
                    voice_seed,
                ],
                outputs=[voice_output_audio, voice_output_spectrogram, voice_status] + refresh_outputs,
                concurrency_limit=1,
                concurrency_id="studio_compute",
            )

            edit_preview_btn.click(
                render_edit_preview,
                inputs=[
                    active_project,
                    edit_source_choice,
                    edit_take_name,
                    edit_target_text,
                    edit_replacement_text,
                    edit_occurrence,
                    edit_action,
                    edit_preserve_timing,
                    edit_nfe_step,
                    edit_render_spectrogram,
                ],
                outputs=[edit_output_audio, edit_output_spectrogram, edit_plan_preview, edit_status] + refresh_outputs,
                concurrency_limit=1,
                concurrency_id="studio_compute",
            )

            edit_final_btn.click(
                render_edit_final,
                inputs=[
                    active_project,
                    edit_source_choice,
                    edit_take_name,
                    edit_target_text,
                    edit_replacement_text,
                    edit_occurrence,
                    edit_action,
                    edit_preserve_timing,
                    edit_nfe_step,
                    edit_render_spectrogram,
                ],
                outputs=[edit_output_audio, edit_output_spectrogram, edit_plan_preview, edit_status] + refresh_outputs,
                concurrency_limit=1,
                concurrency_id="studio_compute",
            )

            diagnostic_run_btn.click(
                diagnose_take,
                inputs=[active_project, diagnostic_asset_choice, diagnostic_reference_choice, diagnostic_expected_text],
                outputs=[diagnostic_status, diagnostic_report],
                concurrency_limit=1,
                concurrency_id="studio_compute",
            )

            batch_submit_btn.click(
                submit_batch,
                inputs=[active_project, batch_reference, batch_style, batch_mode, batch_use_style, batch_context, batch_scripts],
                outputs=[batch_status] + refresh_outputs,
            )

            cancel_job_btn.click(
                cancel_job,
                inputs=[active_project, cancel_job_choice],
                outputs=[batch_status] + refresh_outputs,
            )

            compare_btn.click(
                compare_takes,
                inputs=[take_a, take_b],
                outputs=[compare_audio_a, compare_meta_a, compare_audio_b, compare_meta_b],
            )

            export_btn.click(export_take, inputs=[export_take_choice], outputs=[export_file, export_status])

            save_settings_btn.click(
                save_settings,
                inputs=[active_project, profile_choice, asr_backend, idle_unload_seconds, checkpoint_path, use_ema],
                outputs=refresh_outputs + [settings_status, trained_page_status],
            )

            warm_engine_btn.click(
                warm_engine,
                inputs=[active_project],
                outputs=refresh_outputs + [settings_status, trained_page_status],
                concurrency_limit=1,
                concurrency_id="studio_compute",
            )

            detect_checkpoint_btn.click(
                detect_checkpoint_widgets,
                outputs=[checkpoint_path, trained_checkpoint_choice, settings_status, trained_page_status],
            )

            trained_detect_checkpoint_btn.click(
                detect_checkpoint_widgets,
                outputs=[checkpoint_path, trained_checkpoint_choice, settings_status, trained_page_status],
            )

            trained_apply_btn.click(
                apply_trained_checkpoint,
                inputs=[active_project, trained_checkpoint_choice, trained_use_ema],
                outputs=refresh_outputs + [trained_page_status, settings_status],
            )

            trained_base_btn.click(
                switch_to_base_model,
                inputs=[active_project],
                outputs=refresh_outputs + [trained_page_status, settings_status],
            )

            save_rule_btn.click(
                save_rule,
                inputs=[active_project, rule_source, rule_replacement],
                outputs=[rule_status] + refresh_outputs,
            )

            app.load(project_updates, outputs=refresh_outputs)
            app.load(service.warm_profile_if_needed, outputs=[render_status])
            app.unload(service.maybe_unload_idle_engine)

    return app


def build_studio_app():
    return create_studio_app()
