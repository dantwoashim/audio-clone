import tempfile
import threading
import time

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio

from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import (
    chunk_text,
    hop_length,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    target_sample_rate,
    tempfile_kwargs,
)


TITLE = "Voice Studio"

APP_CSS = """
:root {
  --studio-ink: #16231b;
  --studio-muted: #5f6c63;
  --studio-paper: #f5f1e8;
  --studio-card: rgba(255, 252, 246, 0.88);
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
  max-width: 1240px;
  margin: 0 auto;
  padding: 28px 20px 36px;
}

.studio-hero {
  position: relative;
  overflow: hidden;
  border: 1px solid var(--studio-line);
  border-radius: 28px;
  padding: 30px;
  margin-bottom: 24px;
  background:
    linear-gradient(135deg, rgba(255, 249, 240, 0.92), rgba(243, 233, 214, 0.72)),
    linear-gradient(135deg, rgba(15, 124, 91, 0.1), rgba(217, 91, 67, 0.1));
  box-shadow: 0 30px 80px rgba(60, 48, 28, 0.08);
}

.studio-hero::after {
  content: "";
  position: absolute;
  inset: auto -8% -24% auto;
  width: 320px;
  height: 320px;
  border-radius: 999px;
  background: radial-gradient(circle, rgba(15, 124, 91, 0.16), transparent 65%);
  pointer-events: none;
}

.studio-kicker {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 14px;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(22, 35, 27, 0.06);
  color: var(--studio-muted);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.studio-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) minmax(280px, 0.9fr);
  gap: 18px;
  align-items: end;
}

.studio-display h1 {
  margin: 0;
  font-size: clamp(2.8rem, 6vw, 4.8rem);
  line-height: 0.92;
  letter-spacing: -0.06em;
  font-weight: 700;
}

.studio-display p {
  max-width: 640px;
  margin: 18px 0 0;
  color: var(--studio-muted);
  font-size: 1.02rem;
  line-height: 1.65;
}

.studio-right {
  display: grid;
  gap: 12px;
}

.studio-metric {
  border: 1px solid rgba(22, 35, 27, 0.1);
  border-radius: 22px;
  padding: 18px 18px 16px;
  background: rgba(255, 250, 244, 0.75);
  backdrop-filter: blur(6px);
}

.studio-metric strong {
  display: block;
  margin-bottom: 8px;
  font-size: 0.82rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--studio-muted);
}

.studio-metric span {
  display: block;
  font-size: 1.4rem;
  line-height: 1.1;
  font-weight: 700;
}

.studio-metric p {
  margin: 8px 0 0;
  color: var(--studio-muted);
  line-height: 1.5;
  font-size: 0.95rem;
}

.studio-workspace,
.studio-panel {
  border: 1px solid var(--studio-line);
  border-radius: 26px;
  background: var(--studio-card);
  backdrop-filter: blur(10px);
  box-shadow: 0 18px 55px rgba(60, 48, 28, 0.07);
}

.studio-workspace {
  padding: 14px;
}

.studio-panel {
  padding: 18px;
}

.studio-step {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  border-radius: 999px;
  margin-bottom: 12px;
  background: rgba(15, 124, 91, 0.12);
  color: var(--studio-accent);
  font-weight: 700;
}

.studio-panel h3 {
  margin: 0 0 8px;
  font-size: 1.35rem;
  letter-spacing: -0.03em;
}

.studio-panel p,
.studio-notes li {
  color: var(--studio-muted);
  line-height: 1.6;
}

.studio-notes {
  margin: 0;
  padding-left: 18px;
}

.studio-divider {
  height: 1px;
  margin: 18px 0;
  background: linear-gradient(90deg, transparent, rgba(22, 35, 27, 0.12), transparent);
}

.studio-status {
  border-radius: 22px;
  border: 1px solid rgba(15, 124, 91, 0.12);
  background: linear-gradient(180deg, rgba(15, 124, 91, 0.08), rgba(255, 255, 255, 0.44));
}

.studio-status h4 {
  margin: 0 0 10px;
  font-size: 1rem;
}

.studio-preset button,
.studio-generate button {
  min-height: 52px;
  border-radius: 18px !important;
}

.studio-generate button {
  background: linear-gradient(135deg, #0f7c5b, #136d87) !important;
  box-shadow: 0 16px 35px var(--studio-glow);
}

.studio-footer {
  margin-top: 22px;
  color: var(--studio-muted);
  font-size: 0.92rem;
}

@media (max-width: 960px) {
  .studio-grid {
    grid-template-columns: 1fr;
  }
}
"""

HERO_HTML = f"""
<section class="studio-hero">
  <div class="studio-kicker">F5-TTS Voice Cloning</div>
  <div class="studio-grid">
    <div class="studio-display">
      <h1>{TITLE}</h1>
      <p>
        Start with a clean voice sample, confirm the transcript, then generate polished speech in a few guided steps.
        This studio keeps the main workflow simple while still giving you room to tune quality when you need it.
      </p>
    </div>
    <div class="studio-right">
      <div class="studio-metric">
        <strong>Best Input</strong>
        <span>6-12 seconds</span>
        <p>Short, clean audio with one speaker and a natural ending gives the strongest clone.</p>
      </div>
      <div class="studio-metric">
        <strong>Core Flow</strong>
        <span>Sample -> Script -> Audio</span>
        <p>Keep the journey obvious: upload, verify, write, then synthesize.</p>
      </div>
    </div>
  </div>
</section>
"""

EXAMPLE_SCRIPT = """Hello there. This demo uses my reference voice to read new text with a calm, natural delivery.

I can use it for short explainers, previews, character tests, and prototype narration."""

SCRIPT_PRESETS = {
    "Product demo": """Welcome to Voice Studio. Drop in a clean reference clip, confirm the transcript, and generate a polished preview in seconds.

You can adjust delivery speed and sampling steps later if you want a more refined pass.""",
    "Podcast intro": """Welcome back to the show. Today we're testing a fresh voice clone using F5-TTS and a much simpler studio workflow.

If this sounds right, we can turn it into a full intro package next.""",
    "Support voiceover": """Thanks for calling. Your request has been received, and we're preparing an update for you now.

Please stay with us for just a moment while we finish the next step.""",
}

f5tts = None
engine_init_lock = threading.Lock()
engine_status = {
    "ready": False,
    "warming": False,
}
timing_stats = {
    "normalized_rtf_samples": [],
}
CONTEXT_MODIFIERS = {
    "slow": {"speed": 0.9, "duration": 1.12},
    "calm": {"speed": 0.93, "duration": 1.08},
    "gentle": {"speed": 0.94, "duration": 1.08},
    "soft": {"speed": 0.95, "duration": 1.05},
    "whisper": {"speed": 0.92, "duration": 1.1},
    "fast": {"speed": 1.08, "duration": 0.92},
    "urgent": {"speed": 1.12, "duration": 0.9},
    "energetic": {"speed": 1.08, "duration": 0.94},
    "excited": {"speed": 1.06, "duration": 0.95},
    "dramatic": {"speed": 0.95, "duration": 1.12},
    "cinematic": {"speed": 0.96, "duration": 1.1},
    "narrator": {"speed": 0.95, "duration": 1.08},
}


def get_engine():
    global f5tts
    if f5tts is None:
        with engine_init_lock:
            if f5tts is None:
                f5tts = F5TTS(model="F5TTS_v1_Base")
    return f5tts


def warm_engine():
    if engine_status["ready"] or engine_status["warming"]:
        return "Voice engine is ready."

    def _worker():
        engine_status["warming"] = True
        try:
            get_engine().warm_up(show_info=lambda *_args, **_kwargs: None)
            engine_status["ready"] = True
        finally:
            engine_status["warming"] = False

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return "Voice engine is warming up in the background. The first render should be noticeably faster after this."


def format_duration(seconds):
    seconds = max(float(seconds), 0.0)
    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remainder:06.3f}"
    return f"{minutes:02d}:{remainder:06.3f}"


def get_default_normalized_rtf():
    device = get_engine().device
    if str(device).startswith("mps"):
        return 1.15
    if str(device).startswith("cuda"):
        return 0.22
    if str(device).startswith("xpu"):
        return 0.75
    return 2.4


def get_normalized_rtf():
    samples = timing_stats["normalized_rtf_samples"]
    if samples:
        return sum(samples) / len(samples)
    return get_default_normalized_rtf()


def estimate_output_seconds(prepared_reference, gen_text, speed):
    audio = prepared_reference["audio"]
    sample_rate = prepared_reference["sample_rate"]
    ref_text = prepared_reference["ref_text"]
    ref_duration_seconds = audio.shape[-1] / sample_rate
    ref_audio_len = int(ref_duration_seconds * target_sample_rate / hop_length)
    ref_text_len = len(ref_text.encode("utf-8"))
    max_chars = int(ref_text_len / ref_duration_seconds * (22 - ref_duration_seconds) * speed)
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)

    total_output_seconds = 0.0
    for gen_text_chunk in gen_text_batches:
        local_speed = speed
        if len(gen_text_chunk.encode("utf-8")) < 10:
            local_speed = 0.3
        gen_text_len = len(gen_text_chunk.encode("utf-8"))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)
        output_frames = max(duration - ref_audio_len, 0)
        total_output_seconds += output_frames * hop_length / target_sample_rate
    return total_output_seconds, len(gen_text_batches)


def estimate_generation_time(ref_audio, ref_text, gen_text, style_state, use_style_prompt, speed, nfe_step):
    if not ref_audio:
        return "Upload a reference clip first."
    if not gen_text.strip():
        return "Enter a script to estimate generation time."

    prepared_reference = get_engine().prepare_reference(ref_audio, ref_text, show_info=lambda *_args, **_kwargs: None)
    effective_speed = speed
    output_seconds = None
    if use_style_prompt and style_state is not None:
        effective_speed = style_state["recommended_speed"]
        if style_state["suggested_fix_duration"] is not None:
            ref_duration = prepared_reference["audio"].shape[-1] / prepared_reference["sample_rate"]
            output_seconds = max(style_state["suggested_fix_duration"] - ref_duration, 0.0)

    if output_seconds is None:
        output_seconds, batch_count = estimate_output_seconds(prepared_reference, gen_text, effective_speed)
    else:
        _, batch_count = estimate_output_seconds(prepared_reference, gen_text, effective_speed)
    normalized_rtf = get_normalized_rtf()
    estimated_compute_seconds = output_seconds * normalized_rtf * (nfe_step / 32)
    cold_start_penalty = 0.0 if engine_status["ready"] else 2.5
    estimated_total_seconds = estimated_compute_seconds + cold_start_penalty + max(batch_count - 1, 0) * 0.12

    confidence = "calibrated from your recent runs" if timing_stats["normalized_rtf_samples"] else "first-run estimate"
    return (
        f"Estimated generation time: {format_duration(estimated_total_seconds)}\n"
        f"Predicted audio length: {format_duration(output_seconds)}\n"
        f"Chunks: {batch_count}\n"
        f"Applied speed estimate: {effective_speed:.2f}\n"
        f"Estimate source: {confidence}"
    )


def apply_context_modifiers(base_speed, base_duration, context_notes):
    if not context_notes.strip():
        return base_speed, base_duration, []

    speed = base_speed
    duration = base_duration
    notes = []
    lowered = context_notes.lower()
    for keyword, modifier in CONTEXT_MODIFIERS.items():
        if keyword in lowered:
            speed *= modifier["speed"]
            if duration is not None:
                duration *= modifier["duration"]
            notes.append(keyword)

    speed = min(max(speed, 0.6), 1.5)
    if duration is not None:
        duration = max(duration, 0.5)
    return speed, duration, notes


def analyze_style_prompt(style_audio, style_text, context_notes, ref_audio, ref_text, gen_text):
    if not style_audio and not context_notes.strip():
        return style_text, gr.update(), "No style prompt or context provided yet.", None

    engine = get_engine()
    style_summary = []
    inferred_style_text = style_text
    recommended_speed = 1.0
    suggested_fix_duration = None

    if style_audio:
        if not inferred_style_text.strip():
            inferred_style_text = engine.transcribe(style_audio)
        style_audio_tensor, style_sample_rate = torchaudio.load(style_audio)
        style_duration_seconds = style_audio_tensor.shape[-1] / style_sample_rate
        style_text_len = max(len(inferred_style_text.encode("utf-8")), 1)
        style_rate = style_text_len / style_duration_seconds
        recommended_speed = min(max(style_rate / 12.5, 0.72), 1.28)
        style_summary.append(f"Style prompt length: {format_duration(style_duration_seconds)}")
        style_summary.append(f"Style transcript: {inferred_style_text.strip()}")
        style_summary.append(f"Style-derived speed: {recommended_speed:.2f}")

        if ref_audio and gen_text.strip():
            prepared_reference = engine.prepare_reference(ref_audio, ref_text, show_info=lambda *_args, **_kwargs: None)
            ref_duration_seconds = prepared_reference["audio"].shape[-1] / prepared_reference["sample_rate"]
            target_text_len = max(len(gen_text.encode("utf-8")), 1)
            suggested_gen_seconds = style_duration_seconds * (target_text_len / style_text_len)
            suggested_fix_duration = ref_duration_seconds + suggested_gen_seconds

    recommended_speed, suggested_fix_duration, matched_keywords = apply_context_modifiers(
        recommended_speed,
        suggested_fix_duration,
        context_notes,
    )
    if matched_keywords:
        style_summary.append(f"Context matched: {', '.join(matched_keywords)}")

    style_state = {
        "style_text": inferred_style_text,
        "context_notes": context_notes,
        "recommended_speed": recommended_speed,
        "suggested_fix_duration": suggested_fix_duration,
    }

    if suggested_fix_duration is not None:
        style_summary.append(f"Suggested total duration target: {format_duration(suggested_fix_duration)}")
    style_summary.append("Voice identity still comes from the main reference. The style prompt guides pacing and delivery.")
    return (
        inferred_style_text,
        gr.update(value=recommended_speed),
        "\n".join(style_summary),
        style_state,
    )


def set_script_preset(preset_name):
    return gr.update(value=SCRIPT_PRESETS[preset_name])


def inspect_reference(ref_audio, ref_text):
    if not ref_audio:
        raise gr.Error("Upload a reference audio clip first.")

    processed_audio, processed_text = preprocess_ref_audio_text(ref_audio, ref_text, show_info=gr.Info)
    audio, sample_rate = torchaudio.load(processed_audio)
    duration = audio.shape[-1] / sample_rate
    summary = (
        f"Prepared reference clip: {duration:.2f}s at {sample_rate} Hz.\n"
        f"Transcript ready: {processed_text.strip()}"
    )
    return processed_text, summary, get_engine().prepare_reference(ref_audio, processed_text, show_info=lambda *_args, **_kwargs: None)


def render_spectrogram(spec):
    if spec is None:
        raise gr.Error("Generate audio first, then render the spectrogram when you need it.")

    with tempfile.NamedTemporaryFile(suffix=".png", **tempfile_kwargs) as spec_file:
        spec_path = spec_file.name
    save_spectrogram(spec, spec_path)
    return spec_path


def generate_voice(
    _prepared_reference,
    ref_audio,
    ref_text,
    gen_text,
    style_state,
    use_style_prompt,
    speed,
    nfe_step,
    cross_fade_duration,
    cfg_strength,
    sway_sampling_coef,
    remove_silence,
    render_spec,
):
    if not ref_audio:
        raise gr.Error("Upload a reference audio clip before generating.")
    if not gen_text.strip():
        raise gr.Error("Enter the script you want the cloned voice to speak.")

    engine = get_engine()
    prepared_reference = engine.prepare_reference(ref_audio, ref_text, show_info=gr.Info)
    started_at = time.perf_counter()
    effective_speed = speed
    effective_fix_duration = None

    if use_style_prompt and style_state is not None:
        effective_speed = style_state["recommended_speed"]
        effective_fix_duration = style_state["suggested_fix_duration"]

    wav, sr, spec = engine.infer_prepared(
        prepared_reference,
        gen_text=gen_text,
        speed=effective_speed,
        nfe_step=nfe_step,
        cross_fade_duration=cross_fade_duration,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        progress=None,
        remove_silence=False,
        fix_duration=effective_fix_duration,
    )
    elapsed_seconds = time.perf_counter() - started_at

    if wav is None or spec is None:
        raise gr.Error("Generation returned an empty result. Try a shorter script or a cleaner reference clip.")

    with tempfile.NamedTemporaryFile(suffix=".wav", **tempfile_kwargs) as wav_file:
        wav_path = wav_file.name
    sf.write(wav_path, wav, sr)
    if remove_silence:
        remove_silence_for_generated_wav(wav_path)
        wav, _ = torchaudio.load(wav_path)
        wav = wav.squeeze().cpu().numpy()

    spec_path = None
    if render_spec:
        spec_path = render_spectrogram(spec)

    words = len(gen_text.split())
    runtime_seconds = len(wav) / sr if len(wav) else 0
    if runtime_seconds > 0:
        normalized_rtf = (elapsed_seconds / runtime_seconds) * (32 / nfe_step)
        timing_stats["normalized_rtf_samples"].append(normalized_rtf)
        timing_stats["normalized_rtf_samples"][:] = timing_stats["normalized_rtf_samples"][-6:]
    status = (
        f"Generation complete.\n"
        f"Actual generation time: {format_duration(elapsed_seconds)}\n"
        f"Words: {words}\n"
        f"Output length: {format_duration(runtime_seconds)}\n"
        f"Applied speed: {effective_speed:.2f}\n"
        f"Model: F5TTS_v1_Base on Apple Silicon-optimized single-device path"
    )
    if effective_fix_duration is not None:
        status += f"\nStyle-guided duration target: {format_duration(effective_fix_duration)}"

    return (sr, np.asarray(wav)), spec_path, status, spec, prepared_reference


def build_app():
    theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="orange",
        neutral_hue="stone",
        radius_size=gr.themes.sizes.radius_lg,
        text_size=gr.themes.sizes.text_md,
    )

    with gr.Blocks(title=TITLE, theme=theme, css=APP_CSS) as app:
        with gr.Column(elem_classes=["studio-shell"]):
            gr.HTML(HERO_HTML)
            prepared_reference = gr.State(value=None)
            latest_spectrogram = gr.State(value=None)
            style_state = gr.State(value=None)

            with gr.Row():
                with gr.Column(scale=5, elem_classes=["studio-workspace"]):
                    with gr.Row():
                        with gr.Column(scale=6, elem_classes=["studio-panel"]):
                            gr.HTML('<div class="studio-step">1</div>')
                            gr.Markdown(
                                "### Bring in your voice\nUpload a clean clip from the speaker you want to clone. "
                                "Reference text is optional, but confirming it improves accuracy."
                            )
                            ref_audio = gr.Audio(
                                label="Reference audio",
                                type="filepath",
                                sources=["upload", "microphone"],
                            )
                            ref_text = gr.Textbox(
                                label="Reference transcript",
                                lines=4,
                                placeholder="Optional. Leave blank to auto-transcribe the sample.",
                            )
                            with gr.Row():
                                inspect_btn = gr.Button("Inspect sample", variant="secondary", elem_classes=["studio-preset"])
                                warm_btn = gr.Button("Warm engine", variant="secondary", elem_classes=["studio-preset"])
                                clear_btn = gr.Button("Clear", variant="stop")

                        with gr.Column(scale=7, elem_classes=["studio-panel"]):
                            gr.HTML('<div class="studio-step">2</div>')
                            gr.Markdown(
                                "### Write the script\nKeep it natural and conversational. Longer passages will be chunked automatically."
                            )
                            with gr.Row():
                                preset_choice = gr.Dropdown(
                                    choices=list(SCRIPT_PRESETS.keys()),
                                    value="Product demo",
                                    label="Starter preset",
                                    scale=3,
                                )
                                load_preset_btn = gr.Button(
                                    "Load preset", variant="secondary", elem_classes=["studio-preset"], scale=1
                                )
                            gen_text = gr.Textbox(
                                label="Target script",
                                lines=12,
                                value=EXAMPLE_SCRIPT,
                                placeholder="Type the new text you want the cloned voice to speak.",
                            )
                            gr.Markdown(
                                "### Style and context prompt\nUse this optional layer when you want to keep the same speaker identity but steer pacing, mood, and delivery from a second audio example."
                            )
                            style_audio = gr.Audio(
                                label="Style prompt audio",
                                type="filepath",
                                sources=["upload", "microphone"],
                            )
                            style_text = gr.Textbox(
                                label="Style prompt transcript",
                                lines=3,
                                placeholder="Optional. Leave blank to auto-transcribe the style audio.",
                            )
                            context_notes = gr.Textbox(
                                label="Delivery context",
                                lines=3,
                                placeholder="Examples: calm product demo, urgent support voice, cinematic narrator, soft whisper.",
                            )
                            style_status = gr.Textbox(
                                label="Style guidance",
                                interactive=False,
                                value="Optional. Analyze a style prompt to turn it into timing and delivery guidance.",
                                lines=6,
                            )
                            with gr.Row():
                                analyze_style_btn = gr.Button(
                                    "Analyze style prompt",
                                    variant="secondary",
                                    elem_classes=["studio-preset"],
                                )
                                use_style_prompt = gr.Checkbox(
                                    label="Use style prompt during generation",
                                    value=True,
                                )

                    with gr.Row():
                        with gr.Column(scale=5, elem_classes=["studio-panel", "studio-status"]):
                            gr.HTML('<div class="studio-step">3</div>')
                            gr.Markdown("### Generate\nUse the default settings for the first pass, then refine if needed.")
                            with gr.Accordion("Advanced tuning", open=False):
                                speed = gr.Slider(label="Speed", minimum=0.6, maximum=1.5, value=1.0, step=0.05)
                                nfe_step = gr.Slider(label="NFE steps", minimum=8, maximum=64, value=32, step=2)
                                cross_fade_duration = gr.Slider(
                                    label="Cross-fade duration", minimum=0.0, maximum=0.6, value=0.15, step=0.01
                                )
                                cfg_strength = gr.Slider(
                                    label="CFG strength", minimum=1.0, maximum=4.0, value=2.0, step=0.1
                                )
                                sway_sampling_coef = gr.Slider(
                                    label="Sway sampling", minimum=-1.0, maximum=1.0, value=-1.0, step=0.1
                                )
                                remove_silence = gr.Checkbox(
                                    label="Trim long silence in the final output",
                                    value=False,
                                )
                                render_spec = gr.Checkbox(
                                    label="Render spectrogram during generation",
                                    value=False,
                                )
                            generate_btn = gr.Button(
                                "Generate voice clone", variant="primary", elem_classes=["studio-generate"]
                            )
                            estimate_btn = gr.Button(
                                "Estimate generation time",
                                variant="secondary",
                                elem_classes=["studio-preset"],
                            )
                            render_spec_btn = gr.Button(
                                "Render spectrogram now",
                                variant="secondary",
                                elem_classes=["studio-preset"],
                            )
                            run_status = gr.Textbox(
                                label="Status",
                                interactive=False,
                                value="Ready for your first pass. On an M4 Air, keeping the same reference and leaving spectrogram rendering off will feel noticeably faster.",
                                lines=4,
                            )

                        with gr.Column(scale=7, elem_classes=["studio-panel"]):
                            gr.Markdown("### Output")
                            generated_audio = gr.Audio(label="Generated audio")
                            spectrogram = gr.Image(label="Spectrogram")

                with gr.Column(scale=3):
                    with gr.Column(elem_classes=["studio-panel"]):
                        gr.Markdown("### Quick guidance")
                        gr.Markdown(
                            """
<ul class="studio-notes">
  <li>Use a single speaker with low background noise.</li>
  <li>Clips around 6 to 12 seconds usually work best.</li>
  <li>If pronunciation feels off, edit the reference transcript manually.</li>
  <li>Generate a short sample first before committing to a long script.</li>
  <li>Use the style prompt for delivery, not identity. It tells the system how to speak, not who is speaking.</li>
</ul>
"""
                        )
                        gr.HTML('<div class="studio-divider"></div>')
                        sample_status = gr.Textbox(
                            label="Reference check",
                            interactive=False,
                            value="Upload a clip, then inspect it here.",
                            lines=5,
                        )

                    with gr.Column(elem_classes=["studio-panel"]):
                        gr.Markdown("### What this app is for")
                        gr.Markdown(
                            """
Voice Studio is a guided layer on top of F5-TTS. It focuses on the main cloning workflow rather than the full
multi-speaker demo, which makes it easier to hand to non-technical users.
"""
                        )

            gr.Markdown(
                '<div class="studio-footer">Launch this app with <code>f5-tts_voice-studio</code> after installing the package in editable mode.</div>'
            )

            load_preset_btn.click(set_script_preset, inputs=[preset_choice], outputs=[gen_text])
            inspect_btn.click(
                inspect_reference,
                inputs=[ref_audio, ref_text],
                outputs=[ref_text, sample_status, prepared_reference],
            )
            analyze_style_btn.click(
                analyze_style_prompt,
                inputs=[style_audio, style_text, context_notes, ref_audio, ref_text, gen_text],
                outputs=[style_text, speed, style_status, style_state],
            )
            warm_btn.click(warm_engine, outputs=[run_status])
            estimate_btn.click(
                estimate_generation_time,
                inputs=[ref_audio, ref_text, gen_text, style_state, use_style_prompt, speed, nfe_step],
                outputs=[run_status],
            )
            generate_btn.click(
                generate_voice,
                inputs=[
                    prepared_reference,
                    ref_audio,
                    ref_text,
                    gen_text,
                    style_state,
                    use_style_prompt,
                    speed,
                    nfe_step,
                    cross_fade_duration,
                    cfg_strength,
                    sway_sampling_coef,
                    remove_silence,
                    render_spec,
                ],
                outputs=[generated_audio, spectrogram, run_status, latest_spectrogram, prepared_reference],
            )
            render_spec_btn.click(
                render_spectrogram,
                inputs=[latest_spectrogram],
                outputs=[spectrogram],
            )
            clear_btn.click(
                lambda: (
                    None,
                    "",
                    EXAMPLE_SCRIPT,
                    None,
                    "",
                    "",
                    "Optional. Analyze a style prompt to turn it into timing and delivery guidance.",
                    "Upload a clip, then inspect it here.",
                    "Ready for your first pass. On an M4 Air, keeping the same reference and leaving spectrogram rendering off will feel noticeably faster.",
                    None,
                    None,
                    None,
                    None,
                    1.0,
                ),
                outputs=[
                    ref_audio,
                    ref_text,
                    gen_text,
                    style_audio,
                    style_text,
                    context_notes,
                    style_status,
                    sample_status,
                    run_status,
                    spectrogram,
                    prepared_reference,
                    latest_spectrogram,
                    style_state,
                    speed,
                ],
            )
            app.load(warm_engine, outputs=[run_status])

    return app


app = build_app()


@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option("--share", "-s", default=False, is_flag=True, help="Share the app via Gradio share link")
@click.option("--inbrowser", "-i", is_flag=True, default=False, help="Automatically open the app in your browser")
def main(port, host, share, inbrowser):
    app.queue().launch(server_name=host, server_port=port, share=share, inbrowser=inbrowser)


if __name__ == "__main__":
    main()
