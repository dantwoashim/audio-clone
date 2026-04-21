from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    label: str
    description: str
    idle_unload_seconds: int
    warm_on_start: bool
    preview_nfe_step: int
    final_nfe_step: int
    render_spectrogram_default: bool
    trim_silence_default: bool
    keep_asr_loaded: bool
    preview_char_limit: int
    preferred_asr_backend: str
    preferred_asr_model: str
    mps_high_watermark_ratio: float
    mps_low_watermark_ratio: float


PROFILE_MAP = {
    "eco": RuntimeProfile(
        name="eco",
        label="Eco",
        description="Keeps the machine responsive. Best when the M4 Air is doing other work too.",
        idle_unload_seconds=300,
        warm_on_start=False,
        preview_nfe_step=18,
        final_nfe_step=28,
        render_spectrogram_default=False,
        trim_silence_default=False,
        keep_asr_loaded=False,
        preview_char_limit=180,
        preferred_asr_backend="mlx_whisper",
        preferred_asr_model="mlx-community/whisper-small",
        mps_high_watermark_ratio=0.92,
        mps_low_watermark_ratio=0.78,
    ),
    "balanced": RuntimeProfile(
        name="balanced",
        label="Balanced",
        description="The default profile for everyday use on a 16GB M4 Air.",
        idle_unload_seconds=600,
        warm_on_start=False,
        preview_nfe_step=22,
        final_nfe_step=32,
        render_spectrogram_default=False,
        trim_silence_default=False,
        keep_asr_loaded=False,
        preview_char_limit=240,
        preferred_asr_backend="mlx_whisper",
        preferred_asr_model="mlx-community/whisper-medium",
        mps_high_watermark_ratio=1.0,
        mps_low_watermark_ratio=0.9,
    ),
    "quality": RuntimeProfile(
        name="quality",
        label="Quality",
        description="Maximizes output quality, but only warms the engine when you explicitly ask for it.",
        idle_unload_seconds=1200,
        warm_on_start=False,
        preview_nfe_step=24,
        final_nfe_step=40,
        render_spectrogram_default=True,
        trim_silence_default=False,
        keep_asr_loaded=False,
        preview_char_limit=320,
        preferred_asr_backend="mlx_whisper",
        preferred_asr_model="mlx-community/whisper-medium",
        mps_high_watermark_ratio=1.0,
        mps_low_watermark_ratio=0.92,
    ),
}

DEFAULT_PROFILE = "balanced"


def get_runtime_profile(name: str | None) -> RuntimeProfile:
    if not name:
        return PROFILE_MAP[DEFAULT_PROFILE]
    return PROFILE_MAP.get(name, PROFILE_MAP[DEFAULT_PROFILE])
