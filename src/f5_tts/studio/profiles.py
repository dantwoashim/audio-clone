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
    ),
    "balanced": RuntimeProfile(
        name="balanced",
        label="Balanced",
        description="The default profile for everyday use on a 16GB M4 Air.",
        idle_unload_seconds=600,
        warm_on_start=True,
        preview_nfe_step=22,
        final_nfe_step=32,
        render_spectrogram_default=False,
        trim_silence_default=False,
        keep_asr_loaded=False,
        preview_char_limit=240,
    ),
    "quality": RuntimeProfile(
        name="quality",
        label="Quality",
        description="Maximizes output quality and keeps the engine warm longer, while still respecting a shared device.",
        idle_unload_seconds=1200,
        warm_on_start=True,
        preview_nfe_step=24,
        final_nfe_step=40,
        render_spectrogram_default=True,
        trim_silence_default=False,
        keep_asr_loaded=True,
        preview_char_limit=320,
    ),
}

DEFAULT_PROFILE = "balanced"


def get_runtime_profile(name: str | None) -> RuntimeProfile:
    if not name:
        return PROFILE_MAP[DEFAULT_PROFILE]
    return PROFILE_MAP.get(name, PROFILE_MAP[DEFAULT_PROFILE])
