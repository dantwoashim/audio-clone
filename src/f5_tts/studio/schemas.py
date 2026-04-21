from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)


class PronunciationRuleCreate(BaseModel):
    project_id: int
    source: str = Field(min_length=1, max_length=120)
    replacement: str = Field(min_length=1, max_length=240)


class EditRenderRequest(BaseModel):
    project_id: int
    source_asset_id: int
    name: str = Field(default="Speech Edit", max_length=120)
    target_text: str = Field(min_length=1)
    replacement_text: str = Field(min_length=1)
    occurrence: int = Field(default=1, ge=1)
    preserve_timing: bool = True
    nfe_step: int | None = None
    render_spectrogram: bool = False


class ReferenceAnalysis(BaseModel):
    transcript: str
    duration_seconds: float
    sample_rate: int
    channels: int
    rms: float
    peak: float
    trailing_silence_seconds: float
    speech_seconds: float | None = None
    speech_ratio: float | None = None
    backend: str
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class StyleAnalysis(BaseModel):
    transcript: str
    duration_seconds: float
    recommended_speed: float
    suggested_fix_duration: float | None = None
    matched_keywords: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class VoiceAssetView(BaseModel):
    id: int
    project_id: int
    kind: Literal["reference", "style"]
    name: str
    audio_path: str
    transcript: str
    analysis: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class PronunciationRuleView(BaseModel):
    id: int
    project_id: int
    source: str
    replacement: str
    created_at: str
    updated_at: str


class AudioAssetView(BaseModel):
    id: int
    project_id: int
    job_id: int | None = None
    kind: str
    label: str
    path: str
    duration_seconds: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class GenerationResultView(BaseModel):
    asset_id: int
    audio_path: str
    spectrogram_path: str | None = None
    duration_seconds: float
    elapsed_seconds: float
    sample_rate: int
    text_excerpt: str


class GenerationRequest(BaseModel):
    project_id: int
    reference_id: int
    style_id: int | None = None
    name: str = Field(default="Untitled Render", max_length=120)
    text: str = Field(min_length=1)
    mode: Literal["preview", "final"] = "final"
    use_style_prompt: bool = True
    context_notes: str = ""
    speed: float | None = None
    nfe_step: int | None = None
    cross_fade_duration: float = 0.15
    cfg_strength: float = 2.0
    sway_sampling_coef: float = -1.0
    remove_silence: bool | None = None
    render_spectrogram: bool | None = None
    checkpoint_path: str | None = None
    use_ema: bool | None = None


class GenerationEstimate(BaseModel):
    estimated_generation_seconds: float
    predicted_output_seconds: float
    chunks: int
    effective_speed: float
    queue_depth: int
    confidence: str


class GenerationJobView(BaseModel):
    id: int
    project_id: int
    name: str
    status: str
    recipe: dict[str, Any]
    result: dict[str, Any] | None = None
    error_text: str | None = None
    created_at: str
    updated_at: str


class ProjectSummary(BaseModel):
    id: int
    slug: str
    name: str
    description: str
    created_at: str
    updated_at: str


class ProjectDetail(ProjectSummary):
    references: list[VoiceAssetView] = Field(default_factory=list)
    styles: list[VoiceAssetView] = Field(default_factory=list)
    assets: list[AudioAssetView] = Field(default_factory=list)
    jobs: list[GenerationJobView] = Field(default_factory=list)
    pronunciation_rules: list[PronunciationRuleView] = Field(default_factory=list)


class SystemProfileView(BaseModel):
    profile: str
    profile_label: str
    description: str
    model_name: str
    checkpoint_path: str | None = None
    use_ema: bool = True
    engine_loaded: bool
    asr_backend: str
    asr_model: str
    device: str
    queue_depth: int
    worker_alive: bool
    current_job_id: int | None = None
    idle_unload_seconds: int
    root_path: str
    cache_path: str
    last_warm_error: str | None = None
