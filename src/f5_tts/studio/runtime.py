from __future__ import annotations

import gc
import html
import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

from f5_tts.studio.diagnostics import diagnose_render, reference_quality_breakdown
from f5_tts.studio.paths import StudioPaths, get_studio_paths
from f5_tts.studio.profiles import DEFAULT_PROFILE, PROFILE_MAP, get_runtime_profile
from f5_tts.studio.schemas import (
    GenerationEstimate,
    GenerationRequest,
    ReferenceAnalysis,
    StyleAnalysis,
    SystemProfileView,
)
from f5_tts.studio.security import ensure_upload_within_limit, get_security_settings
from f5_tts.studio.storage import StudioStore


if TYPE_CHECKING:
    from f5_tts.api import F5TTS


DEFAULT_ASR_BACKEND = "auto"
DEFAULT_INFERENCE_BACKEND = "mlx" if torch.backends.mps.is_available() else "pytorch"
MLX_BACKEND_MODEL = "lucasnewman/f5-tts-mlx"

CONTEXT_MODIFIERS = {
    "slow": {"speed": 0.9, "duration": 1.12},
    "calm": {"speed": 0.93, "duration": 1.08},
    "gentle": {"speed": 0.94, "duration": 1.08},
    "soft": {"speed": 0.95, "duration": 1.05},
    "whisper": {"speed": 0.92, "duration": 1.10},
    "fast": {"speed": 1.08, "duration": 0.92},
    "urgent": {"speed": 1.12, "duration": 0.90},
    "energetic": {"speed": 1.08, "duration": 0.94},
    "excited": {"speed": 1.06, "duration": 0.95},
    "dramatic": {"speed": 0.95, "duration": 1.12},
    "cinematic": {"speed": 0.96, "duration": 1.10},
    "narrator": {"speed": 0.95, "duration": 1.08},
}

tempfile_kwargs = {"delete_on_close": False} if sys.version_info >= (3, 12) else {"delete": False}
target_sample_rate = 24000
hop_length = 256

_SMALL_NUMBERS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_TIMING_TOKEN_PATTERN = re.compile(r"[\u3400-\u9FFF]|[A-Za-z0-9']+|[^\s]")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")


class JobCancelledError(RuntimeError):
    pass


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remainder:05.2f}"
    return f"{minutes:02d}:{remainder:05.2f}"


def chunk_text(text: str, max_chars: int = 135) -> list[str]:
    chunks = []
    current_chunk = ""
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)
    for sentence in sentences:
        if not sentence:
            continue
        if text_timing_units(current_chunk) + text_timing_units(sentence) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def text_timing_units(text: str) -> int:
    return len(_TIMING_TOKEN_PATTERN.findall(text.strip()))


def word_tokens(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text.lower())


def number_to_words(value: int) -> str:
    if value < 20:
        return _SMALL_NUMBERS[value]
    if value < 100:
        tens, remainder = divmod(value, 10)
        return _TENS[tens] if remainder == 0 else f"{_TENS[tens]} {_SMALL_NUMBERS[remainder]}"
    if value < 1000:
        hundreds, remainder = divmod(value, 100)
        prefix = f"{_SMALL_NUMBERS[hundreds]} hundred"
        return prefix if remainder == 0 else f"{prefix} {number_to_words(remainder)}"
    if value < 10000:
        thousands, remainder = divmod(value, 1000)
        prefix = f"{_SMALL_NUMBERS[thousands]} thousand"
        return prefix if remainder == 0 else f"{prefix} {number_to_words(remainder)}"
    return str(value)


def normalize_script(text: str, pronunciation_rules: list[dict]) -> str:
    text = text.strip()
    if not text:
        return text

    for rule in pronunciation_rules:
        pattern = re.compile(rf"\b{re.escape(rule['source'])}\b", flags=re.IGNORECASE)
        text = pattern.sub(rule["replacement"], text)

    def replace_time(match: re.Match[str]) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if minute == 0:
            return f"{number_to_words(hour)} o'clock"
        return f"{number_to_words(hour)} {number_to_words(minute)}"

    def replace_number(match: re.Match[str]) -> str:
        value = int(match.group(0))
        if value > 9999:
            return match.group(0)
        return number_to_words(value)

    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", replace_time, text)
    text = re.sub(r"\b\d{1,4}\b", replace_number, text)
    text = re.sub(r"([,.;!?])(?=\S)", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_context_modifiers(
    base_speed: float, base_duration: float | None, context_notes: str
) -> tuple[float, float | None, list[str]]:
    if not context_notes.strip():
        return base_speed, base_duration, []

    speed = base_speed
    duration = base_duration
    matched: list[str] = []
    lowered = context_notes.lower()
    for keyword, modifier in CONTEXT_MODIFIERS.items():
        if keyword in lowered:
            speed *= modifier["speed"]
            if duration is not None:
                duration *= modifier["duration"]
            matched.append(keyword)

    speed = min(max(speed, 0.6), 1.5)
    if duration is not None:
        duration = max(duration, 0.5)
    return speed, duration, matched


def release_memory() -> None:
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
    except Exception:
        pass
    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except Exception:
        pass


class StudioEngine:
    def __init__(self, store: StudioStore):
        self.store = store
        self._lock = threading.Lock()
        self._engine: F5TTS | None = None
        self._engine_signature: tuple[str, str, bool] | None = None
        self._mlx_engine = None
        self._mlx_engine_signature: tuple[str, int | None] | None = None
        self._silero_model = None
        self._last_used_at = 0.0
        self._timing_samples: dict[str, list[float]] = {"pytorch": [], "mlx": []}
        self._reference_cache: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self._reference_cache_limit = 8

    @property
    def device(self) -> str:
        if self._engine is not None:
            return str(self._engine.device)
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def mlx_available(self) -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            import f5_tts_mlx  # noqa: F401
        except Exception:
            return False
        return True

    def _touch(self) -> None:
        self._last_used_at = time.time()

    def is_loaded(self, backend: str | None = None) -> bool:
        if backend == "mlx":
            return self._mlx_engine is not None
        if backend == "pytorch":
            return self._engine is not None
        return self._engine is not None or self._mlx_engine is not None

    def ensure_engine(self, backend: str = "pytorch", ckpt_file: str | None = None, use_ema: bool = True):
        if backend == "mlx":
            if not self.mlx_available():
                raise RuntimeError("The Apple MLX backend is not available on this machine.")
            if ckpt_file:
                raise RuntimeError("The Apple MLX backend currently supports only the shipped base model.")
            signature = (MLX_BACKEND_MODEL, None)
            if self._mlx_engine is None or self._mlx_engine_signature != signature:
                with self._lock:
                    if self._mlx_engine is None or self._mlx_engine_signature != signature:
                        from f5_tts_mlx import F5TTS as MLXF5TTS

                        self._mlx_engine = MLXF5TTS.from_pretrained(MLX_BACKEND_MODEL)
                        self._mlx_engine_signature = signature
            self._touch()
            return self._mlx_engine

        signature = ("F5TTS_v1_Base", ckpt_file or "", bool(use_ema))
        if self._engine is None or self._engine_signature != signature:
            with self._lock:
                if self._engine is None or self._engine_signature != signature:
                    from f5_tts.api import F5TTS

                    self._engine = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt_file or "", use_ema=use_ema)
                    self._engine_signature = signature
        self._touch()
        return self._engine

    def prepare_reference(self, ref_file: str, ref_text: str, transcription_model: str | None = None) -> dict[str, Any]:
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text

        ref_audio, prepared_text = preprocess_ref_audio_text(
            ref_file,
            ref_text,
            show_info=lambda *_args, **_kwargs: None,
            transcription_model=transcription_model,
        )
        cache_key = (ref_audio, prepared_text)
        if cache_key not in self._reference_cache:
            audio, sample_rate = torchaudio.load(ref_audio)
            self._reference_cache[cache_key] = {
                "ref_audio": ref_audio,
                "ref_text": prepared_text,
                "audio": audio,
                "sample_rate": sample_rate,
            }
            while len(self._reference_cache) > self._reference_cache_limit:
                self._reference_cache.popitem(last=False)
        else:
            self._reference_cache.move_to_end(cache_key)
        return self._reference_cache[cache_key]

    def warm_up(self, backend: str = "pytorch", ckpt_file: str | None = None, use_ema: bool = True) -> None:
        if backend == "mlx":
            prepared_reference = self.prepare_reference(
                str(Path(__file__).resolve().parents[1] / "infer" / "examples" / "basic" / "basic_ref_en.wav"),
                "Some call me nature, others call me mother nature.",
            )
            self._infer_with_mlx(
                prepared_reference,
                "Warm up the Apple MLX voice model.",
                nfe_step=4,
                speed=1.0,
                cfg_strength=2.0,
                sway_sampling_coef=-1.0,
                seed=0,
            )
        else:
            self.ensure_engine(backend=backend, ckpt_file=ckpt_file, use_ema=use_ema).warm_up(
                show_info=lambda *_args, **_kwargs: None
            )
        self._touch()

    def maybe_unload(self, idle_unload_seconds: int, force: bool = False) -> None:
        if self._engine is None and self._mlx_engine is None:
            return
        if not force and idle_unload_seconds > 0 and (time.time() - self._last_used_at) < idle_unload_seconds:
            return
        with self._lock:
            self._engine = None
            self._engine_signature = None
            self._mlx_engine = None
            self._mlx_engine_signature = None
        release_memory()

    def _load_silero(self):
        if self._silero_model is not None:
            return self._silero_model
        try:
            from silero_vad import load_silero_vad
        except ImportError:
            return None
        self._silero_model = load_silero_vad()
        return self._silero_model

    def _speech_metrics(
        self, audio_path: str, mono_audio: torch.Tensor, sample_rate: int
    ) -> tuple[float | None, float | None]:
        model = self._load_silero()
        if model is None:
            return None, None

        try:
            from silero_vad import get_speech_timestamps, read_audio
        except ImportError:
            return None, None

        try:
            vad_audio = read_audio(audio_path)
            timestamps = get_speech_timestamps(vad_audio, model, return_seconds=True)
        except Exception:
            return None, None

        speech_seconds = sum(segment["end"] - segment["start"] for segment in timestamps)
        duration = mono_audio.shape[-1] / sample_rate
        speech_ratio = 0.0 if duration == 0 else min(max(speech_seconds / duration, 0.0), 1.0)
        return speech_seconds, speech_ratio

    def transcribe_audio(
        self,
        audio_path: str,
        language: str | None = None,
        backend: str = DEFAULT_ASR_BACKEND,
        model_name: str | None = None,
        keep_loaded: bool = False,
    ) -> tuple[str, str]:
        chosen_backend = backend
        if backend == "auto":
            chosen_backend = "mlx_whisper" if self.device.startswith("mps") else "transformers"

        if chosen_backend == "mlx_whisper":
            try:
                import mlx_whisper

                result = mlx_whisper.transcribe(
                    audio_path,
                    path_or_hf_repo=model_name or "mlx-community/whisper-small",
                    language=language,
                    word_timestamps=False,
                )
                self._touch()
                if not keep_loaded:
                    release_memory()
                return result["text"].strip(), "mlx_whisper"
            except Exception:
                chosen_backend = "transformers"

        from f5_tts.infer.utils_infer import transcribe as infer_transcribe
        from f5_tts.infer.utils_infer import unload_asr_pipeline

        transformers_model = model_name
        if not transformers_model or transformers_model.startswith("mlx-community/"):
            if transformers_model and "small" in transformers_model:
                transformers_model = "openai/whisper-small"
            elif transformers_model and "medium" in transformers_model:
                transformers_model = "openai/whisper-medium"
            else:
                transformers_model = "openai/whisper-large-v3-turbo"

        try:
            transcript = infer_transcribe(audio_path, language=language, model_name=transformers_model)
            self._touch()
            return transcript, "transformers"
        finally:
            if not keep_loaded:
                unload_asr_pipeline()
                release_memory()

    def analyze_reference(
        self,
        audio_path: str,
        transcript: str = "",
        backend: str = DEFAULT_ASR_BACKEND,
        model_name: str | None = None,
        keep_loaded: bool = False,
    ) -> ReferenceAnalysis:
        audio, sample_rate = torchaudio.load(audio_path)
        mono_audio = audio.mean(dim=0).float()
        duration_seconds = mono_audio.shape[-1] / sample_rate
        rms = float(torch.sqrt(torch.mean(mono_audio.square())).item()) if mono_audio.numel() else 0.0
        peak = float(mono_audio.abs().max().item()) if mono_audio.numel() else 0.0

        window = max(sample_rate // 20, 1)
        unfolded = (
            mono_audio.abs().unfold(0, window, window)
            if mono_audio.shape[-1] >= window
            else mono_audio.abs().unsqueeze(0)
        )
        energies = unfolded.mean(dim=-1)
        silence_threshold = max(rms * 0.35, 0.01)
        trailing_silence_seconds = 0.0
        for value in reversed(energies.tolist()):
            if value > silence_threshold:
                break
            trailing_silence_seconds += window / sample_rate

        speech_seconds, speech_ratio = self._speech_metrics(audio_path, mono_audio, sample_rate)
        transcript_text = transcript.strip()
        used_backend = "manual"
        if not transcript_text:
            transcript_text, used_backend = self.transcribe_audio(
                audio_path,
                backend=backend,
                model_name=model_name,
                keep_loaded=keep_loaded,
            )

        warnings: list[str] = []
        notes: list[str] = [
            f"Sample length: {format_duration(duration_seconds)}",
            f"Trailing silence: {format_duration(trailing_silence_seconds)}",
        ]
        if duration_seconds < 4:
            warnings.append("Reference is shorter than 4 seconds. Cloning may sound unstable.")
        elif duration_seconds < 6:
            warnings.append("Reference is usable, but 6 to 12 seconds usually gives more reliable identity.")
        if duration_seconds > 14:
            warnings.append(
                "Reference is longer than 14 seconds. Trim closer to 12 seconds for faster and safer inference."
            )
        if peak > 0.98:
            warnings.append("Reference may be clipping. A cleaner take should improve naturalness.")
        if rms < 0.015:
            warnings.append("Reference level is quite low. Louder, clearer speech should transcribe more reliably.")
        if trailing_silence_seconds < 0.25:
            warnings.append("Add a little clean silence at the end of the sample to reduce truncation risk.")
        if speech_ratio is not None and speech_ratio < 0.55:
            warnings.append("The clip contains limited detected speech relative to its duration.")

        quality = reference_quality_breakdown(
            {
                "duration_seconds": duration_seconds,
                "rms": rms,
                "peak": peak,
                "trailing_silence_seconds": trailing_silence_seconds,
                "speech_ratio": speech_ratio,
                "warnings": warnings,
            }
        )
        notes.extend(quality["strengths"])
        if quality["issues"] and not warnings:
            warnings.extend(quality["issues"][:2])

        return ReferenceAnalysis(
            transcript=transcript_text,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channels=audio.shape[0],
            rms=rms,
            peak=peak,
            trailing_silence_seconds=trailing_silence_seconds,
            speech_seconds=speech_seconds,
            speech_ratio=speech_ratio,
            backend=used_backend,
            quality_score=quality["score"],
            quality_rating=quality["rating"],
            warnings=warnings,
            notes=notes,
        )

    def analyze_style(
        self,
        style_audio_path: str,
        style_text: str,
        context_notes: str,
        gen_text: str = "",
        backend: str = DEFAULT_ASR_BACKEND,
        model_name: str | None = None,
        keep_loaded: bool = False,
    ) -> StyleAnalysis:
        transcript = style_text.strip()
        if not transcript:
            transcript, _ = self.transcribe_audio(
                style_audio_path,
                backend=backend,
                model_name=model_name,
                keep_loaded=keep_loaded,
            )

        audio, sample_rate = torchaudio.load(style_audio_path)
        duration_seconds = audio.shape[-1] / sample_rate
        style_text_len = max(text_timing_units(transcript), 1)
        style_rate = style_text_len / max(duration_seconds, 0.1)
        recommended_speed = min(max(style_rate / 2.7, 0.72), 1.22)
        suggested_fix_duration = None
        if gen_text.strip():
            target_text_len = max(text_timing_units(gen_text), 1)
            suggested_fix_duration = duration_seconds * (target_text_len / style_text_len)

        recommended_speed, suggested_fix_duration, matched = apply_context_modifiers(
            recommended_speed,
            suggested_fix_duration,
            context_notes,
        )
        notes = [
            f"Style prompt length: {format_duration(duration_seconds)}",
            f"Recommended speed: {recommended_speed:.2f}",
        ]
        if suggested_fix_duration is not None:
            notes.append(f"Suggested generated speech target: {format_duration(suggested_fix_duration)}")
        if matched:
            notes.append(f"Context matched: {', '.join(matched)}")

        return StyleAnalysis(
            transcript=transcript,
            duration_seconds=duration_seconds,
            recommended_speed=recommended_speed,
            suggested_fix_duration=suggested_fix_duration,
            matched_keywords=matched,
            notes=notes,
        )

    def default_normalized_rtf(self, backend: str = "pytorch") -> float:
        if backend == "mlx":
            return 0.72 if self.mlx_available() else 1.15
        device = self.device
        if device.startswith("mps"):
            return 1.15
        if device.startswith("cuda"):
            return 0.22
        if device.startswith("xpu"):
            return 0.75
        return 2.4

    def normalized_rtf(self, backend: str = "pytorch") -> float:
        samples = self._timing_samples.get(backend, [])
        if samples:
            return sum(samples) / len(samples)
        return self.default_normalized_rtf(backend=backend)

    def _record_runtime(
        self, elapsed_seconds: float, runtime_seconds: float, nfe_step: int, backend: str = "pytorch"
    ) -> None:
        if runtime_seconds <= 0:
            return
        normalized = (elapsed_seconds / runtime_seconds) * (32 / max(nfe_step, 1))
        samples = self._timing_samples.setdefault(backend, [])
        samples.append(normalized)
        self._timing_samples[backend] = samples[-6:]

    def estimate_generation(
        self,
        reference_audio_path: str,
        ref_text: str,
        text: str,
        speed: float,
        nfe_step: int,
        engine_loaded: bool,
        queue_depth: int,
        backend: str = "pytorch",
    ) -> GenerationEstimate:
        prepared = self.prepare_reference(
            reference_audio_path,
            ref_text,
        )
        audio = prepared["audio"]
        sample_rate = prepared["sample_rate"]
        duration_seconds = audio.shape[-1] / sample_rate
        ref_audio_len = int(duration_seconds * target_sample_rate / hop_length)
        ref_text_len = max(text_timing_units(prepared["ref_text"]), 1)
        max_chars = int(ref_text_len / max(duration_seconds, 0.1) * (22 - duration_seconds) * speed)
        max_chars = max(max_chars, 60)
        text_batches = chunk_text(text, max_chars=max_chars)

        predicted_output_seconds = 0.0
        for chunk in text_batches:
            chunk_units = max(text_timing_units(chunk), 1)
            local_speed = speed if chunk_units >= 4 else min(speed, 0.45)
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * chunk_units / local_speed)
            frames = max(duration - ref_audio_len, 0)
            predicted_output_seconds += frames * hop_length / target_sample_rate

        compute_seconds = predicted_output_seconds * self.normalized_rtf(backend=backend) * (nfe_step / 32)
        cold_start_penalty = 0.0 if engine_loaded else 2.5
        total_seconds = compute_seconds + cold_start_penalty + max(len(text_batches) - 1, 0) * 0.12 + queue_depth * 0.4
        samples = self._timing_samples.get(backend, [])
        confidence = "calibrated from recent local runs" if samples else "first-run estimate"
        return GenerationEstimate(
            estimated_generation_seconds=total_seconds,
            predicted_output_seconds=predicted_output_seconds,
            chunks=len(text_batches),
            effective_speed=speed,
            queue_depth=queue_depth,
            confidence=confidence,
        )

    def _cross_fade_waves(self, generated_waves: list[np.ndarray], cross_fade_duration: float) -> np.ndarray:
        if not generated_waves:
            return np.array([], dtype=np.float32)
        if len(generated_waves) == 1 or cross_fade_duration <= 0:
            return np.concatenate(generated_waves)

        final_wave = generated_waves[0]
        for next_wave in generated_waves[1:]:
            prev_wave = final_wave
            cross_fade_samples = int(cross_fade_duration * target_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))
            if cross_fade_samples <= 0:
                final_wave = np.concatenate([prev_wave, next_wave])
                continue

            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]
            fade_out = np.linspace(1, 0, cross_fade_samples, dtype=np.float32)
            fade_in = np.linspace(0, 1, cross_fade_samples, dtype=np.float32)
            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in
            final_wave = np.concatenate(
                [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
            )
        return final_wave.astype(np.float32, copy=False)

    def _infer_with_mlx(
        self,
        prepared_reference: dict[str, Any],
        gen_text: str,
        *,
        nfe_step: int,
        speed: float,
        cfg_strength: float,
        sway_sampling_coef: float,
        seed: int | None,
        cross_fade_duration: float = 0.15,
    ) -> tuple[np.ndarray, int, None]:
        import mlx.core as mx
        from f5_tts_mlx.utils import convert_char_to_pinyin

        engine = self.ensure_engine(backend="mlx")

        audio_tensor = prepared_reference["audio"].mean(dim=0).float()
        sample_rate = int(prepared_reference["sample_rate"])
        if sample_rate != target_sample_rate:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor.unsqueeze(0), sample_rate, target_sample_rate
            ).squeeze(0)
            sample_rate = target_sample_rate

        ref_audio = audio_tensor.cpu().numpy().astype(np.float32, copy=False)
        ref_text = prepared_reference["ref_text"].strip()
        ref_audio_mx = mx.array(ref_audio)
        rms = mx.sqrt(mx.mean(mx.square(ref_audio_mx)))
        rms_value = float(rms.item())
        if 0.0 < rms_value < 0.1:
            ref_audio_mx = ref_audio_mx * (0.1 / rms)

        batches = chunk_text(gen_text, max_chars=220) or [gen_text]
        generated_waves: list[np.ndarray] = []
        ref_audio_len = ref_audio_mx.shape[0] // hop_length
        zh_pause_punc = r"。，、；：？！"
        ref_text_len = len(ref_text.encode("utf-8")) + 3 * len(re.findall(zh_pause_punc, ref_text))
        ref_text_len = max(ref_text_len, 1)
        for chunk in batches:
            text = convert_char_to_pinyin([f"{ref_text} {chunk}".strip()])
            chunk_text_len = len(chunk.encode("utf-8")) + 3 * len(re.findall(zh_pause_punc, chunk))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * max(chunk_text_len, 1) / max(speed, 0.1))
            wave, _trajectory = engine.sample(
                mx.expand_dims(ref_audio_mx, axis=0),
                text=text,
                duration=duration,
                steps=max(int(nfe_step), 1),
                method="rk4",
                cfg_strength=cfg_strength,
                speed=speed,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
            )
            wave = wave[ref_audio_mx.shape[0] :]
            mx.eval(wave)
            generated_waves.append(np.array(wave, dtype=np.float32))

        return self._cross_fade_waves(generated_waves, cross_fade_duration), sample_rate, None

    def _resolve_render_targets(
        self,
        request: GenerationRequest,
        reference: dict,
        style: dict | None,
        pronunciation_rules: list[dict],
        profile_name: str,
    ):
        profile = get_runtime_profile(profile_name)
        if request.mode == "preview":
            preview_nfe = profile.preview_nfe_step
        else:
            preview_nfe = profile.final_nfe_step

        normalized_text = normalize_script(request.text, pronunciation_rules)
        speed = request.speed if request.speed is not None else 1.0
        fix_duration = None
        style_notes: list[str] = []

        if style and request.use_style_prompt:
            analysis = style["analysis"]
            style_duration = float(analysis.get("duration_seconds", 0.0) or 0.0)
            style_text = style["transcript"] or analysis.get("transcript", "")
            if request.speed is None:
                speed = float(analysis.get("recommended_speed", speed))
            suggested_fix_duration = None
            if style_duration > 0 and style_text:
                target_text_len = max(text_timing_units(normalized_text), 1)
                style_text_len = max(text_timing_units(style_text), 1)
                ref_audio, ref_sr = torchaudio.load(reference["audio_path"])
                ref_duration = ref_audio.shape[-1] / ref_sr
                suggested_fix_duration = ref_duration + style_duration * (target_text_len / style_text_len)
            speed, fix_duration, matched = apply_context_modifiers(speed, suggested_fix_duration, request.context_notes)
            if matched:
                style_notes.append(f"Applied style context: {', '.join(matched)}")

        if request.mode == "preview":
            normalized_text = chunk_text(
                normalized_text,
                max_chars=profile.preview_char_limit,
            )[0]

        nfe_step = request.nfe_step
        if nfe_step is None:
            nfe_step = preview_nfe
        remove_silence = profile.trim_silence_default if request.remove_silence is None else request.remove_silence
        render_spectrogram = (
            profile.render_spectrogram_default if request.render_spectrogram is None else request.render_spectrogram
        )

        return normalized_text, speed, fix_duration, nfe_step, remove_silence, render_spectrogram, style_notes

    def render(
        self,
        request: GenerationRequest,
        reference: dict,
        style: dict | None,
        pronunciation_rules: list[dict],
        output_dir: Path,
        backend: str = "pytorch",
        ckpt_file: str | None = None,
        use_ema: bool = True,
    ) -> dict:
        prepared = self.prepare_reference(
            reference["audio_path"],
            reference["transcript"],
        )
        profile_name = self.store.get_setting("runtime_profile", DEFAULT_PROFILE) or DEFAULT_PROFILE
        render_text, speed, fix_duration, nfe_step, remove_silence, render_spectrogram, style_notes = (
            self._resolve_render_targets(
                request,
                reference,
                style,
                pronunciation_rules,
                profile_name,
            )
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        filename_stem = re.sub(r"[^a-z0-9]+", "-", request.name.lower()).strip("-") or "render"
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        wav_path = output_dir / f"{timestamp}-{filename_stem}-{request.mode}.wav"
        spec_path = output_dir / f"{timestamp}-{filename_stem}-{request.mode}.png"

        started_at = time.perf_counter()
        if backend == "mlx":
            wav, sample_rate, spec = self._infer_with_mlx(
                prepared,
                render_text,
                nfe_step=nfe_step,
                speed=speed,
                cfg_strength=request.cfg_strength,
                sway_sampling_coef=request.sway_sampling_coef,
                seed=request.seed,
                cross_fade_duration=request.cross_fade_duration,
            )
            resolved_seed = request.seed
        else:
            engine = self.ensure_engine(backend=backend, ckpt_file=ckpt_file, use_ema=use_ema)
            wav, sample_rate, spec = engine.infer_prepared(
                prepared,
                gen_text=render_text,
                speed=speed,
                nfe_step=nfe_step,
                cross_fade_duration=request.cross_fade_duration,
                cfg_strength=request.cfg_strength,
                sway_sampling_coef=request.sway_sampling_coef,
                progress=None,
                remove_silence=False,
                fix_duration=fix_duration,
                seed=request.seed,
            )
            resolved_seed = getattr(engine, "seed", request.seed)
        elapsed_seconds = time.perf_counter() - started_at

        sf.write(wav_path, wav, sample_rate)
        if remove_silence:
            from f5_tts.infer.utils_infer import remove_silence_for_generated_wav

            remove_silence_for_generated_wav(str(wav_path))
            trimmed, _ = torchaudio.load(str(wav_path))
            wav = trimmed.squeeze().cpu().numpy()
        if render_spectrogram and spec is not None:
            from f5_tts.infer.utils_infer import save_spectrogram

            save_spectrogram(spec, str(spec_path))
        else:
            spec_path = None

        runtime_seconds = len(wav) / sample_rate if len(wav) else 0.0
        self._record_runtime(elapsed_seconds, runtime_seconds, nfe_step, backend=backend)
        self._touch()
        return {
            "audio_path": str(wav_path),
            "spectrogram_path": str(spec_path) if spec_path is not None else None,
            "duration_seconds": runtime_seconds,
            "elapsed_seconds": elapsed_seconds,
            "sample_rate": sample_rate,
            "text_excerpt": render_text[:240],
            "effective_speed": speed,
            "nfe_step": nfe_step,
            "seed": resolved_seed,
            "inference_backend": backend,
            "inference_backend_label": "Apple MLX F5-TTS" if backend == "mlx" else "PyTorch F5-TTS",
            "style_notes": style_notes,
        }


def _render_job_process(
    paths: StudioPaths,
    request_payload: dict[str, Any],
    reference: dict[str, Any],
    style: dict[str, Any] | None,
    pronunciation_rules: list[dict[str, Any]],
    output_dir: str,
    engine_options: dict[str, Any],
    result_queue: mp.Queue,
) -> None:
    try:
        os.environ.setdefault("F5_TTS_PREPARED_REF_DIR", str(paths.cache / "prepared-references"))
        store = StudioStore(paths)
        engine = StudioEngine(store)
        request = GenerationRequest.model_validate(request_payload)
        result = engine.render(
            request,
            reference,
            style,
            pronunciation_rules,
            output_dir=Path(output_dir),
            **engine_options,
        )
        result_queue.put({"ok": True, "result": result})
    except Exception as exc:  # pragma: no cover - exercised through service integration
        result_queue.put({"ok": False, "error": str(exc)})


class StudioService:
    def __init__(self, paths: StudioPaths | None = None):
        self.paths = paths or get_studio_paths()
        os.environ.setdefault("F5_TTS_PREPARED_REF_DIR", str(self.paths.cache / "prepared-references"))
        self.store = StudioStore(self.paths)
        self.store.ensure_default_project()
        self.engine = StudioEngine(self.store)
        self._job_queue: queue.Queue[int] = queue.Queue()
        self._cancelled_jobs: set[int] = set()
        self._current_job_id: int | None = None
        self._worker_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._warm_thread: threading.Thread | None = None
        self._last_warm_error: str | None = None
        self._process_context = mp.get_context("spawn")
        self._active_process: mp.Process | None = None
        self._active_process_lock = threading.Lock()
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self._apply_profile_environment(self.get_runtime_profile_name())

    def _runtime_profile(self):
        return get_runtime_profile(self.get_runtime_profile_name())

    def _apply_profile_environment(self, profile_name: str) -> None:
        profile = get_runtime_profile(profile_name)
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(profile.mps_high_watermark_ratio)
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(profile.mps_low_watermark_ratio)

    def _search_checkpoint_roots(self) -> list[Path]:
        candidates: list[Path] = []
        seen: set[str] = set()

        def register(root: Path) -> None:
            try:
                resolved = root.resolve()
            except FileNotFoundError:
                resolved = root
            key = str(resolved)
            if key not in seen and (resolved / "ckpts").exists():
                seen.add(key)
                candidates.append(resolved)

        env_root = os.environ.get("F5_TTS_REPO_ROOT")
        if env_root:
            register(Path(env_root))

        for origin in (Path.cwd(), Path(__file__).resolve().parent):
            for parent in [origin, *origin.parents]:
                register(parent)

        return candidates

    def detect_latest_checkpoint(self) -> str:
        candidates = self.list_checkpoint_candidates(limit=1)
        if not candidates:
            return ""
        return candidates[0]

    def list_checkpoint_candidates(self, limit: int = 16) -> list[str]:
        matches: list[Path] = []
        patterns = (
            "*/snapshot_*.safetensors",
            "*/snapshot_*.pt",
            "*/model_last.safetensors",
            "*/model_last.pt",
            "*/model_*.safetensors",
            "*/model_*.pt",
        )
        for root in self._search_checkpoint_roots():
            ckpt_root = root / "ckpts"
            for pattern in patterns:
                matches.extend(ckpt_root.glob(pattern))

        filtered = [path.resolve() for path in matches if path.is_file() and not path.name.startswith("pretrained_")]
        filtered.sort(key=lambda path: path.stat().st_mtime, reverse=True)

        unique: list[str] = []
        seen: set[str] = set()
        for path in filtered:
            normalized = str(path)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(normalized)
            if len(unique) >= limit:
                break
        return unique

    def get_checkpoint_path(self) -> str:
        saved = (self.store.get_setting("checkpoint_path", "") or "").strip()
        if not saved:
            return ""
        return saved if Path(saved).exists() else ""

    def set_checkpoint_path(self, checkpoint_path: str) -> str:
        normalized = checkpoint_path.strip()
        if normalized:
            path = Path(normalized).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            normalized = str(path.resolve())
        self.store.set_setting("checkpoint_path", normalized)
        self.engine.maybe_unload(0, force=True)
        return normalized

    def get_use_ema(self) -> bool:
        return (self.store.get_setting("use_ema", "true") or "true").lower() != "false"

    def set_use_ema(self, value: bool) -> None:
        self.store.set_setting("use_ema", "true" if value else "false")
        self.engine.maybe_unload(0, force=True)

    def get_inference_backend(self) -> str:
        saved = (
            self.store.get_setting("inference_backend", DEFAULT_INFERENCE_BACKEND) or DEFAULT_INFERENCE_BACKEND
        ).strip()
        if saved == "mlx" and not self.engine.mlx_available():
            return "pytorch"
        return saved if saved in {"pytorch", "mlx"} else "pytorch"

    def set_inference_backend(self, backend: str) -> str:
        normalized = "mlx" if backend == "mlx" and self.engine.mlx_available() else "pytorch"
        self.store.set_setting("inference_backend", normalized)
        self.engine.maybe_unload(0, force=True)
        return normalized

    def available_inference_backends(self) -> list[tuple[str, str]]:
        choices = [("PyTorch F5-TTS", "pytorch")]
        if self.engine.mlx_available():
            choices.insert(0, ("Apple MLX F5-TTS", "mlx"))
        return choices

    def _backend_label(self, backend: str) -> str:
        return "Apple MLX F5-TTS" if backend == "mlx" else "PyTorch F5-TTS"

    def _engine_options(self) -> dict[str, Any]:
        return self._engine_options_for_request()

    @staticmethod
    def _render_engine_options_from(options: dict[str, Any]) -> dict[str, Any]:
        return {
            "backend": options["backend"],
            "ckpt_file": options["ckpt_file"],
            "use_ema": options["use_ema"],
        }

    def _engine_options_for_request(self, request: GenerationRequest | None = None) -> dict[str, Any]:
        if request is None or request.checkpoint_path is None:
            checkpoint_path = self.get_checkpoint_path() or None
        else:
            normalized = request.checkpoint_path.strip()
            if normalized:
                path = Path(normalized).expanduser()
                if not path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {path}")
                checkpoint_path = str(path.resolve())
            else:
                checkpoint_path = None

        if request is None or request.use_ema is None:
            use_ema = self.get_use_ema()
        else:
            use_ema = bool(request.use_ema)

        requested_backend = self.get_inference_backend()
        if request is not None and request.inference_backend is not None:
            requested_backend = request.inference_backend

        backend = requested_backend
        backend_reason = None
        if backend == "mlx" and checkpoint_path:
            backend = "pytorch"
            backend_reason = (
                "Apple MLX currently supports only the shipped base model, so checkpoint renders stay on PyTorch."
            )
        elif backend == "mlx" and not self.engine.mlx_available():
            backend = "pytorch"
            backend_reason = "Apple MLX is not available in this environment, so renders stay on PyTorch."

        return {
            "backend": backend,
            "requested_backend": requested_backend,
            "backend_reason": backend_reason,
            "ckpt_file": checkpoint_path,
            "use_ema": use_ema,
        }

    def get_runtime_profile_name(self) -> str:
        return self.store.get_setting("runtime_profile", DEFAULT_PROFILE) or DEFAULT_PROFILE

    def set_runtime_profile(self, value: str) -> None:
        profile = value if value in PROFILE_MAP else DEFAULT_PROFILE
        self.store.set_setting("runtime_profile", profile)
        self._apply_profile_environment(profile)
        if not get_runtime_profile(profile).warm_on_start:
            self.engine.maybe_unload(0, force=True)

    def get_asr_backend(self) -> str:
        return self.store.get_setting("asr_backend", DEFAULT_ASR_BACKEND) or DEFAULT_ASR_BACKEND

    def set_asr_backend(self, backend: str) -> None:
        self.store.set_setting("asr_backend", backend)

    def get_asr_model_name(self) -> str:
        return self._runtime_profile().preferred_asr_model

    def _effective_asr_backend(self) -> str:
        saved = self.get_asr_backend()
        if saved == "auto":
            return self._runtime_profile().preferred_asr_backend
        return saved

    def get_idle_unload_seconds(self) -> int:
        saved = self.store.get_setting("idle_unload_seconds")
        if saved is not None:
            return int(saved)
        return get_runtime_profile(self.get_runtime_profile_name()).idle_unload_seconds

    def set_idle_unload_seconds(self, seconds: int) -> None:
        self.store.set_setting("idle_unload_seconds", str(max(seconds, 0)))

    def warm_profile_if_needed(self) -> str:
        profile = get_runtime_profile(self.get_runtime_profile_name())
        if not profile.warm_on_start:
            return "Warm-up skipped in the current runtime profile."
        options = self._engine_options()
        if self.engine.is_loaded(options["backend"]):
            return "Voice engine is already warm."
        if self._warm_thread and self._warm_thread.is_alive():
            return "Voice engine warm-up is already in progress."

        def worker():
            try:
                self._last_warm_error = None
                self.engine.warm_up(
                    backend=options["backend"],
                    ckpt_file=options["ckpt_file"],
                    use_ema=options["use_ema"],
                )
            except Exception as exc:
                self._last_warm_error = str(exc)

        self._warm_thread = threading.Thread(target=worker, daemon=True)
        self._warm_thread.start()
        note = f"Voice engine warm-up started in the background using {self._backend_label(options['backend'])}."
        if options.get("backend_reason"):
            note += f" {options['backend_reason']}"
        return note

    def warm_engine_now(self) -> str:
        try:
            self._last_warm_error = None
            options = self._engine_options()
            self.engine.warm_up(
                backend=options["backend"],
                ckpt_file=options["ckpt_file"],
                use_ema=options["use_ema"],
            )
            note = f"Voice engine is warm and ready using {self._backend_label(options['backend'])}."
            if options.get("backend_reason"):
                note += f" {options['backend_reason']}"
            return note
        except Exception as exc:
            self._last_warm_error = str(exc)
            raise

    def maybe_unload_idle_engine(self) -> None:
        self.engine.maybe_unload(self.get_idle_unload_seconds())

    def list_projects(self) -> list[dict]:
        return self.store.list_projects()

    def create_project(self, name: str, description: str = "") -> dict:
        return self.store.create_project(name, description)

    def get_project_detail(self, project_id: int) -> dict:
        self.store.sync_default_voice_profile(project_id)
        return self.store.get_project_detail(project_id)

    def list_references(self, project_id: int) -> list[dict]:
        return self.store.list_voice_assets(project_id, "reference")

    def list_styles(self, project_id: int) -> list[dict]:
        return self.store.list_voice_assets(project_id, "style")

    def list_voice_profiles(self, project_id: int) -> list[dict]:
        self.store.sync_default_voice_profile(project_id)
        return self.store.list_voice_profiles(project_id)

    def get_voice_profile(self, profile_id: int) -> dict:
        return self.store.get_voice_profile(profile_id)

    def save_voice_profile(
        self,
        project_id: int,
        name: str,
        reference_ids: list[int],
        *,
        description: str = "",
        profile_id: int | None = None,
        is_default: bool = False,
    ) -> dict:
        profile = self.store.save_voice_profile(
            project_id,
            name,
            reference_ids,
            description=description,
            profile_id=profile_id,
            is_default=is_default,
        )
        self.maybe_unload_idle_engine()
        return profile

    def list_assets(self, project_id: int) -> list[dict]:
        return self.store.list_audio_assets(project_id)

    def list_sources(self, project_id: int) -> list[dict]:
        return self.store.list_audio_assets(project_id, kind="source")

    def list_jobs(self, project_id: int | None = None) -> list[dict]:
        return self.store.list_jobs(project_id)

    def recommend_references(self, project_id: int) -> list[dict[str, Any]]:
        recommendations: list[dict[str, Any]] = []
        for item in self.list_references(project_id):
            analysis = dict(item.get("analysis") or {})
            quality = reference_quality_breakdown(analysis)
            recommendations.append(
                {
                    "id": item["id"],
                    "name": item["name"],
                    "score": quality["score"],
                    "rating": quality["rating"],
                    "summary": " / ".join((quality["strengths"] or quality["issues"] or ["No notes available."])[:2]),
                }
            )
        recommendations.sort(key=lambda row: (-row["score"], row["name"].lower(), row["id"]))
        return recommendations

    def _desired_reference_rate(self, style: dict | None, context_notes: str) -> float:
        desired_rate = 3.2
        if style:
            desired_rate *= float(style.get("analysis", {}).get("recommended_speed", 1.0) or 1.0)
        lowered = context_notes.lower()
        for keyword, modifier in CONTEXT_MODIFIERS.items():
            if keyword in lowered:
                desired_rate *= float(modifier["speed"])
        return min(max(desired_rate, 1.8), 5.4)

    def _score_profile_reference(
        self,
        reference: dict,
        *,
        style: dict | None,
        context_notes: str,
        text: str,
        mode: str = "final",
    ) -> dict[str, Any]:
        analysis = dict(reference.get("analysis") or {})
        quality = reference_quality_breakdown(analysis)
        duration = float(analysis.get("duration_seconds", 0.0) or 0.0)
        speech_seconds = float(analysis.get("speech_seconds", 0.0) or 0.0)
        denominator = speech_seconds if speech_seconds > 0 else max(duration, 0.1)
        reference_rate = text_timing_units(reference.get("transcript", "")) / max(denominator, 0.1)
        desired_rate = self._desired_reference_rate(style, context_notes)
        rate_alignment = max(0.0, 1.0 - (abs(reference_rate - desired_rate) / max(desired_rate, 0.1)))
        duration_bonus = 6.0 if 6.0 <= duration <= 10.5 else 3.0 if 4.5 <= duration <= 12.0 else 0.0
        speech_ratio = float(analysis.get("speech_ratio", 0.0) or 0.0)
        speech_bonus = max(min((speech_ratio - 0.72) * 18.0, 5.0), 0.0)

        keyword_bonus = 0.0
        keyword_hits: list[str] = []
        searchable = " ".join(
            [
                str(reference.get("name", "")).lower(),
                str(reference.get("transcript", "")).lower(),
            ]
        )
        for keyword in sorted(set(word_tokens(context_notes))):
            if keyword in searchable and len(keyword) > 2:
                keyword_bonus += 2.5
                keyword_hits.append(keyword)
        if style:
            for keyword in style.get("analysis", {}).get("matched_keywords", []):
                lowered = str(keyword).lower()
                if lowered and lowered in searchable and lowered not in keyword_hits:
                    keyword_bonus += 2.0
                    keyword_hits.append(lowered)

        target_tokens = set(word_tokens(text))
        reference_tokens = set(word_tokens(reference.get("transcript", "")))
        lexical_overlap = len(target_tokens & reference_tokens) * 0.3 if target_tokens and reference_tokens else 0.0

        if mode == "preview":
            latency_penalty = max(duration - 7.5, 0.0) * 2.4
            shorter_bonus = max(7.5 - duration, 0.0) * 0.8
        else:
            latency_penalty = max(duration - 9.0, 0.0) * 0.7
            shorter_bonus = max(8.0 - duration, 0.0) * 0.2

        final_score = (
            float(quality["score"])
            + (rate_alignment * 10.0)
            + duration_bonus
            + speech_bonus
            + keyword_bonus
            + lexical_overlap
            + shorter_bonus
            - latency_penalty
        )
        reasons = [
            f"quality {quality['score']:.1f}/{quality['rating']}",
            f"pace {reference_rate:.2f}u/s vs target {desired_rate:.2f}u/s",
        ]
        if duration_bonus:
            reasons.append(f"duration {duration:.1f}s in the sweet spot")
        if latency_penalty > 0:
            reasons.append(f"latency penalty for {duration:.1f}s reference")
        if keyword_hits:
            reasons.append("matched cues: " + ", ".join(keyword_hits[:3]))

        return {
            "id": int(reference["id"]),
            "name": reference["name"],
            "score": round(final_score, 2),
            "quality_score": round(float(quality["score"]), 2),
            "quality_rating": quality["rating"],
            "reference_rate": round(reference_rate, 3),
            "desired_rate": round(desired_rate, 3),
            "duration_seconds": duration,
            "keyword_hits": keyword_hits,
            "summary": " · ".join(reasons[:3]),
        }

    def recommend_profile_references(
        self,
        profile_id: int,
        *,
        style_id: int | None = None,
        context_notes: str = "",
        text: str = "",
        mode: str = "final",
    ) -> list[dict[str, Any]]:
        profile = self.store.get_voice_profile(profile_id)
        style = self.store.get_voice_asset(style_id) if style_id else None
        candidates = [
            self._score_profile_reference(
                member,
                style=style,
                context_notes=context_notes,
                text=text,
                mode=mode,
            )
            for member in profile.get("members", [])
        ]
        candidates.sort(key=lambda row: (-float(row["score"]), str(row["name"]).lower(), int(row["id"])))
        return candidates

    def _resolve_generation_route(self, request: GenerationRequest) -> dict[str, Any]:
        style = self.store.get_voice_asset(request.style_id) if request.style_id else None
        if request.voice_profile_id:
            profile = self.store.get_voice_profile(int(request.voice_profile_id))
            if int(profile["project_id"]) != int(request.project_id):
                raise ValueError("Selected voice profile does not belong to the active project.")
            ranked = self.recommend_profile_references(
                int(profile["id"]),
                style_id=request.style_id,
                context_notes=request.context_notes,
                text=request.text,
                mode=request.mode,
            )
            if not ranked:
                raise ValueError("The selected voice profile does not contain any usable references.")
            reference = self.store.get_voice_asset(int(ranked[0]["id"]))
            return {
                "reference": reference,
                "style": style,
                "voice_profile": profile,
                "reference_selection": ranked[0],
                "reference_candidates": ranked[:3],
            }

        if request.reference_id is None:
            raise ValueError("Choose a reference voice or a voice profile before rendering.")
        reference = self.store.get_voice_asset(int(request.reference_id))
        return {
            "reference": reference,
            "style": style,
            "voice_profile": None,
            "reference_selection": {
                "id": int(reference["id"]),
                "name": reference["name"],
                "score": round(float(reference_quality_breakdown(reference.get("analysis") or {})["score"]), 2),
                "summary": "Explicit reference selected.",
            },
            "reference_candidates": [],
        }

    def diagnose_asset(
        self,
        project_id: int,
        asset_id: int,
        *,
        reference_id: int | None = None,
        expected_text: str = "",
    ) -> dict[str, Any]:
        asset = self.store.get_audio_asset(asset_id)
        if int(asset["project_id"]) != int(project_id):
            raise ValueError("The selected take does not belong to the active project.")

        metadata = dict(asset.get("metadata") or {})
        reference_audio_path = None
        resolved_reference_id = reference_id or metadata.get("reference_id")
        if resolved_reference_id:
            reference = self.store.get_voice_asset(int(resolved_reference_id))
            reference_audio_path = reference["audio_path"]

        expected = expected_text.strip()
        if not expected:
            expected = (
                metadata.get("requested_text") or metadata.get("edited_text") or metadata.get("text_excerpt") or ""
            )
        if not expected and asset.get("job_id"):
            try:
                job = self.store.get_job(int(asset["job_id"]))
                expected = str(job.get("recipe", {}).get("text", "")).strip()
            except Exception:
                expected = ""

        profile = self._runtime_profile()
        transcript, backend = self.engine.transcribe_audio(
            asset["path"],
            backend=self._effective_asr_backend(),
            model_name=self.get_asr_model_name(),
            keep_loaded=profile.keep_asr_loaded,
        )
        if not profile.keep_asr_loaded:
            release_memory()

        report = diagnose_render(
            generated_audio_path=asset["path"],
            expected_text=expected,
            observed_text=transcript,
            reference_audio_path=reference_audio_path,
        )
        report.update(
            {
                "asset_id": asset_id,
                "asset_label": asset["label"],
                "asset_kind": asset["kind"],
                "reference_id": int(resolved_reference_id) if resolved_reference_id else None,
                "transcript_backend": backend,
            }
        )
        return report

    def save_pronunciation_rule(self, project_id: int, source: str, replacement: str) -> dict:
        return self.store.upsert_pronunciation_rule(project_id, source, replacement)

    def ingest_reference(self, project_id: int, name: str, audio_path: str, transcript: str = "") -> tuple[dict, dict]:
        profile = self._runtime_profile()
        analysis = self.engine.analyze_reference(
            audio_path,
            transcript,
            backend=self._effective_asr_backend(),
            model_name=self.get_asr_model_name(),
            keep_loaded=profile.keep_asr_loaded,
        )
        saved = self.store.save_voice_asset(
            project_id=project_id,
            kind="reference",
            name=name,
            audio_path=audio_path,
            transcript=analysis.transcript,
            analysis=analysis.model_dump(),
        )
        self.store.sync_default_voice_profile(project_id)
        self.maybe_unload_idle_engine()
        return saved, analysis.model_dump()

    def ingest_style(
        self,
        project_id: int,
        name: str,
        audio_path: str,
        transcript: str = "",
        context_notes: str = "",
        gen_text: str = "",
    ) -> tuple[dict, dict]:
        profile = self._runtime_profile()
        analysis = self.engine.analyze_style(
            audio_path,
            transcript,
            context_notes,
            gen_text=gen_text,
            backend=self._effective_asr_backend(),
            model_name=self.get_asr_model_name(),
            keep_loaded=profile.keep_asr_loaded,
        )
        payload = analysis.model_dump()
        payload["context_notes"] = context_notes
        saved = self.store.save_voice_asset(
            project_id=project_id,
            kind="style",
            name=name,
            audio_path=audio_path,
            transcript=analysis.transcript,
            analysis=payload,
        )
        self.maybe_unload_idle_engine()
        return saved, payload

    def ingest_edit_source(
        self,
        project_id: int,
        name: str,
        audio_path: str,
        transcript: str = "",
    ) -> tuple[dict, dict]:
        from f5_tts.studio.editing import align_transcript

        profile = self._runtime_profile()
        alignment = align_transcript(
            audio_path,
            transcript,
            model_name=self.get_asr_model_name(),
        )
        audio, sample_rate = torchaudio.load(audio_path)
        metadata = {
            "transcript": alignment["transcript"],
            "alignment": alignment["words"],
            "alignment_backend": alignment["alignment_backend"],
            "warnings": alignment.get("warnings", []),
            "asr_transcript": alignment.get("asr_transcript", ""),
            "duration_seconds": audio.shape[-1] / sample_rate,
        }
        if not profile.keep_asr_loaded:
            release_memory()
        asset = self.store.save_source_asset(project_id, name, audio_path, metadata)
        return asset, metadata

    def render_edit_now(
        self,
        *,
        project_id: int,
        source_asset_id: int,
        take_name: str,
        target_text: str,
        replacement_text: str,
        occurrence: int = 1,
        action: str = "replace",
        preserve_timing: bool = True,
        nfe_step: int | None = None,
        render_spectrogram: bool = False,
    ) -> dict:
        from f5_tts.studio.editing import build_text_edit_plan, render_speech_edit

        source_asset = self.store.get_audio_asset(source_asset_id)
        if source_asset["project_id"] != project_id:
            raise ValueError("Selected source asset does not belong to the active project.")

        metadata = source_asset["metadata"]
        transcript = metadata.get("transcript", "")
        alignment = metadata.get("alignment") or []
        if not transcript or not alignment:
            raise ValueError("The selected source asset does not have alignment metadata yet.")

        plan = build_text_edit_plan(
            transcript,
            alignment,
            target_text,
            replacement_text,
            occurrence=occurrence,
            action=action,
            preserve_timing=preserve_timing,
        )
        project = self.store.get_project_summary(project_id)
        output_dir = self._job_output_dir(project)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename_stem = re.sub(r"[^a-z0-9]+", "-", take_name.lower()).strip("-") or "edit"
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        wav_path = output_dir / f"{timestamp}-{filename_stem}-edit.wav"
        spec_path = output_dir / f"{timestamp}-{filename_stem}-edit.png"
        profile = self._runtime_profile()
        effective_nfe = nfe_step or profile.final_nfe_step

        result = render_speech_edit(
            source_asset["path"],
            original_text=plan["original_text"],
            edited_text=plan["edited_text"],
            spans=plan["spans"],
            ckpt_file=self._engine_options().get("ckpt_file") or "",
            use_ema=self._engine_options().get("use_ema", True),
            nfe_step=effective_nfe,
            output_wav_path=str(wav_path),
            output_spec_path=str(spec_path) if render_spectrogram else None,
            preserve_timing=preserve_timing,
        )
        result["plan"] = plan
        asset = self.store.save_audio_asset(
            project_id=project_id,
            job_id=None,
            kind="edit",
            label=take_name,
            path=result["audio_path"],
            duration_seconds=result["duration_seconds"],
            metadata={**result, "source_asset_id": source_asset_id, "action": action},
        )
        return {"asset_id": asset["id"], **result}

    def estimate(self, request: GenerationRequest) -> dict:
        resolved = self._resolve_generation_route(request)
        reference = resolved["reference"]
        style = resolved["style"]
        engine_options = self._engine_options_for_request(request)
        pronunciation_rules = self.store.list_pronunciation_rules(request.project_id)
        normalized_text = normalize_script(request.text, pronunciation_rules)
        speed = request.speed if request.speed is not None else 1.0
        if style and request.use_style_prompt and request.speed is None:
            speed = float(style["analysis"].get("recommended_speed", speed))
        speed, _, _ = apply_context_modifiers(speed, None, request.context_notes)
        nfe_step = request.nfe_step or (
            get_runtime_profile(self.get_runtime_profile_name()).preview_nfe_step
            if request.mode == "preview"
            else get_runtime_profile(self.get_runtime_profile_name()).final_nfe_step
        )
        estimate = self.engine.estimate_generation(
            reference_audio_path=reference["audio_path"],
            ref_text=reference["transcript"],
            text=normalized_text,
            speed=speed,
            nfe_step=nfe_step,
            engine_loaded=self.engine.is_loaded(engine_options["backend"]),
            queue_depth=self._job_queue.qsize(),
            backend=engine_options["backend"],
        )
        self.maybe_unload_idle_engine()
        return estimate.model_dump()

    def _job_output_dir(self, project: dict) -> Path:
        output_dir = self.store.project_dir(project["slug"]) / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _finalize_completed_job(
        self, job_id: int, request: GenerationRequest, result: dict, resolved: dict[str, Any]
    ) -> dict:
        reference = resolved["reference"]
        voice_profile = resolved.get("voice_profile")
        selection = dict(resolved.get("reference_selection") or {})
        result_with_provenance = {
            **result,
            "requested_text": request.text,
            "reference_id": int(reference["id"]),
            "requested_reference_id": request.reference_id,
            "voice_profile_id": int(voice_profile["id"]) if voice_profile else None,
            "voice_profile_name": voice_profile["name"] if voice_profile else None,
            "resolved_reference_name": reference["name"],
            "reference_selection": selection,
            "style_id": request.style_id,
            "mode": request.mode,
            "checkpoint_path": request.checkpoint_path,
            "use_ema": request.use_ema,
            "seed": result.get("seed", request.seed),
        }
        asset = self.store.save_audio_asset(
            project_id=request.project_id,
            job_id=job_id,
            kind=request.mode,
            label=request.name,
            path=result["audio_path"],
            duration_seconds=result["duration_seconds"],
            metadata=result_with_provenance,
        )
        final_result = {"asset_id": asset["id"], **result_with_provenance}
        self.store.update_job(job_id, status="completed", result=final_result)
        self.maybe_unload_idle_engine()
        return self.store.get_job(job_id)

    def _execute_job(self, job_id: int, request: GenerationRequest) -> dict:
        self.store.update_job(job_id, status="running")
        project = self.store.get_project_summary(request.project_id)
        resolved = self._resolve_generation_route(request)
        reference = resolved["reference"]
        style = resolved["style"]
        pronunciation_rules = self.store.list_pronunciation_rules(request.project_id)
        engine_options = self._engine_options_for_request(request)
        result = self.engine.render(
            request,
            reference,
            style,
            pronunciation_rules,
            output_dir=self._job_output_dir(project),
            **self._render_engine_options_from(engine_options),
        )
        return self._finalize_completed_job(job_id, request, result, resolved)

    def _execute_job_in_subprocess(self, job_id: int, request: GenerationRequest) -> dict:
        self.store.update_job(job_id, status="running")
        project = self.store.get_project_summary(request.project_id)
        resolved = self._resolve_generation_route(request)
        reference = resolved["reference"]
        style = resolved["style"]
        pronunciation_rules = self.store.list_pronunciation_rules(request.project_id)
        engine_options = self._engine_options_for_request(request)
        result_queue: mp.Queue = self._process_context.Queue()
        process = self._process_context.Process(
            target=_render_job_process,
            args=(
                self.paths,
                request.model_dump(),
                reference,
                style,
                pronunciation_rules,
                str(self._job_output_dir(project)),
                self._render_engine_options_from(engine_options),
                result_queue,
            ),
            daemon=True,
        )
        with self._active_process_lock:
            self._active_process = process
        process.start()

        try:
            while True:
                try:
                    payload = result_queue.get(timeout=0.5)
                    break
                except queue.Empty:
                    if job_id in self._cancelled_jobs:
                        raise JobCancelledError("Cancelled while rendering.")
                    if not process.is_alive() and result_queue.empty():
                        raise RuntimeError("Render worker exited before returning a result.")
        except JobCancelledError:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            self._cancelled_jobs.discard(job_id)
            return self.store.update_job(job_id, status="cancelled", error_text="Cancelled while rendering.")
        finally:
            if process.is_alive():
                process.join(timeout=0.1)
            result_queue.close()
            with self._active_process_lock:
                self._active_process = None

        process.join(timeout=5)
        if not payload.get("ok"):
            raise RuntimeError(payload.get("error", "Render worker failed."))
        return self._finalize_completed_job(job_id, request, payload["result"], resolved)

    def render_now(self, request: GenerationRequest) -> dict:
        job = self.store.create_job(request.project_id, request.name, request.model_dump())
        try:
            completed = self._execute_job(job["id"], request)
        except Exception as exc:
            completed = self.store.update_job(job["id"], status="failed", error_text=str(exc))
        return completed

    def _worker_loop(self) -> None:
        while True:
            try:
                job_id = self._job_queue.get(timeout=0.5)
            except queue.Empty:
                if self._job_queue.empty():
                    with self._worker_lock:
                        self._worker = None
                    self.maybe_unload_idle_engine()
                    return
                continue

            if job_id in self._cancelled_jobs:
                self._cancelled_jobs.discard(job_id)
                self.store.update_job(job_id, status="cancelled", error_text="Cancelled before execution.")
                self._job_queue.task_done()
                continue

            job = self.store.get_job(job_id)
            self._current_job_id = job_id
            try:
                request = GenerationRequest.model_validate(job["recipe"])
                self._execute_job_in_subprocess(job_id, request)
            except JobCancelledError:
                self.store.update_job(job_id, status="cancelled", error_text="Cancelled while rendering.")
            except Exception as exc:
                self.store.update_job(job_id, status="failed", error_text=str(exc))
            finally:
                self._current_job_id = None
                self._job_queue.task_done()

    def _ensure_worker(self) -> None:
        with self._worker_lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._worker_loop, daemon=True)
                self._worker.start()

    def enqueue_generation(self, request: GenerationRequest) -> dict:
        job = self.store.create_job(request.project_id, request.name, request.model_dump())
        self._job_queue.put(job["id"])
        self._ensure_worker()
        return job

    def cancel_job(self, job_id: int) -> dict:
        if self._current_job_id == job_id:
            self._cancelled_jobs.add(job_id)
            with self._active_process_lock:
                active_process = self._active_process
            if active_process is not None and active_process.is_alive():
                active_process.terminate()
                active_process.join(timeout=5)
            return self.store.update_job(job_id, status="cancelled", error_text="Cancelled while rendering.")
        self._cancelled_jobs.add(job_id)
        return self.store.update_job(job_id, status="cancelled", error_text="Cancellation requested.")

    def export_asset_bundle(self, asset_id: int) -> str:
        asset = self.store.get_audio_asset(asset_id)
        project = self.store.get_project_summary(asset["project_id"])
        bundle_dir = self.paths.exports / f"{project['slug']}-asset-{asset_id}"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        audio_source = Path(asset["path"])
        audio_target = bundle_dir / audio_source.name
        shutil.copy2(audio_source, audio_target)

        spec_source = (
            Path(asset["metadata"].get("spectrogram_path", "")) if asset["metadata"].get("spectrogram_path") else None
        )
        spec_target = None
        if spec_source and spec_source.exists():
            spec_target = bundle_dir / spec_source.name
            shutil.copy2(spec_source, spec_target)

        metadata_pretty = html.escape(json.dumps(asset["metadata"], indent=2, ensure_ascii=True))
        image_block = (
            f'<img src="{spec_target.name}" alt="Spectrogram" style="max-width: 100%; border-radius: 14px;" />'
            if spec_target
            else ""
        )
        index_path = bundle_dir / "index.html"
        index_path.write_text(
            f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(asset["label"])}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; max-width: 860px; margin: 0 auto; padding: 32px; background: #f4efe6; color: #182018; }}
    main {{ background: rgba(255,255,255,0.82); border: 1px solid rgba(24,32,24,0.1); border-radius: 24px; padding: 24px; }}
    audio {{ width: 100%; margin: 16px 0 24px; }}
    pre {{ overflow: auto; padding: 16px; border-radius: 16px; background: #112316; color: #e6f2e9; }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(asset["label"])}</h1>
    <p>Project: {html.escape(project["name"])}</p>
    <audio controls src="{audio_target.name}"></audio>
    {image_block}
    <h2>Metadata</h2>
    <pre>{metadata_pretty}</pre>
  </main>
</body>
</html>
""",
            encoding="utf-8",
        )
        return str(index_path)

    def system_profile(self) -> dict:
        profile_name = self.get_runtime_profile_name()
        profile = get_runtime_profile(profile_name)
        checkpoint_path = self.get_checkpoint_path()
        security = get_security_settings()
        engine_options = self._engine_options()
        effective_backend = engine_options["backend"]
        requested_backend = engine_options["requested_backend"]
        return SystemProfileView(
            profile=profile_name,
            profile_label=profile.label,
            description=profile.description,
            engine_loaded=self.engine.is_loaded(effective_backend),
            asr_backend=self._effective_asr_backend(),
            asr_model=self.get_asr_model_name(),
            device="apple-mlx" if effective_backend == "mlx" else self.engine.device,
            queue_depth=self._job_queue.qsize(),
            worker_alive=bool(self._worker and self._worker.is_alive()),
            current_job_id=self._current_job_id,
            idle_unload_seconds=self.get_idle_unload_seconds(),
            root_path=str(self.paths.root),
            cache_path=str(self.paths.cache),
            model_name="F5TTS_v1_Base",
            checkpoint_path=checkpoint_path or None,
            use_ema=self.get_use_ema(),
            requested_inference_backend=requested_backend,
            effective_inference_backend=effective_backend,
            inference_backend_label=self._backend_label(effective_backend),
            inference_backend_reason=engine_options.get("backend_reason"),
            mlx_available=self.engine.mlx_available(),
            last_warm_error=self._last_warm_error,
            auth_mode=security.auth_mode,
            auth_enabled=security.auth_enabled,
            public_surface=security.public_surface,
            upload_limit_mb=security.max_upload_mb,
            sharing_warning=security.sharing_warning,
            public_url=security.public_url or None,
        ).model_dump()

    def stage_upload(self, source_path: str) -> str:
        source = Path(source_path)
        ensure_upload_within_limit(source.stat().st_size)
        suffix = source.suffix or ".wav"
        with tempfile.NamedTemporaryFile(dir=self.paths.incoming, suffix=suffix, **tempfile_kwargs) as handle:
            staged_path = Path(handle.name)
        shutil.copy2(source, staged_path)
        return str(staged_path)


_service: StudioService | None = None
_service_lock = threading.Lock()


def get_service(paths: StudioPaths | None = None) -> StudioService:
    global _service
    if paths is not None:
        return StudioService(paths=paths)
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = StudioService()
    return _service
