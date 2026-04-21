from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torchaudio


TOKEN_PATTERN = re.compile(r"[\u3400-\u9FFF]|[A-Za-z0-9']+|[^\s]", re.UNICODE)


def normalized_word_tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "") if re.search(r"\w|[\u3400-\u9FFF]", token)]


def normalized_char_tokens(text: str) -> list[str]:
    lowered = (text or "").lower()
    return [char for char in lowered if not char.isspace()]


def _levenshtein(left: Sequence[Any], right: Sequence[Any]) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        current = [i]
        for j, right_token in enumerate(right, start=1):
            substitution_cost = 0 if left_token == right_token else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(expected_text: str, observed_text: str) -> float | None:
    expected_tokens = normalized_word_tokens(expected_text)
    observed_tokens = normalized_word_tokens(observed_text)
    if not expected_tokens:
        return None
    return _levenshtein(expected_tokens, observed_tokens) / max(len(expected_tokens), 1)


def char_error_rate(expected_text: str, observed_text: str) -> float | None:
    expected_tokens = normalized_char_tokens(expected_text)
    observed_tokens = normalized_char_tokens(observed_text)
    if not expected_tokens:
        return None
    return _levenshtein(expected_tokens, observed_tokens) / max(len(expected_tokens), 1)


def _load_mono_audio(audio_path: str, target_sample_rate: int = 24000) -> tuple[torch.Tensor, int]:
    audio, sample_rate = torchaudio.load(audio_path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        audio = torchaudio.functional.resample(audio, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return audio.float(), sample_rate


def audio_health_report(audio_path: str, target_sample_rate: int = 24000) -> dict[str, float]:
    audio, sample_rate = _load_mono_audio(audio_path, target_sample_rate=target_sample_rate)
    mono = audio.squeeze(0)
    duration_seconds = mono.shape[-1] / sample_rate if mono.numel() else 0.0
    if mono.numel() == 0:
        return {
            "duration_seconds": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "clipping_ratio": 0.0,
            "silence_ratio": 1.0,
            "trailing_silence_seconds": 0.0,
            "dynamic_range": 0.0,
        }

    magnitude = mono.abs()
    rms = float(torch.sqrt(torch.mean(mono.square())).item())
    peak = float(magnitude.max().item())
    clipping_ratio = float((magnitude >= 0.98).float().mean().item())
    silence_threshold = max(rms * 0.25, 0.008)
    silence_ratio = float((magnitude <= silence_threshold).float().mean().item())

    window = max(sample_rate // 20, 1)
    trailing_silence_seconds = 0.0
    for start in range(max(mono.shape[-1] - window, 0), -1, -window):
        segment = magnitude[start : start + window]
        if segment.numel() == 0 or float(segment.mean().item()) > silence_threshold:
            break
        trailing_silence_seconds += segment.shape[-1] / sample_rate

    p05 = float(torch.quantile(magnitude, 0.05).item())
    p95 = float(torch.quantile(magnitude, 0.95).item())
    dynamic_range = max(p95 - p05, 0.0)

    return {
        "duration_seconds": duration_seconds,
        "rms": rms,
        "peak": peak,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
        "trailing_silence_seconds": trailing_silence_seconds,
        "dynamic_range": dynamic_range,
    }


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float | None:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))


def voice_similarity_proxy(reference_audio_path: str, candidate_audio_path: str, target_sample_rate: int = 24000) -> float | None:
    try:
        reference_audio, _ = _load_mono_audio(reference_audio_path, target_sample_rate=target_sample_rate)
        candidate_audio, _ = _load_mono_audio(candidate_audio_path, target_sample_rate=target_sample_rate)
    except Exception:
        return None

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
    )
    reference_spec = torch.log(mel(reference_audio).clamp_min(1e-5)).mean(dim=-1).squeeze(0).cpu().numpy()
    candidate_spec = torch.log(mel(candidate_audio).clamp_min(1e-5)).mean(dim=-1).squeeze(0).cpu().numpy()
    return _cosine_similarity(reference_spec, candidate_spec)


def reference_quality_breakdown(analysis: dict[str, Any]) -> dict[str, Any]:
    duration = float(analysis.get("duration_seconds", 0.0) or 0.0)
    rms = float(analysis.get("rms", 0.0) or 0.0)
    peak = float(analysis.get("peak", 0.0) or 0.0)
    trailing_silence = float(analysis.get("trailing_silence_seconds", 0.0) or 0.0)
    speech_ratio = analysis.get("speech_ratio")
    speech_ratio = float(speech_ratio) if speech_ratio is not None else None
    warnings = list(analysis.get("warnings") or [])

    score = 70.0
    strengths: list[str] = []
    issues: list[str] = []

    if 6.0 <= duration <= 12.0:
        score += 16.0
        strengths.append("Ideal 6-12 second identity window.")
    elif 4.0 <= duration <= 14.0:
        score += 8.0
        strengths.append("Usable duration for zero-shot cloning.")
    elif duration > 0:
        score -= 12.0
        issues.append("Duration is outside the most reliable range.")

    if 0.018 <= rms <= 0.16:
        score += 8.0
        strengths.append("Healthy reference loudness.")
    elif rms < 0.012:
        score -= 10.0
        issues.append("Reference level is quiet.")

    if peak < 0.96:
        score += 6.0
        strengths.append("Peak level leaves headroom.")
    else:
        score -= 12.0
        issues.append("Peaks suggest clipping risk.")

    if 0.2 <= trailing_silence <= 1.2:
        score += 5.0
        strengths.append("Trailing silence is in a safe range.")
    elif trailing_silence < 0.08:
        score -= 6.0
        issues.append("Very little trailing silence may truncate endings.")

    if speech_ratio is not None:
        if speech_ratio >= 0.72:
            score += 10.0
            strengths.append("Speech dominates the clip.")
        elif speech_ratio < 0.55:
            score -= 8.0
            issues.append("There is limited speech relative to clip length.")

    score -= min(len(warnings), 4) * 3.0
    issues.extend(warnings[:3])

    score = max(0.0, min(score, 100.0))
    if score >= 88:
        rating = "excellent"
    elif score >= 76:
        rating = "strong"
    elif score >= 62:
        rating = "usable"
    else:
        rating = "needs work"

    return {
        "score": round(score, 1),
        "rating": rating,
        "strengths": strengths[:3],
        "issues": issues[:4],
    }


def diagnose_render(
    *,
    generated_audio_path: str,
    expected_text: str = "",
    observed_text: str = "",
    reference_audio_path: str | None = None,
) -> dict[str, Any]:
    health = audio_health_report(generated_audio_path)
    wer = word_error_rate(expected_text, observed_text)
    cer = char_error_rate(expected_text, observed_text)
    similarity = voice_similarity_proxy(reference_audio_path, generated_audio_path) if reference_audio_path else None

    warnings: list[str] = []
    if health["peak"] >= 0.98 or health["clipping_ratio"] > 0.002:
        warnings.append("Rendered audio may be clipping.")
    if health["silence_ratio"] > 0.7:
        warnings.append("Rendered audio contains a lot of silence.")
    if wer is not None and wer > 0.22:
        warnings.append("ASR drift is high relative to the requested text.")
    if similarity is not None and similarity < 0.78:
        warnings.append("Voice similarity proxy is weak against the selected reference.")

    summary_bits = [
        f"RMS {health['rms']:.3f}",
        f"Peak {health['peak']:.3f}",
    ]
    if wer is not None:
        summary_bits.append(f"WER {wer:.2%}")
    if similarity is not None:
        summary_bits.append(f"Similarity {similarity:.2f}")

    return {
        "expected_text": expected_text,
        "observed_text": observed_text,
        "word_error_rate": wer,
        "char_error_rate": cer,
        "voice_similarity_proxy": similarity,
        "audio_health": health,
        "warnings": warnings,
        "summary": " | ".join(summary_bits),
    }
