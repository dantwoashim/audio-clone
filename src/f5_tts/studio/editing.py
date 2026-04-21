from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio

from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import save_spectrogram
from f5_tts.model.utils import convert_char_to_pinyin


TOKEN_PATTERN = re.compile(r"[\u3400-\u9FFF]|[A-Za-z0-9']+|[^\s]", re.UNICODE)


def tokenize_transcript(text: str) -> list[dict[str, Any]]:
    return [
        {
            "text": match.group(0),
            "normalized": normalize_token(match.group(0)),
            "start_char": match.start(),
            "end_char": match.end(),
        }
        for match in TOKEN_PATTERN.finditer(text)
        if normalize_token(match.group(0))
    ]


def normalize_token(text: str) -> str:
    return re.sub(r"[^\w\u3400-\u9FFF]+", "", text.lower(), flags=re.UNICODE)


def _extract_asr_words(result: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []) or []:
            token = (word.get("word") or "").strip()
            normalized = normalize_token(token)
            if not normalized:
                continue
            words.append(
                {
                    "text": token,
                    "normalized": normalized,
                    "start": float(word.get("start", 0.0) or 0.0),
                    "end": float(word.get("end", 0.0) or 0.0),
                }
            )
    return words


def align_transcript(
    audio_path: str,
    transcript: str = "",
    *,
    language: str | None = None,
    model_name: str = "mlx-community/whisper-medium",
) -> dict[str, Any]:
    try:
        import mlx_whisper
    except ImportError:
        mlx_whisper = None

    if mlx_whisper is not None:
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_name,
            language=language,
            word_timestamps=True,
        )
        asr_words = _extract_asr_words(result)
        confirmed_text = transcript.strip() or result["text"].strip()
    else:
        result = {"text": transcript.strip(), "segments": []}
        asr_words = []
        confirmed_text = transcript.strip()
        if not confirmed_text:
            raise ImportError("mlx-whisper is required to auto-align a transcript when no manual transcript is provided.")
    transcript_tokens = tokenize_transcript(confirmed_text)
    if not transcript_tokens:
        raise ValueError("Transcript is empty after tokenization.")

    if not asr_words:
        audio, sample_rate = torchaudio.load(audio_path)
        duration = audio.shape[-1] / sample_rate
        step = duration / max(len(transcript_tokens), 1)
        aligned_words = []
        for index, token in enumerate(transcript_tokens):
            start = index * step
            end = min(duration, (index + 1) * step)
            aligned_words.append(
                {
                    "index": index,
                    "word": token["text"],
                    "normalized": token["normalized"],
                    "start": start,
                    "end": end,
                    "confidence": 0.0,
                    "char_start": token["start_char"],
                    "char_end": token["end_char"],
                }
            )
        return {
            "transcript": confirmed_text,
            "alignment_backend": "interpolated",
            "asr_transcript": result["text"].strip(),
            "words": aligned_words,
            "warnings": ["Word timestamps were unavailable, so alignment was interpolated across the clip."],
        }

    matched_positions: list[int | None] = []
    search_from = 0
    for token in transcript_tokens:
        best_idx = None
        best_score = 0.0
        for idx in range(search_from, min(len(asr_words), search_from + 10)):
            candidate = asr_words[idx]["normalized"]
            if candidate == token["normalized"]:
                best_idx = idx
                best_score = 1.0
                break
            score = SequenceMatcher(None, token["normalized"], candidate).ratio()
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is not None and best_score >= 0.6:
            matched_positions.append(best_idx)
            search_from = best_idx + 1
        else:
            matched_positions.append(None)

    average_duration = sum(max(word["end"] - word["start"], 0.05) for word in asr_words) / max(len(asr_words), 1)
    full_audio, sample_rate = torchaudio.load(audio_path)
    clip_duration = full_audio.shape[-1] / sample_rate

    aligned_words: list[dict[str, Any]] = []
    for index, token in enumerate(transcript_tokens):
        matched_idx = matched_positions[index]
        if matched_idx is not None:
            start = asr_words[matched_idx]["start"]
            end = asr_words[matched_idx]["end"]
            confidence = 1.0
        else:
            prev_match = next((matched_positions[i] for i in range(index - 1, -1, -1) if matched_positions[i] is not None), None)
            next_match = next(
                (matched_positions[i] for i in range(index + 1, len(matched_positions)) if matched_positions[i] is not None),
                None,
            )
            if prev_match is not None and next_match is not None:
                start = asr_words[prev_match]["end"]
                end = asr_words[next_match]["start"]
                gap = max(end - start, average_duration)
                start = start + gap * 0.15
                end = min(end, start + average_duration)
            elif prev_match is not None:
                start = asr_words[prev_match]["end"]
                end = min(clip_duration, start + average_duration)
            elif next_match is not None:
                end = asr_words[next_match]["start"]
                start = max(0.0, end - average_duration)
            else:
                start = index * average_duration
                end = min(clip_duration, start + average_duration)
            confidence = 0.45

        aligned_words.append(
            {
                "index": index,
                "word": token["text"],
                "normalized": token["normalized"],
                "start": float(max(start, 0.0)),
                "end": float(max(end, start + 0.02)),
                "confidence": confidence,
                "char_start": token["start_char"],
                "char_end": token["end_char"],
            }
        )

    unmatched = sum(1 for item in aligned_words if item["confidence"] < 0.6)
    warnings: list[str] = []
    if unmatched:
        warnings.append(f"{unmatched} token(s) used interpolated timing because exact ASR alignment was unavailable.")

    return {
        "transcript": confirmed_text,
        "alignment_backend": "mlx_whisper_words",
        "asr_transcript": result["text"].strip(),
        "words": aligned_words,
        "warnings": warnings,
    }


def build_text_edit_plan(
    transcript: str,
    aligned_words: list[dict[str, Any]],
    target_text: str,
    replacement_text: str,
    *,
    occurrence: int = 1,
    preserve_timing: bool = True,
) -> dict[str, Any]:
    if occurrence < 1:
        raise ValueError("Occurrence must be at least 1.")

    transcript_tokens = tokenize_transcript(transcript)
    target_tokens = [token["normalized"] for token in tokenize_transcript(target_text) if token["normalized"]]
    if not target_tokens:
        raise ValueError("Target text is empty after tokenization.")

    normalized_transcript = [token["normalized"] for token in transcript_tokens]
    matches: list[tuple[int, int]] = []
    for index in range(0, len(normalized_transcript) - len(target_tokens) + 1):
        if normalized_transcript[index : index + len(target_tokens)] == target_tokens:
            matches.append((index, index + len(target_tokens) - 1))

    if len(matches) < occurrence:
        raise ValueError(f'Could not find occurrence {occurrence} of "{target_text}" in the transcript.')

    start_idx, end_idx = matches[occurrence - 1]
    start_word = aligned_words[start_idx]
    end_word = aligned_words[end_idx]
    char_start = transcript_tokens[start_idx]["start_char"]
    char_end = transcript_tokens[end_idx]["end_char"]
    edited_text = transcript[:char_start] + replacement_text.strip() + transcript[char_end:]

    original_duration = max(end_word["end"] - start_word["start"], 0.05)
    return {
        "original_text": transcript,
        "edited_text": edited_text,
        "target_text": target_text,
        "replacement_text": replacement_text.strip(),
        "occurrence": occurrence,
        "preserve_timing": preserve_timing,
        "spans": [
            {
                "start_word_index": start_idx,
                "end_word_index": end_idx,
                "start_seconds": float(start_word["start"]),
                "end_seconds": float(end_word["end"]),
                "original_duration_seconds": original_duration,
                "original_text": transcript[char_start:char_end],
                "replacement_text": replacement_text.strip(),
            }
        ],
    }


def render_speech_edit(
    audio_path: str,
    *,
    original_text: str,
    edited_text: str,
    spans: list[dict[str, Any]],
    ckpt_file: str = "",
    use_ema: bool = True,
    nfe_step: int = 32,
    cfg_strength: float = 2.0,
    sway_sampling_coef: float = -1.0,
    output_wav_path: str | None = None,
    output_spec_path: str | None = None,
    preserve_timing: bool = True,
) -> dict[str, Any]:
    tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt_file, use_ema=use_ema)
    tts._ensure_loaded()

    audio, sr = torchaudio.load(audio_path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    target_rms = 0.1
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != tts.target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, tts.target_sample_rate)
        audio = resampler(audio)

    audio = audio.to(tts.device)
    model = tts.ema_model
    hop_length = model.mel_spec.hop_length
    n_mel_channels = model.num_channels

    with torch.inference_mode():
        original_mel = model.mel_spec(audio).permute(0, 2, 1)

    mel_cond = torch.zeros(1, 0, n_mel_channels, device=tts.device)
    edit_mask = torch.zeros(1, 0, dtype=torch.bool, device=tts.device)
    offset_frame = 0

    for span_index, span in enumerate(spans):
        start_frame = round(span["start_seconds"] * tts.target_sample_rate / hop_length)
        end_frame = round(span["end_seconds"] * tts.target_sample_rate / hop_length)
        original_duration = span["original_duration_seconds"]
        if preserve_timing:
            duration_seconds = original_duration
        else:
            original_units = max(len(tokenize_transcript(span["original_text"])), 1)
            replacement_units = max(len(tokenize_transcript(span["replacement_text"])), 1)
            duration_seconds = original_duration * (replacement_units / original_units)
        duration_frames = round(duration_seconds * tts.target_sample_rate / hop_length)
        keep_frames = max(start_frame - offset_frame, 0)

        mel_cond = torch.cat(
            (
                mel_cond,
                original_mel[:, offset_frame:start_frame, :],
                torch.zeros(1, duration_frames, n_mel_channels, device=tts.device),
            ),
            dim=1,
        )
        edit_mask = torch.cat(
            (
                edit_mask,
                torch.ones(1, keep_frames, dtype=torch.bool, device=tts.device),
                torch.zeros(1, duration_frames, dtype=torch.bool, device=tts.device),
            ),
            dim=-1,
        )
        offset_frame = end_frame

    mel_cond = torch.cat((mel_cond, original_mel[:, offset_frame:, :]), dim=1)
    edit_mask = F.pad(edit_mask, (0, mel_cond.shape[1] - edit_mask.shape[-1]), value=True)

    final_text_list = convert_char_to_pinyin([edited_text])
    duration = mel_cond.shape[1]
    with torch.inference_mode():
        generated, _ = model.sample(
            cond=mel_cond,
            text=final_text_list,
            duration=duration,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            edit_mask=edit_mask,
        )
        generated = generated.to(torch.float32)
        gen_mel_spec = generated.permute(0, 2, 1)
        if tts.mel_spec_type == "vocos":
            generated_wave = tts.vocoder.decode(gen_mel_spec).cpu()
        else:
            generated_wave = tts.vocoder(gen_mel_spec).squeeze(0).cpu()

        if rms < target_rms:
            generated_wave = generated_wave * rms / target_rms

    generated_wave_np = generated_wave.squeeze().numpy()
    if output_wav_path:
        torchaudio.save(output_wav_path, generated_wave, tts.target_sample_rate)
    if output_spec_path:
        save_spectrogram(gen_mel_spec[0].cpu().numpy(), output_spec_path)

    return {
        "audio_path": output_wav_path,
        "spectrogram_path": output_spec_path,
        "sample_rate": tts.target_sample_rate,
        "duration_seconds": len(generated_wave_np) / tts.target_sample_rate,
        "text_excerpt": edited_text[:240],
        "original_text": original_text,
        "edited_text": edited_text,
        "edit_spans": spans,
        "nfe_step": nfe_step,
    }
