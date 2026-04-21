from __future__ import annotations

import argparse
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from unidecode import unidecode

from f5_tts.infer.utils_infer import (
    cfg_strength as default_cfg_strength,
    cross_fade_duration as default_cross_fade_duration,
    device as default_device,
    fix_duration as default_fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    mel_spec_type as default_mel_spec_type,
    nfe_step as default_nfe_step,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    speed as default_speed,
    sway_sampling_coef as default_sway_sampling_coef,
    target_rms as default_target_rms,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 infer-cli.py",
        description="Command line interface for E2/F5 TTS with advanced batch processing.",
        epilog="Specify options above to override one or more settings from config.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=os.path.join(files("f5_tts").joinpath("infer/examples/basic"), "basic.toml"),
        help="The configuration file, default see infer/examples/basic/basic.toml",
    )
    parser.add_argument("-m", "--model", type=str, help="The model name: F5TTS_v1_Base | F5TTS_Base | E2TTS_Base | etc.")
    parser.add_argument("-mc", "--model_cfg", type=str, help="The path to F5-TTS model config file .yaml")
    parser.add_argument("-p", "--ckpt_file", type=str, help="The path to model checkpoint .pt, leave blank to use default")
    parser.add_argument("-v", "--vocab_file", type=str, help="The path to vocab file .txt, leave blank to use default")
    parser.add_argument("-r", "--ref_audio", type=str, help="The reference audio file.")
    parser.add_argument("-s", "--ref_text", type=str, help="The transcript/subtitle for the reference audio")
    parser.add_argument("-t", "--gen_text", type=str, help="The text to make model synthesize a speech")
    parser.add_argument("-f", "--gen_file", type=str, help="The file with text to generate, will ignore --gen_text")
    parser.add_argument("-o", "--output_dir", type=str, help="The path to output folder")
    parser.add_argument("-w", "--output_file", type=str, help="The name of output file")
    parser.add_argument("--save_chunk", action="store_true", help="Save each audio chunk during inference")
    parser.add_argument(
        "--no_legacy_text",
        action="store_false",
        help="Do not use lossy ASCII transliterations of unicode text in saved file names.",
    )
    parser.add_argument("--remove_silence", action="store_true", help="Remove long silence found in output")
    parser.add_argument(
        "--load_vocoder_from_local",
        action="store_true",
        help="Load vocoder from local dir, default to ../checkpoints/vocos-mel-24khz",
    )
    parser.add_argument(
        "--vocoder_name",
        type=str,
        choices=["vocos", "bigvgan"],
        help=f"Used vocoder name: vocos | bigvgan, default {default_mel_spec_type}",
    )
    parser.add_argument(
        "--target_rms",
        type=float,
        help=f"Target output speech loudness normalization value, default {default_target_rms}",
    )
    parser.add_argument(
        "--cross_fade_duration",
        type=float,
        help=f"Duration of cross-fade between audio segments in seconds, default {default_cross_fade_duration}",
    )
    parser.add_argument("--nfe_step", type=int, help=f"The number of denoising steps, default {default_nfe_step}")
    parser.add_argument("--cfg_strength", type=float, help=f"Classifier-free guidance strength, default {default_cfg_strength}")
    parser.add_argument(
        "--sway_sampling_coef",
        type=float,
        help=f"Sway sampling coefficient, default {default_sway_sampling_coef}",
    )
    parser.add_argument("--speed", type=float, help=f"The speed of the generated audio, default {default_speed}")
    parser.add_argument(
        "--fix_duration",
        type=float,
        help=f"Fix the total duration (reference and generated audio) in seconds, default {default_fix_duration}",
    )
    parser.add_argument("--device", type=str, help="Specify the device to run on")
    return parser


def _resolve_example_path(value: str) -> str:
    if not value:
        return value
    if "infer/examples/" in value:
        return str(files("f5_tts").joinpath(value))
    return value


def _load_config(path: str) -> dict:
    with open(path, "rb") as handle:
        return tomli.load(handle)


def _prepare_model(
    model: str,
    model_cfg_path: str | None,
    ckpt_file: str,
    vocab_file: str,
    vocoder_name: str,
    load_vocoder_from_local: bool,
    device: str,
):
    if vocoder_name == "vocos":
        vocoder_local_path = "../checkpoints/vocos-mel-24khz"
    else:
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"

    vocoder = load_vocoder(
        vocoder_name=vocoder_name,
        is_local=load_vocoder_from_local,
        local_path=vocoder_local_path,
        device=device,
    )

    model_cfg = OmegaConf.load(
        model_cfg_path or str(files("f5_tts").joinpath(f"configs/{model}.yaml"))
    )
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"

    if model != "F5TTS_Base":
        assert vocoder_name == model_cfg.model.mel_spec.mel_spec_type

    if model == "F5TTS_Base":
        if vocoder_name == "vocos":
            ckpt_step = 1200000
        elif vocoder_name == "bigvgan":
            model = "F5TTS_Base_bigvgan"
            ckpt_type = "pt"
    elif model == "E2TTS_Base":
        repo_name = "E2-TTS"
        ckpt_step = 1200000

    if not ckpt_file:
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}"))
    elif ckpt_file.startswith("hf://"):
        ckpt_file = str(cached_path(ckpt_file))

    if vocab_file.startswith("hf://"):
        vocab_file = str(cached_path(vocab_file))

    print(f"Using {model}...")
    ema_model = load_model(
        model_cls,
        model_arc,
        ckpt_file,
        mel_spec_type=vocoder_name,
        vocab_file=vocab_file,
        device=device,
    )
    return ema_model, vocoder


def main():
    args = build_parser().parse_args()
    config = _load_config(args.config)

    model = args.model or config.get("model", "F5TTS_v1_Base")
    ckpt_file = args.ckpt_file or config.get("ckpt_file", "")
    vocab_file = args.vocab_file or config.get("vocab_file", "")

    ref_audio = _resolve_example_path(args.ref_audio or config.get("ref_audio", "infer/examples/basic/basic_ref_en.wav"))
    ref_text = (
        args.ref_text
        if args.ref_text is not None
        else config.get("ref_text", "Some call me nature, others call me mother nature.")
    )
    gen_text = args.gen_text or config.get("gen_text", "Here we generate something just for test.")
    gen_file = _resolve_example_path(args.gen_file or config.get("gen_file", ""))

    output_dir = args.output_dir or config.get("output_dir", "tests")
    output_file = args.output_file or config.get(
        "output_file",
        f"infer_cli_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav",
    )

    save_chunk = args.save_chunk or config.get("save_chunk", False)
    use_legacy_text = args.no_legacy_text or config.get("no_legacy_text", False)
    if save_chunk and use_legacy_text:
        print(
            "\nWarning to --save_chunk: lossy ASCII transliterations of unicode text for legacy (.wav) file names, --no_legacy_text to disable.\n"
        )

    remove_silence = args.remove_silence or config.get("remove_silence", False)
    load_vocoder_from_local = args.load_vocoder_from_local or config.get("load_vocoder_from_local", False)

    vocoder_name = args.vocoder_name or config.get("vocoder_name", default_mel_spec_type)
    target_rms = args.target_rms or config.get("target_rms", default_target_rms)
    cross_fade_duration = args.cross_fade_duration or config.get("cross_fade_duration", default_cross_fade_duration)
    nfe_step = args.nfe_step or config.get("nfe_step", default_nfe_step)
    cfg_strength = args.cfg_strength or config.get("cfg_strength", default_cfg_strength)
    sway_sampling_coef = args.sway_sampling_coef or config.get("sway_sampling_coef", default_sway_sampling_coef)
    speed = args.speed or config.get("speed", default_speed)
    fix_duration = args.fix_duration or config.get("fix_duration", default_fix_duration)
    device = args.device or config.get("device", default_device)

    if "voices" in config:
        for voice in config["voices"]:
            config["voices"][voice]["ref_audio"] = _resolve_example_path(config["voices"][voice]["ref_audio"])

    if gen_file:
        gen_text = codecs.open(gen_file, "r", "utf-8").read()

    wave_path = Path(output_dir) / output_file
    output_chunk_dir = Path(output_dir) / f"{Path(output_file).stem}_chunks"
    if save_chunk:
        output_chunk_dir.mkdir(parents=True, exist_ok=True)

    ema_model, vocoder = _prepare_model(
        model=model,
        model_cfg_path=args.model_cfg or config.get("model_cfg"),
        ckpt_file=ckpt_file,
        vocab_file=vocab_file,
        vocoder_name=vocoder_name,
        load_vocoder_from_local=load_vocoder_from_local,
        device=device,
    )

    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    voices = dict(config.get("voices", {}))
    voices["main"] = main_voice

    for voice_name, payload in voices.items():
        print("Voice:", voice_name)
        print("ref_audio ", payload["ref_audio"])
        payload["ref_audio"], payload["ref_text"] = preprocess_ref_audio_text(
            payload["ref_audio"],
            payload["ref_text"],
        )
        print("ref_audio_", payload["ref_audio"], "\n\n")

    generated_audio_segments = []
    chunks = re.split(r"(?=\[\w+\])", gen_text)
    voice_tag_pattern = r"\[(\w+)\]"
    final_sample_rate = 24000

    for text in chunks:
        if not text.strip():
            continue
        match = re.match(voice_tag_pattern, text)
        voice_name = match[1] if match else "main"
        if voice_name not in voices:
            print(f"Voice {voice_name} not found, using main.")
            voice_name = "main"
        text = re.sub(voice_tag_pattern, "", text)
        ref_audio_ = voices[voice_name]["ref_audio"]
        ref_text_ = voices[voice_name]["ref_text"]
        local_speed = voices[voice_name].get("speed", speed)
        gen_text_ = text.strip()
        print(f"Voice: {voice_name}")
        audio_segment, final_sample_rate, _ = infer_process(
            ref_audio_,
            ref_text_,
            gen_text_,
            ema_model,
            vocoder,
            mel_spec_type=vocoder_name,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=local_speed,
            fix_duration=fix_duration,
            device=device,
        )
        generated_audio_segments.append(audio_segment)

        if save_chunk:
            chunk_label = gen_text_
            if len(chunk_label) > 200:
                chunk_label = chunk_label[:200] + " ... "
            if use_legacy_text:
                chunk_label = unidecode(chunk_label)
            sf.write(
                output_chunk_dir / f"{len(generated_audio_segments) - 1}_{chunk_label}.wav",
                audio_segment,
                final_sample_rate,
            )

    if not generated_audio_segments:
        return

    final_wave = np.concatenate(generated_audio_segments)
    wave_path.parent.mkdir(parents=True, exist_ok=True)
    with open(wave_path, "wb") as handle:
        sf.write(handle.name, final_wave, final_sample_rate)
        if remove_silence:
            remove_silence_for_generated_wav(handle.name)
        print(handle.name)


if __name__ == "__main__":
    main()
