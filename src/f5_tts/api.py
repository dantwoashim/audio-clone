import random
import threading
from collections import OrderedDict
from collections.abc import Callable
from importlib.resources import files

import soundfile as sf
import torchaudio
import tqdm
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    transcribe,
)
from f5_tts.model.utils import seed_everything


class F5TTS:
    def __init__(
        self,
        model="F5TTS_v1_Base",
        ckpt_file="",
        vocab_file="",
        ode_method="euler",
        use_ema=True,
        vocoder_local_path=None,
        device=None,
        hf_cache_dir=None,
    ):
        model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
        self._model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        self._model_arc = model_cfg.model.arch

        self.mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
        self.target_sample_rate = model_cfg.model.mel_spec.target_sample_rate

        self.model_name = model
        self.ckpt_file = ckpt_file
        self.vocab_file = vocab_file
        self.ode_method = ode_method
        self.use_ema = use_ema
        self.vocoder_local_path = vocoder_local_path
        self.hf_cache_dir = hf_cache_dir
        self._reference_cache: OrderedDict[tuple[str, str], dict] = OrderedDict()
        self._reference_cache_limit = 8
        self._is_warmed = False
        self._infer_lock = threading.RLock()
        self.vocoder = None
        self.ema_model = None

        if device is not None:
            self.device = device
        else:
            import torch

            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "xpu"
                if torch.xpu.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

    def _resolve_checkpoint_file(self) -> str:
        if self.ckpt_file:
            return self.ckpt_file

        repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"
        model_name = self.model_name

        if model_name == "F5TTS_Base":
            if self.mel_spec_type == "vocos":
                ckpt_step = 1200000
            elif self.mel_spec_type == "bigvgan":
                model_name = "F5TTS_Base_bigvgan"
                ckpt_type = "pt"
        elif model_name == "E2TTS_Base":
            repo_name = "E2-TTS"
            ckpt_step = 1200000

        return str(
            cached_path(
                f"hf://SWivid/{repo_name}/{model_name}/model_{ckpt_step}.{ckpt_type}",
                cache_dir=self.hf_cache_dir,
            )
        )

    def _ensure_loaded(self):
        if self.vocoder is not None and self.ema_model is not None:
            return
        with self._infer_lock:
            if self.vocoder is not None and self.ema_model is not None:
                return
            self.vocoder = load_vocoder(
                self.mel_spec_type,
                self.vocoder_local_path is not None,
                self.vocoder_local_path,
                self.device,
                self.hf_cache_dir,
            )
            self.ema_model = load_model(
                self._model_cls,
                self._model_arc,
                self._resolve_checkpoint_file(),
                self.mel_spec_type,
                self.vocab_file,
                self.ode_method,
                self.use_ema,
                self.device,
            )

    def prepare_reference(self, ref_file, ref_text, show_info: Callable = print):
        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text, show_info=show_info)
        cache_key = (ref_file, ref_text)
        if cache_key not in self._reference_cache:
            audio, sr = torchaudio.load(ref_file)
            self._reference_cache[cache_key] = {
                "ref_audio": ref_file,
                "ref_text": ref_text,
                "audio": audio,
                "sample_rate": sr,
            }
            while len(self._reference_cache) > self._reference_cache_limit:
                self._reference_cache.popitem(last=False)
        else:
            self._reference_cache.move_to_end(cache_key)
        return self._reference_cache[cache_key]

    def warm_up(self, show_info: Callable = print):
        if self._is_warmed:
            return

        with self._infer_lock:
            if self._is_warmed:
                return
            self._ensure_loaded()

            warm_ref = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
            prepared_reference = self.prepare_reference(
                warm_ref,
                "Some call me nature, others call me mother nature.",
                show_info=show_info,
            )
            infer_process(
                (prepared_reference["audio"], prepared_reference["sample_rate"]),
                prepared_reference["ref_text"],
                "Warm up the voice model.",
                self.ema_model,
                self.vocoder,
                self.mel_spec_type,
                show_info=show_info,
                progress=None,
                nfe_step=8,
                cfg_strength=2,
                sway_sampling_coef=-1,
                speed=1.0,
                device=self.device,
            )
            self._is_warmed = True

    def transcribe(self, ref_audio, language=None):
        return transcribe(ref_audio, language)

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)

        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spec, file_spec):
        save_spectrogram(spec, file_spec)

    def infer_prepared(
        self,
        prepared_reference,
        gen_text,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        file_wave=None,
        file_spec=None,
        seed=None,
    ):
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        seed_everything(seed)
        self.seed = seed

        self._ensure_loaded()
        with self._infer_lock:
            wav, sr, spec = infer_process(
                (prepared_reference["audio"], prepared_reference["sample_rate"]),
                prepared_reference["ref_text"],
                gen_text,
                self.ema_model,
                self.vocoder,
                self.mel_spec_type,
                show_info=show_info,
                progress=progress,
                target_rms=target_rms,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                speed=speed,
                fix_duration=fix_duration,
                device=self.device,
            )

        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)

        if file_spec is not None:
            self.export_spectrogram(spec, file_spec)

        return wav, sr, spec

    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        file_wave=None,
        file_spec=None,
        seed=None,
    ):
        prepared_reference = self.prepare_reference(ref_file, ref_text, show_info=show_info)
        return self.infer_prepared(
            prepared_reference,
            gen_text,
            show_info=show_info,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            sway_sampling_coef=sway_sampling_coef,
            cfg_strength=cfg_strength,
            nfe_step=nfe_step,
            speed=speed,
            fix_duration=fix_duration,
            remove_silence=remove_silence,
            file_wave=file_wave,
            file_spec=file_spec,
            seed=seed,
        )


if __name__ == "__main__":
    f5tts = F5TTS()

    wav, sr, spec = f5tts.infer(
        ref_file=str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")),
        ref_text="Some call me nature, others call me mother nature.",
        gen_text="I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring.",
        file_wave=str(files("f5_tts").joinpath("../../tests/api_out.wav")),
        file_spec=str(files("f5_tts").joinpath("../../tests/api_out.png")),
        seed=None,
    )

    print("seed :", f5tts.seed)
