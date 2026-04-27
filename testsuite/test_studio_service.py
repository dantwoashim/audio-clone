from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from f5_tts.studio.paths import StudioPaths
from f5_tts.studio.runtime import StudioService
from f5_tts.studio.schemas import GenerationRequest, ReferenceAnalysis
from f5_tts.studio.storage import StudioStore


def make_paths(base: Path) -> StudioPaths:
    return StudioPaths(
        root=base / "support",
        cache=base / "cache",
        projects=base / "support" / "projects",
        exports=base / "support" / "exports",
        incoming=base / "cache" / "incoming",
        logs=base / "support" / "logs",
        db_file=base / "support" / "app.db",
    ).ensure()


def write_test_tone(path: Path, seconds: float = 1.0, sample_rate: int = 24000) -> None:
    frames = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    wave = 0.2 * np.sin(2 * np.pi * 220 * frames)
    sf.write(path, wave, sample_rate)


class StudioServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name)
        self.paths = make_paths(self.base)
        store = StudioStore(self.paths)
        store.set_setting("runtime_profile", "eco")
        self.service = StudioService(paths=self.paths)
        self.audio_path = self.base / "reference.wav"
        write_test_tone(self.audio_path, seconds=1.5)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_ingest_reference_and_render_job(self):
        def fake_analysis(audio_path: str, transcript: str = "", backend: str = "auto", **_kwargs):
            return ReferenceAnalysis(
                transcript=transcript or "hello from the reference",
                duration_seconds=1.5,
                sample_rate=24000,
                channels=1,
                rms=0.2,
                peak=0.2,
                trailing_silence_seconds=0.3,
                speech_seconds=1.2,
                speech_ratio=0.8,
                backend="manual",
                warnings=[],
                notes=["ok"],
            )

        self.service.engine.analyze_reference = fake_analysis
        project = self.service.list_projects()[0]
        saved_reference, analysis = self.service.ingest_reference(project["id"], "Lead", str(self.audio_path), "")
        self.assertEqual(saved_reference["name"], "Lead")
        self.assertEqual(analysis["transcript"], "hello from the reference")

        output_path = self.paths.projects / project["slug"] / "outputs" / "dummy.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_test_tone(output_path, seconds=0.8)

        def fake_render(request, reference, style, pronunciation_rules, output_dir, **_kwargs):
            return {
                "audio_path": str(output_path),
                "spectrogram_path": None,
                "duration_seconds": 0.8,
                "elapsed_seconds": 0.2,
                "sample_rate": 24000,
                "text_excerpt": request.text[:40],
                "effective_speed": 1.0,
                "nfe_step": 20,
                "style_notes": [],
            }

        self.service.engine.render = fake_render
        request = GenerationRequest(
            project_id=project["id"],
            reference_id=saved_reference["id"],
            text="Render this voice for a quick smoke test.",
            name="Smoke Test",
            mode="preview",
        )
        job = self.service.render_now(request)
        self.assertEqual(job["status"], "completed")
        self.assertIn("asset_id", job["result"])
        detail = self.service.get_project_detail(project["id"])
        self.assertEqual(len(detail["references"]), 1)
        self.assertEqual(len(detail["assets"]), 1)

    def test_recommend_references_and_diagnose_asset(self):
        def fake_analysis(audio_path: str, transcript: str = "", backend: str = "auto", **_kwargs):
            return ReferenceAnalysis(
                transcript=transcript or "hello from the reference",
                duration_seconds=8.0,
                sample_rate=24000,
                channels=1,
                rms=0.04,
                peak=0.7,
                trailing_silence_seconds=0.4,
                speech_seconds=6.5,
                speech_ratio=0.81,
                backend="manual",
                quality_score=91.0,
                quality_rating="excellent",
                warnings=[],
                notes=["ok"],
            )

        self.service.engine.analyze_reference = fake_analysis
        self.service.engine.transcribe_audio = lambda *args, **kwargs: ("hello from the reference", "manual")

        project = self.service.list_projects()[0]
        saved_reference, _ = self.service.ingest_reference(project["id"], "Lead", str(self.audio_path), "")
        recommendations = self.service.recommend_references(project["id"])
        self.assertEqual(recommendations[0]["id"], saved_reference["id"])

        asset = self.service.store.save_audio_asset(
            project_id=project["id"],
            job_id=None,
            kind="final",
            label="Diagnostic Take",
            path=str(self.audio_path),
            duration_seconds=1.5,
            metadata={"requested_text": "hello from the reference", "reference_id": saved_reference["id"]},
        )

        report = self.service.diagnose_asset(project["id"], asset["id"], reference_id=saved_reference["id"])
        self.assertEqual(report["asset_id"], asset["id"])
        self.assertEqual(report["transcript_backend"], "manual")
        self.assertIsNotNone(report["word_error_rate"])

    def test_voice_profile_resolves_best_reference_for_render(self):
        def fake_analysis(audio_path: str, transcript: str = "", backend: str = "auto", **_kwargs):
            quality_score = 96.0 if "calm" in transcript else 84.0
            return ReferenceAnalysis(
                transcript=transcript or "hello from the reference",
                duration_seconds=8.5 if "calm" in transcript else 5.0,
                sample_rate=24000,
                channels=1,
                rms=0.04,
                peak=0.7,
                trailing_silence_seconds=0.3,
                speech_seconds=6.8 if "calm" in transcript else 4.4,
                speech_ratio=0.82 if "calm" in transcript else 0.74,
                backend="manual",
                quality_score=quality_score,
                quality_rating="excellent" if quality_score >= 90 else "good",
                warnings=[],
                notes=["ok"],
            )

        self.service.engine.analyze_reference = fake_analysis
        project = self.service.list_projects()[0]
        calm_reference, _ = self.service.ingest_reference(project["id"], "Calm Anchor", str(self.audio_path), "calm steady narrator")
        brisk_reference, _ = self.service.ingest_reference(project["id"], "Brisk Anchor", str(self.audio_path), "fast energetic promo")

        profile = self.service.save_voice_profile(
            project["id"],
            "Narrator Profile",
            [brisk_reference["id"], calm_reference["id"]],
            description="Curated for narration",
        )
        recommendations = self.service.recommend_profile_references(
            profile["id"],
            context_notes="calm narrator",
            text="This should sound measured and steady.",
        )
        self.assertEqual(recommendations[0]["id"], calm_reference["id"])

        output_path = self.paths.projects / project["slug"] / "outputs" / "profile.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_test_tone(output_path, seconds=0.8)

        def fake_render(request, reference, style, pronunciation_rules, output_dir, **_kwargs):
            return {
                "audio_path": str(output_path),
                "spectrogram_path": None,
                "duration_seconds": 0.8,
                "elapsed_seconds": 0.2,
                "sample_rate": 24000,
                "text_excerpt": request.text[:40],
                "effective_speed": 0.95,
                "nfe_step": 20,
                "seed": 1234,
                "style_notes": [],
            }

        self.service.engine.render = fake_render
        job = self.service.render_now(
            GenerationRequest(
                project_id=project["id"],
                voice_profile_id=profile["id"],
                text="This should sound measured and steady.",
                name="Profile Render",
                mode="final",
                context_notes="calm narrator",
            )
        )
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["result"]["reference_id"], calm_reference["id"])
        self.assertEqual(job["result"]["voice_profile_id"], profile["id"])
        self.assertEqual(job["result"]["resolved_reference_name"], "Calm Anchor")

    def test_mlx_backend_selection_and_checkpoint_fallback(self):
        self.service.engine.mlx_available = lambda: True
        self.service.set_inference_backend("mlx")

        base_request = GenerationRequest(
            project_id=self.service.list_projects()[0]["id"],
            reference_id=1,
            text="Base model request",
        )
        base_options = self.service._engine_options_for_request(base_request)
        self.assertEqual(base_options["backend"], "mlx")
        self.assertIsNone(base_options["backend_reason"])

        checkpoint_path = self.base / "dummy.safetensors"
        checkpoint_path.write_bytes(b"checkpoint")
        checkpoint_request = GenerationRequest(
            project_id=self.service.list_projects()[0]["id"],
            reference_id=1,
            text="Checkpoint request",
            checkpoint_path=str(checkpoint_path),
        )
        checkpoint_options = self.service._engine_options_for_request(checkpoint_request)
        self.assertEqual(checkpoint_options["backend"], "pytorch")
        self.assertIn("base model", checkpoint_options["backend_reason"])


if __name__ == "__main__":
    unittest.main()
