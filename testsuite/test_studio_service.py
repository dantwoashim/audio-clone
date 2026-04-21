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


if __name__ == "__main__":
    unittest.main()
