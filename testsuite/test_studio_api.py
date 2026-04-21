from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from f5_tts.studio.paths import StudioPaths
from f5_tts.studio.runtime import StudioService
from f5_tts.studio.schemas import ReferenceAnalysis
from f5_tts.studio.server import create_server_app
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


class StudioApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name)
        self.paths = make_paths(self.base)
        store = StudioStore(self.paths)
        store.set_setting("runtime_profile", "eco")
        self.service = StudioService(paths=self.paths)
        self.audio_path = self.base / "api-reference.wav"
        write_test_tone(self.audio_path)

        def fake_analysis(audio_path: str, transcript: str = "", backend: str = "auto"):
            return ReferenceAnalysis(
                transcript=transcript or "api transcript",
                duration_seconds=1.0,
                sample_rate=24000,
                channels=1,
                rms=0.2,
                peak=0.2,
                trailing_silence_seconds=0.3,
                speech_seconds=0.8,
                speech_ratio=0.8,
                backend="manual",
                warnings=[],
                notes=["ok"],
            )

        self.service.engine.analyze_reference = fake_analysis
        self.client = TestClient(create_server_app(mount_studio=False, service=self.service))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_project_and_reference_routes(self):
        create_response = self.client.post("/api/v1/projects", json={"name": "API Demo", "description": "local"})
        self.assertEqual(create_response.status_code, 200)
        project = create_response.json()
        self.assertEqual(project["name"], "API Demo")

        with self.audio_path.open("rb") as handle:
            response = self.client.post(
                "/api/v1/references/analyze",
                data={"project_id": str(project["id"]), "name": "Lead", "kind": "reference"},
                files={"audio": ("api-reference.wav", handle, "audio/wav")},
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["asset"]["name"], "Lead")
        self.assertEqual(payload["analysis"]["transcript"], "api transcript")


if __name__ == "__main__":
    unittest.main()
