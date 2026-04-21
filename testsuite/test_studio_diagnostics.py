from __future__ import annotations

import unittest

from f5_tts.studio.diagnostics import char_error_rate, reference_quality_breakdown, word_error_rate


class StudioDiagnosticsTests(unittest.TestCase):
    def test_error_rates_drop_for_matching_text(self):
        self.assertEqual(word_error_rate("Turn around now", "turn around now"), 0.0)
        self.assertEqual(char_error_rate("Hello", "hello"), 0.0)

    def test_reference_quality_rewards_clean_reference_window(self):
        strong = reference_quality_breakdown(
            {
                "duration_seconds": 8.5,
                "rms": 0.04,
                "peak": 0.82,
                "trailing_silence_seconds": 0.4,
                "speech_ratio": 0.82,
                "warnings": [],
            }
        )
        weak = reference_quality_breakdown(
            {
                "duration_seconds": 2.1,
                "rms": 0.008,
                "peak": 0.995,
                "trailing_silence_seconds": 0.01,
                "speech_ratio": 0.34,
                "warnings": ["Reference may be clipping."],
            }
        )

        self.assertGreater(strong["score"], weak["score"])
        self.assertIn(strong["rating"], {"excellent", "strong"})
        self.assertIn(weak["rating"], {"usable", "needs work"})


if __name__ == "__main__":
    unittest.main()
