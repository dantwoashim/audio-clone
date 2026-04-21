from __future__ import annotations

import unittest

from f5_tts.studio.runtime import normalize_script


class TextNormalizationTests(unittest.TestCase):
    def test_pronunciation_and_time_normalization(self):
        rules = [{"source": "GPU", "replacement": "gee pee you"}]
        text = "GPU demo starts at 7:30! 12 users joined."
        normalized = normalize_script(text, rules)
        self.assertIn("gee pee you", normalized.lower())
        self.assertIn("seven thirty", normalized)
        self.assertIn("twelve users", normalized)


if __name__ == "__main__":
    unittest.main()
