from __future__ import annotations

import unittest

from f5_tts.studio.editing import build_text_edit_plan


class StudioEditingTests(unittest.TestCase):
    def setUp(self):
        self.transcript = "Turn around. Hands on the desk. Turn around again."
        self.alignment = [
            {
                "index": 0,
                "word": "Turn",
                "normalized": "turn",
                "start": 0.00,
                "end": 0.20,
                "char_start": 0,
                "char_end": 4,
            },
            {
                "index": 1,
                "word": "around",
                "normalized": "around",
                "start": 0.21,
                "end": 0.52,
                "char_start": 5,
                "char_end": 11,
            },
            {
                "index": 2,
                "word": "Hands",
                "normalized": "hands",
                "start": 0.80,
                "end": 1.05,
                "char_start": 13,
                "char_end": 18,
            },
            {
                "index": 3,
                "word": "on",
                "normalized": "on",
                "start": 1.06,
                "end": 1.20,
                "char_start": 19,
                "char_end": 21,
            },
            {
                "index": 4,
                "word": "the",
                "normalized": "the",
                "start": 1.21,
                "end": 1.33,
                "char_start": 22,
                "char_end": 25,
            },
            {
                "index": 5,
                "word": "desk",
                "normalized": "desk",
                "start": 1.34,
                "end": 1.60,
                "char_start": 26,
                "char_end": 30,
            },
            {
                "index": 6,
                "word": "Turn",
                "normalized": "turn",
                "start": 2.00,
                "end": 2.18,
                "char_start": 32,
                "char_end": 36,
            },
            {
                "index": 7,
                "word": "around",
                "normalized": "around",
                "start": 2.19,
                "end": 2.46,
                "char_start": 37,
                "char_end": 43,
            },
            {
                "index": 8,
                "word": "again",
                "normalized": "again",
                "start": 2.47,
                "end": 2.72,
                "char_start": 44,
                "char_end": 49,
            },
        ]

    def test_build_text_edit_plan_replaces_requested_occurrence(self):
        plan = build_text_edit_plan(
            self.transcript,
            self.alignment,
            "turn around",
            "look back",
            occurrence=2,
            preserve_timing=True,
        )

        self.assertEqual(plan["replacement_text"], "look back")
        self.assertIn("Turn around.", plan["edited_text"])
        self.assertIn("look back again.", plan["edited_text"])
        self.assertEqual(plan["spans"][0]["start_word_index"], 6)
        self.assertEqual(plan["spans"][0]["end_word_index"], 7)

    def test_build_text_edit_plan_supports_delete(self):
        plan = build_text_edit_plan(
            self.transcript,
            self.alignment,
            "hands on the desk",
            "",
            action="delete",
            preserve_timing=False,
        )

        self.assertEqual(plan["action"], "delete")
        self.assertEqual(plan["replacement_text"], "")
        self.assertNotIn("Hands on the desk", plan["edited_text"])

    def test_build_text_edit_plan_supports_insert_after(self):
        plan = build_text_edit_plan(
            self.transcript,
            self.alignment,
            "turn around",
            "right now",
            action="insert_after",
            occurrence=1,
            preserve_timing=True,
        )

        self.assertEqual(plan["action"], "insert_after")
        self.assertIn("Turn around right now.", plan["edited_text"])
        self.assertEqual(plan["spans"][0]["start_seconds"], self.alignment[1]["end"])


if __name__ == "__main__":
    unittest.main()
