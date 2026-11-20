"""File-only regression tests for importing sealed requests from an incomplete eval503."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import evaluate as ev


class SealedImport(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.src = root / "eval503"
        self.out = root / "eval200"
        self.src.mkdir()
        self.out.mkdir()
        self.parent = {
            "window": 1,
            "max_new": 2,
            "eos_token_id": 40,
            "codec_binding": {"id": "fixture"},
            "checkpoint_sha256": "weight",
            "roster": "original",
            "training": {"models": "model"},
        }
        self.child = {**self.parent, "scope": "eval200", "ordered_ids_sha256": ev.IDS_SHA}
        self.rows = [
            {"request_id": rid, "expected_target_token_ids": [448, 3]} for rid in ("sealed", "partial")
        ]
        ev.write(self.src / "contract.json", self.parent)
        ev.write(self.src / "progress.json", {"completed_requests": 200})
        for arm in ev.ARMS:
            (self.src / arm).mkdir()
            for row in self.rows:
                if arm == "active" and row["request_id"] == "partial":
                    continue
                rid = row["request_id"]
                trial = {
                    "arm": arm,
                    "request_id": rid,
                    "trial_id": rid + "-" + arm,
                    "warmup": False,
                    "output_ids": [448, 3],
                    "contract_id": ev.digest(self.parent),
                    "codec_binding": self.parent["codec_binding"],
                    "verification_rounds": 2,
                    "accepted_draft_tokens": 1,
                    "rounds": [
                        {
                            "candidate_ids": [714],
                            "accepted": 0,
                            "committed_ids": [448],
                            "feedback_bytes": 0 if arm == "raw" else 8,
                        },
                        {"candidate_ids": [3], "accepted": 1, "committed_ids": [3], "feedback_bytes": 0},
                    ],
                }
                ev.write(self.src / arm / (rid + ".json"), trial)

    def recover(self):
        return ev.recover_rows(self.out, self.rows, self.child, self.parent, self.src, sealed_only=True)

    def test_import_and_resume(self):
        before = {p: ev.sha(p) for p in self.src.rglob("*.json")}
        self.assertEqual(self.recover(), ["sealed"])
        for arm in ev.ARMS:
            dest = ev.read(self.out / arm / "sealed.json")
            self.assertEqual(dest["source_row_sha256"], before[self.src / arm / "sealed.json"])
            self.assertEqual(dest["source_contract_id"], ev.digest(self.parent))
            self.assertEqual(dest["measurement_origin"], "archived_eval503")
            self.assertFalse((self.out / arm / "partial.json").exists())
        self.assertEqual(self.recover(), [])
        self.assertEqual(before, {p: ev.sha(p) for p in self.src.rglob("*.json")})
        self.assertFalse((self.out / "completion.json").exists())

    def test_source_contract_mismatch(self):
        ev.write(self.src / "contract.json", {**self.parent, "checkpoint_sha256": "different"})
        with self.assertRaisesRegex(ValueError, "503 source"):
            self.recover()
        self.assertFalse((self.out / "raw" / "sealed.json").exists())

    def test_complete_mode_requires_completion(self):
        with self.assertRaises(FileNotFoundError):
            ev.recover_rows(self.out, self.rows, self.child, self.parent, self.src)

    def test_bad_shadow_or_active_imports_no_arm(self):
        shadow = self.src / "shadow" / "sealed.json"
        orig = ev.read(shadow)
        bad = ev.read(shadow)
        bad["rounds"][0]["candidate_ids"] = [713]  # still a valid rejection, but differs from RAW
        ev.write(shadow, bad)
        with self.assertRaisesRegex(ValueError, "RAW/SHADOW"):
            self.recover()
        ev.write(shadow, orig)
        active = self.src / "active" / "sealed.json"
        bad = ev.read(active)
        bad["rounds"][0]["feedback_bytes"] = 0
        ev.write(active, bad)
        with self.assertRaises(ValueError):
            self.recover()
        self.assertFalse(any(self.out.glob("*/sealed.json")))

    def test_interrupted_import_resumes_without_overwriting(self):
        real_write = ev.write

        def interrupt(path, value):
            if path == self.out / "shadow" / "sealed.json":
                raise OSError("interrupted")
            real_write(path, value)

        with patch.object(ev, "write", side_effect=interrupt), self.assertRaisesRegex(OSError, "interrupted"):
            self.recover()
        raw = self.out / "raw" / "sealed.json"
        original_bytes = raw.read_bytes()
        self.assertEqual(self.recover(), ["sealed"])
        self.assertEqual(raw.read_bytes(), original_bytes)
        self.assertEqual(self.recover(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
