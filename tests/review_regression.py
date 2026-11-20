"""Local regression reproductions against the imported demo directory."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cpu_parity as f


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.draft = f.make(121, f.torch.float32, "kv")
        self.prompt = [1, 2, 3]
        self.draft.boundary(self.prompt)

    def test_incomplete_cache_hidden_hit(self):
        self.draft.cache.crop(2)
        with self.assertRaisesRegex(ValueError, "cache length"):
            self.draft.boundary(self.prompt)

    def test_incomplete_layer_terminal_commit(self):
        self.draft.boundary(self.prompt + [7, 8])
        self.draft.cache.layers[-1].crop(2)
        with self.assertRaisesRegex(ValueError, "cache length"):
            self.draft.commit(self.prompt + [4], {"accepted": 0, "answer": 4})
        self.assertEqual(self.draft.committed, tuple(self.prompt))
        self.draft.reset(f.smoke.Meter("cpu"))
        recovered = self.draft.boundary([5, 6])
        expected = self.draft.backend.teacher_hidden([5, 6], [0])
        f.torch.testing.assert_close(recovered, expected)

    def test_only_rejected_suffix_cropped_on_terminal(self):
        self.draft.boundary(self.prompt + [7, 8])
        calls = self.draft.stats["decoder_calls"]
        self.draft.commit(self.prompt + [7, 9], {"accepted": 1, "answer": 9, "rejected": 8})
        self.assertEqual(self.draft.prefix, tuple(self.prompt + [7]))
        self.assertEqual([layer.get_seq_length() for layer in self.draft.cache.layers], [4, 4])
        self.assertEqual(self.draft.stats["decoder_calls"], calls)
        self.assertEqual(self.draft.snapshot()["pending_committed_tokens"], 1)

    def test_wrong_accepted_token(self):
        self.draft.boundary(self.prompt + [7, 8])
        with self.assertRaises(ValueError):
            self.draft.commit(self.prompt + [6, 9], {"accepted": 1, "answer": 9})
        self.assertEqual(self.draft.prefix, tuple(self.prompt + [7, 8]))

    def test_control_failure_and_retry(self):
        confirmed = self.prompt + [4]
        reply = {"accepted": 0, "answer": 4, "rejected": 3}
        self.draft.commit(confirmed, reply)
        with patch.object(self.draft, "build_control", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                self.draft.prepare(confirmed, reply, bytes(8))
        self.assertIsNone(self.draft.state)
        calls = self.draft.stats["decoder_calls"]
        self.draft.prepare(confirmed, reply, bytes(8))
        self.assertEqual(self.draft.stats["decoder_calls"], calls)
        self.assertIsNotNone(self.draft.control)

    def test_commit_rejects_wrong_answer_without_mutation(self):
        with self.assertRaises(ValueError):
            self.draft.commit(self.prompt + [4], {"accepted": 0, "answer": 5})
        self.assertEqual(self.draft.committed, tuple(self.prompt))

    def test_prepare_requires_committed_boundary(self):
        with self.assertRaises(ValueError):
            self.draft.prepare(self.prompt + [4], {"accepted": 0, "answer": 4, "rejected": 3}, bytes(8))
        self.assertIsNone(self.draft.state)

    def test_uncached_decode_count(self):
        self.draft.cache_control = False
        reply = {"accepted": 0, "answer": 4, "rejected": 3}
        confirmed = self.prompt + [4]
        self.draft.commit(confirmed, reply)
        self.draft.prepare(confirmed, reply, bytes(8))
        self.draft.token(confirmed, 1, "active")
        self.assertEqual(self.draft.meter.calls["edge_codebook_decode"], 1)
        self.assertEqual(self.draft.stats["payload_decodes"], 1)


class ReportTests(unittest.TestCase):
    def summarize(self, single=False, mixed=False, wrong_output=False, mixed_contracts=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            length = 1 if single else 2
            for arm in f.smoke.ARMS:
                ids = [1] * length
                if wrong_output and arm == "active":
                    ids[0] = 2
                trial = "repeat0-requestA-" + arm
                row = dict(trial_id=trial, request_id="A", arm=arm, warmup=False, output_ids=ids,
                           codec_binding=dict(codec_type="grouped_vq8", bits=64, schema_sha256="1"*64, state_sha256="2"*64),
                           contract_id="b" if mixed_contracts and arm == "active" else "a",
                           rounds=[dict(candidate_ids=[1], committed_ids=ids, accepted=1, feedback_bytes=0)],
                           timing=dict(calls={"target_verify": length}, host_wall_seconds={}, gpu_event_seconds={}),
                           wall_seconds=1., ttft_seconds=.1, tpot_seconds=None if single else .9,
                           accepted_draft_tokens=1, verification_rounds=1, framed_application_bytes_up=16,
                           framed_application_bytes_down=16)
                f.smoke.write(root / "cloud/rows" / (trial + ".json"), row)
                row["draft_implementation"] = "reference" if mixed and arm == "raw" else "kv"
                f.smoke.write(root / "edge/rows" / (trial + ".json"), row)
            out = root / "summary.json"
            f.smoke.summarize(root / "edge", root / "cloud", out)
            return json.loads(out.read_text())

    def test_single_token_summary(self):
        report = self.summarize(single=True)
        for arm in f.smoke.ARMS:
            self.assertIsNone(report["arms"][arm]["p50_tpot_seconds"])
            self.assertIsNone(report["arms"][arm]["stages"]["target_verify"]["mean_gpu_event_ms"])

    def test_mixed_implementations(self):
        with self.assertRaisesRegex(ValueError, "implementation"):
            self.summarize(mixed=True)

    def test_unmatched_target_outputs(self):
        with self.assertRaisesRegex(ValueError, "output"):
            self.summarize(wrong_output=True)

    def test_same_mode_different_contract(self):
        with self.assertRaisesRegex(ValueError, "contract"):
            self.summarize(mixed_contracts=True)


if __name__ == "__main__":
    f.torch.set_num_threads(1)
    unittest.main(verbosity=2)
