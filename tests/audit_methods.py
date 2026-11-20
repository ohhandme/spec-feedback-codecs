"""Verify cross-implementation trajectories, negative aggregation, and frozen VQ behavior."""
import argparse
import contextlib
import functools
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import cpu_parity as fixture
from train_codecs import METHODS, ROOT, binding, make_codec, rejects, torch


def read_rows(root):
    return {r["trial_id"]: r for path in (root / "edge/rows").glob("*.json")
            for r in [json.loads(path.read_bytes())]}


def trajectory(row):
    return (row["output_ids"], [(r["candidate_ids"], r["accepted"], r["committed_ids"], r["feedback_bytes"])
                                for r in row["rounds"]])


def negative_summary(root):
    source = root / "pca_int8-kv"
    trial = next(r for r in read_rows(source).values() if r["arm"] == "active")
    result = []
    for key, value in [("codec_type", "grouped_vq8"), ("state_sha256", "a" * 64), ("bits", 32)]:
        with tempfile.TemporaryDirectory() as tmp:
            edges = Path(tmp) / "edge/rows"
            edges.mkdir(parents=True)
            for p in (source / "edge/rows").glob("*.json"):
                row = json.loads(p.read_bytes())
                if row["trial_id"] == trial["trial_id"]:
                    row["codec_binding"][key] = value
                (edges / p.name).write_text(json.dumps(row))
            try:
                fixture.smoke.summarize(edges.parent, source / "cloud", Path(tmp) / "out.json")
            except ValueError as e:
                assert "mixed codec" in str(e)
            else:
                raise AssertionError("mixed codec accepted")
            result.append(key)
    return result


def old_vq():
    src = ROOT / "tests/baseline_vq.py"
    import hashlib
    assert hashlib.sha256(src.read_bytes()).hexdigest() == "d7a70352e64330dfca37c5075ff015ddbc865f7a106eb03eb36770d80981e726"
    spec = importlib.util.spec_from_file_location("reviewed_vq", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old = mod.ConditionedVQ64(target_hidden_size=32, draft_hidden_size=32)
    new = make_codec("grouped_vq8", 32, 32)
    new.load_state_dict(old.state_dict())
    for accepted in range(9):
        x = (torch.randn(1, 32), accepted, torch.randn(1, 32) if accepted < 8 else None, torch.randn(1, 32))
        a, b = old.encode_payload(*x), new.encode_payload(*x)
        assert a == b and torch.equal(old.decode_payload(a), new.decode_payload(b))
        assert torch.equal(old.train_control(*x), new.train_control(*x))
    return 9


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    root = parser.parse_args().out
    torch.set_num_threads(1)
    report = {"reviewed_vq_wire_control_cases": old_vq(), "mixed_summary_rejected": negative_summary(root),
              "methods": {}}
    original = read_rows(ROOT / "evidence/tcp-kv")
    rerun = read_rows(root / "legacy")
    assert len(original) == len(rerun) == 9
    assert all(trajectory(original[k]) == trajectory(rerun[k]) for k in original)
    report["legacy_exact_tcp_trajectories"] = len(original)
    for method in METHODS:
        reference, cached = [read_rows(root / (method + "-" + mode)) for mode in ["reference", "kv"]]
        assert len(reference) == len(cached) == 9
        for key, r in reference.items():
            assert trajectory(r) == trajectory(cached[key])
            assert r["feedback_hex"] == cached[key]["feedback_hex"]
        data = torch.load(root / (method + ".pt"), weights_only=True)
        with patch.object(fixture, "make", functools.partial(fixture.make, codec_checkpoint=data)):
            branches = fixture.forced_branches(121, torch.float32, "kv")["forced_rejection_and_bonus"]
            terminal = fixture.integrated_terminal_gate(torch.float32)
        report["methods"][method] = {"tcp_reference_kv_identical": 9, "same_wire_per_round": True,
                                     "forced_accept_counts": [r["accepted"] for r in branches],
                                     "terminal_cases": len(terminal), "tested_cross_request_reset": True}
    (root / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
