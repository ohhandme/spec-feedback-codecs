import argparse
import tempfile
from pathlib import Path

import evaluate as ev
import torch
from test_formal import fixture, rejects
from raw_data import read, write, sha, digest
from message_codecs import binding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roster", type=Path)
    args = parser.parse_args()
    rows = ev.cohort(args.roster)
    assert len(rows) == 200 and ev.parent.code_binding() == ev.TRAIN_CODE
    torch.set_num_threads(1)
    draft, target, cfg, initial, _ = fixture()
    head = ev.parent.init_head(draft, "mlp_int8", initial, cfg)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        # Native dc6 checkpoint serialization; only terminal counters here are fixture metadata.
        key = {"code": ev.parent.code_binding(), "method": "mlp_int8"}
        ck = root / "epoch1.pt"
        ev.parent.save_pt(
            ck,
            {
                "model": head.trainable_state_dict(),
                "epoch": 1,
                "key": key,
                "codec_binding": binding(head.message),
            },
        )
        hist = [
            {"epoch": i, "request_equal_ce": float(i), "sha256": sha(ck) if i == 1 else "unused"}
            for i in range(1, 6)
        ]
        write(root / "history.json", hist)
        write(
            root / "selected.json", {**hist[0], "epochs": 5, "updates": 24485, "tokens": 6087440, "key": key}
        )
        with torch.no_grad():
            next(head.message.parameters()).add_(1)
        assert ev.load_selected(root, key, head) == sha(ck)
        rejects(lambda: ev.load_selected(root, dict(key, code="different"), head))
        prompt = [1, 4, 10]
        output = []
        for _ in range(9):
            output.append(target.next(prompt + output).token_id)
            if output[-1] == 40:
                break
        row = {"request_id": "fixture", "prompt_token_ids": prompt, "expected_target_token_ids": output}
        orig = {"max_new": 9, "window": 8, "eos_token_id": 40, "codec_binding": binding(head.message)}
        child = {**orig, "scope": "eval200", "ordered_ids_sha256": ev.IDS_SHA}
        src = root / "source"
        src.mkdir()
        ev.loopback([row], target, draft, head.requires_grad_(False).eval(), torch.device("cpu"), src, orig)
        write(src / "contract.json", orig)
        write(src / "completion.json", {"requests": 503, "contract": digest(orig)})
        dest = root / "dest"
        dest.mkdir()
        ev.recover_rows(dest, [row], child, orig, src)
        ev.recover_rows(dest, [row], child, orig)
        for arm in ev.ARMS:
            result = read(dest / arm / "fixture.json")
            assert result["source_row_sha256"] == sha(src / arm / "fixture.json")
            assert result["source_contract_id"] == digest(orig)
        rejects(lambda: ev.recover_rows(dest, [row], {**child, "ordered_ids_sha256": "other"}, orig))
        rejects(lambda: ev.recover_rows(dest, [row], child, {**orig, "window": 7}, src))
        extra = dest / "raw" / "foreign.json"
        write(extra, {})
        rejects(lambda: ev.recover_rows(dest, [row], child, orig))
        # A different equal-score proposal must still be rejected if the actual teacher selected 448.
        tied = {"request_id": "tie", "expected_target_token_ids": [448]}
        shot = {**orig, "max_new": 1}
        good = {
            "request_id": "tie",
            "trial_id": "tie-raw",
            "warmup": False,
            "arm": "raw",
            "contract_id": digest(shot),
            "codec_binding": shot["codec_binding"],
            "output_ids": [448],
            "rounds": [{"candidate_ids": [714], "committed_ids": [448], "accepted": 0, "feedback_bytes": 0}],
            "verification_rounds": 1,
            "accepted_draft_tokens": 0,
        }
        ev.check_trace(good, tied, "raw", shot)
        rejects(
            lambda: ev.check_trace(
                {**good, "accepted_draft_tokens": 1, "rounds": [{**good["rounds"][0], "accepted": 1}]},
                tied,
                "raw",
                shot,
            )
        )
    print(
        "PASS: pinned first 200; dc6 binding and checkpoint load; tiny TCP raw/shadow/active; matched archive import, resume, foreign cohort/checkpoint/ID and tie rejection"
    )
    print("ordered_ids_sha256", ev.IDS_SHA)


if __name__ == "__main__":
    main()
