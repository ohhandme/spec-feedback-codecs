"""Review regression check for all-accepted truncation without a bonus token."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

import cpu_parity as fixture


def main():
    fixture.torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=root.parent / "evidence" / "independent_terminal_results.json")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("review_terminal_smoke", fixture.REFERENCE / "latency_smoke.py")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    results = []
    for dtype in (fixture.torch.float32, fixture.torch.bfloat16):
        for mode in ("boundary", "kv"):
            for arm in ("raw", "shadow"):
                for limit in (1, 8, 9, 16):
                    draft = fixture.make(172, dtype, mode)
                    target = fixture.HFTargetBackend(
                        copy.deepcopy(draft.backend.model), object(), device="cpu", dtype=dtype
                    )
                    prompt = [1, 5, 7]
                    expected = []
                    for _ in range(limit):
                        token = target.next(prompt + expected).token_id
                        expected.append(token)
                        if token == 96:
                            break
                    assert len(expected) == limit, "fixture unexpectedly stopped at EOS"
                    row = {"request_id": "review", "prompt_token_ids": prompt,
                           "expected_target_token_ids": expected}
                    contract = {"window": 8, "max_new": limit, "eos_token_id": 96}

                    def encode(hidden, accepted, rejected, answer):
                        table = draft.backend.model.get_input_embeddings().weight
                        return draft.head.message.encode_payload(
                            hidden.reshape(1, -1), accepted,
                            None if rejected is None else table[rejected].reshape(1, -1),
                            table[answer].reshape(1, -1),
                        )

                    session = smoke.CloudSession(target, encode, {"review": row}, contract, "cpu")

                    class Wire:
                        up = down = 0

                        def rpc(self, request):
                            if request["type"] == "start":
                                reply, payload = session.begin(request)
                            elif request["type"] == "verify":
                                reply, payload = session.verify(request)
                            elif request["type"] == "finish_request":
                                session.finish_request()
                                reply, payload = {"type": "finished"}, b""
                            else:
                                raise AssertionError(request)
                            return reply, payload, 0.0

                    record = smoke.edge_request(Wire(), draft, row, arm, "review", False, contract)
                    assert record["output_ids"] == expected
                    assert all(r["accepted"] == len(r["candidate_ids"]) for r in record["rounds"])
                    terminal = record["rounds"][-1]
                    has_bonus = len(terminal["committed_ids"]) == terminal["accepted"] + 1
                    assert has_bonus == (limit == 9), "unexpected terminal bonus boundary"
                    assert "draft_cache_after_prepare" not in terminal
                    state = terminal["draft_cache_after_commit"]
                    assert state["cache_is_prefix_of_committed"]
                    assert state["pending_committed_tokens"] == 1 + int(has_bonus)
                    assert draft.stats["feedback_prepares"] == (
                        record["verification_rounds"] - 1 if arm == "shadow" else 0
                    )
                    results.append({"dtype": str(dtype), "mode": mode, "arm": arm,
                                    "max_new": limit, "rounds": record["verification_rounds"],
                                    "terminal": state})
    args.out.write_text(
        json.dumps({"status": "pass", "cases": results}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "pass", "all_accepted_truncation_cases": len(results)}))


if __name__ == "__main__":
    main()
