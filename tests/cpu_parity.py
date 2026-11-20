"""CPU correctness gates on actual tiny Qwen3, not GPU or network speed evidence."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import importlib.util

HERE = Path(__file__).resolve().parent
REFERENCE = Path(os.environ.get("KV_REVIEW_DEMO", HERE.parent / "demo")).resolve()
sys.path[:0] = [str(HERE), str(REFERENCE), str(REFERENCE / "vendor")]

import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM
import latency_smoke as smoke
from cached_draft import CachedDraft
from feedback_codec.hf_backend import HFTargetBackend, HFDraftBackend, _FrozenHFBackend
from feedback_codec.codec import CodecConfig
from h_vq64_head import HTaskHead


class DraftBackend(_FrozenHFBackend):
    teacher_hidden = HFDraftBackend.teacher_hidden


def make(seed, dtype, variant, *, codec_checkpoint=None):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=8, max_position_embeddings=256, attention_dropout=0.,
                     pad_token_id=0, bos_token_id=1, eos_token_id=96)
    cfg._attn_implementation = "eager"
    backend = DraftBackend(Qwen3ForCausalLM(cfg), object(), device="cpu", dtype=dtype)
    head = HTaskHead(CodecConfig(32, 32, 97), backend.model.get_output_embeddings(),
                     draft_embedding=backend.model.get_input_embeddings(),
                     codec_type=codec_checkpoint["binding"]["codec_type"] if codec_checkpoint else "grouped_vq8").eval()
    if codec_checkpoint:
        from message_codecs import restore_codec
        head.message = restore_codec(codec_checkpoint["binding"], codec_checkpoint["message"]).eval()
    # Exercise a nonzero trained-style residual; the default zero U is insufficient.
    with torch.no_grad():
        head.base.core.repair.U.weight.normal_(0, .08)
    head.requires_grad_(False)
    instance = (smoke.NativeDraft(backend, head) if variant == "reference"
                else CachedDraft(backend, head, mode=variant))
    instance.reset(smoke.Meter("cpu"))
    return instance


def difference(a, b, dtype):
    assert a.shape == b.shape
    tolerance = {torch.float32: (1e-5, 2e-6), torch.bfloat16: (.02, .02)}[dtype]
    return {"bitwise_equal": bool(torch.equal(a, b)),
            "max_abs_error": float((a.float() - b.float()).abs().max()),
            "within_locked_tolerance": bool(torch.allclose(a.float(), b.float(),
                                                           rtol=tolerance[0], atol=tolerance[1]))}


def inspect_boundary(reference, optimized, prefix, dtype, checks):
    h0 = reference.backend.teacher_hidden(prefix, [0])
    h1 = optimized.boundary(prefix)
    # Compare against the native full-model logits, not just another row projection.
    output, _ = reference.backend._forward_with_final_hidden(reference.backend._ids(prefix))
    l0 = output.logits[:, -1]
    l1 = optimized.backend.model.get_output_embeddings()(h1)
    hidden, logits = difference(h0, h1, dtype), difference(l0, l1, dtype)
    assert hidden["within_locked_tolerance"] and logits["within_locked_tolerance"]
    assert int(l0.argmax()) == int(l1.argmax())
    checks.append({"prefix": prefix, "hidden": hidden, "native_logits": logits,
                   "native_greedy_equal": True})


def forced_branches(seed, dtype, variant):
    reference, optimized = make(seed, dtype, "reference"), make(seed, dtype, variant)
    prefix, checks, branches = [1, 2, 3], [], []
    for accepted in (0, 1, 7, 8, 3):
        proposal = []
        for position in range(1, 9):
            history = prefix + proposal
            inspect_boundary(reference, optimized, history, dtype, checks)
            a = reference.token(history, position, "active")
            b = optimized.token(history, position, "active")
            assert a == b, "candidate changed under cache"
            proposal.append(a)
        rejected = proposal[accepted] if accepted < 8 else None
        correction = ((rejected + 1) % 97) if rejected is not None else 29
        confirmed = prefix + proposal[:accepted] + [correction]
        reply = {"accepted": accepted, "answer": correction, "rejected": rejected}
        payload = bytes(range(8))
        before = optimized.stats.copy()
        optimized.commit(confirmed, reply)
        after_commit = optimized.snapshot()
        assert after_commit["cache_is_prefix_of_committed"]
        reference.prepare(confirmed, reply, payload)
        optimized.prepare(confirmed, reply, payload)
        delta = optimized.stats - before
        assert optimized.prefix == tuple(confirmed)
        if variant == "kv":
            # Last proposed token has not yet been processed when the whole window is accepted.
            assert delta["decoder_input_tokens"] == (2 if accepted == 8 else 1)
        h = optimized.boundary(confirmed)
        h0 = reference.backend.teacher_hidden(confirmed, [0])
        base_diff = difference(reference.state.base.local_hidden, optimized.state.base.local_hidden, dtype)
        assert base_diff["within_locked_tolerance"]
        repairs = []
        for position in range(1, 9):
            # Isolate control caching: identical hidden/state must give bitwise identical logits.
            original = optimized.head.forward_logits(h, optimized.state, position)
            cached = optimized.repaired_logits(h, position)
            assert torch.equal(original, cached), "per-round control changed repair arithmetic"
            original_full = reference.head.forward_logits(h0, reference.state, position)
            repair_diff = difference(original_full, cached, dtype)
            assert repair_diff["within_locked_tolerance"]
            assert int(original_full.argmax()) == int(cached.argmax())
            repairs.append(repair_diff)
        decoder_calls = optimized.stats["decoder_calls"]
        assert reference.token(confirmed, 1, "active") == optimized.token(confirmed, 1, "active")
        assert optimized.stats["decoder_calls"] == decoder_calls, "first candidate repeated the boundary decoder"
        branches.append({"accepted": accepted, "suffix_update_tokens": delta["decoder_input_tokens"],
                         "cache_after_commit": after_commit, "cache_after_prepare": optimized.snapshot(),
                         "prepare_hidden": base_diff, "repaired_logits_by_position": repairs,
                         "same_boundary_first_token_no_decoder": True})
        prefix = confirmed
    # Cross-request and future-boundary reuse must fail, rather than silently reuse stale state.
    try:
        optimized.boundary([1, 4])
    except ValueError:
        pass
    else:
        raise AssertionError("changed prompt reused old cache")
    optimized.reset(smoke.Meter("cpu"))
    assert optimized.cache is None and optimized.state is None and optimized.control is None
    assert not optimized.stats and optimized.prompt is None
    return {"seed": seed, "dtype": str(dtype), "variant": variant,
            "same_prefix_checks": checks, "forced_rejection_and_bonus": branches,
            "changed_prompt_rejected_and_reset_cleared": True}


def actual_target_rounds(seed, dtype, variant, arm):
    draft = make(seed, dtype, variant)
    torch.manual_seed(seed + 1000)
    target = HFTargetBackend(Qwen3ForCausalLM(draft.backend.model.config), object(),
                             device="cpu", dtype=dtype)
    prompt, expected = [1, 5, 7], []
    contract = {"window": 8, "max_new": 18, "eos_token_id": 96}
    for _ in range(contract["max_new"]):
        nxt = int(target.next(prompt + expected).token_id)
        expected.append(nxt)
        if nxt == 96:
            break
    row = {"request_id": "tiny", "prompt_token_ids": prompt, "expected_target_token_ids": expected}
    def encode(hidden, accepted, rejected, answer):
        table = draft.backend.model.get_input_embeddings().weight
        return draft.head.message.encode_payload(hidden.reshape(1, -1), accepted,
            None if rejected is None else table[rejected].reshape(1, -1), table[answer].reshape(1, -1))
    session = smoke.CloudSession(target, encode, {"tiny": row}, contract, "cpu")
    session.begin({"arm": arm, "request_id": "tiny", "trial_id": variant + arm,
                   "warmup": False, "prompt_ids": prompt})
    confirmed, generated, rounds, payloads, boundaries = prompt.copy(), [], [], [], []
    while len(generated) < contract["max_new"]:
        candidate = []
        for position in range(1, min(8, contract["max_new"] - len(generated)) + 1):
            candidate.append(draft.token(confirmed + candidate, position, arm))
            if candidate[-1] == 96:
                break
        reply, payload = session.verify({"round": len(rounds), "prefix_sha256": smoke.digest(confirmed),
                                         "proposals": candidate})
        committed = candidate[:reply["accepted"]] + ([] if reply["answer"] is None else [reply["answer"]])
        confirmed.extend(committed); generated.extend(committed)
        if hasattr(draft, "commit"):
            draft.commit(confirmed, reply)
            boundary = draft.snapshot()
            assert boundary["cache_is_prefix_of_committed"]
            if variant == "kv":
                assert boundary["kv_length"] == boundary["decoder_consumed_length"]
                assert boundary["pending_committed_tokens"] in (0, 1, 2)
            boundaries.append({"done": reply["done"], **boundary})
        rounds.append({"candidate_ids": candidate, "accepted": reply["accepted"],
                       "committed_ids": committed, "done": reply["done"]})
        payloads.append(payload.hex())
        if reply["done"]:
            break
        if arm != "raw":
            draft.prepare(confirmed, reply, payload)
    assert generated == expected
    session.finish_request()
    timing = {"calls": dict(draft.meter.calls)}  # No CPU wall time is promoted to a speed result.
    counters = dict(draft.stats) if hasattr(draft, "stats") else None
    if counters is not None:
        assert counters.get("prompt_prefills", 0) == (1 if variant == "kv" else 0)
        assert counters.get("control_builds", 0) == (0 if arm == "raw" else len(rounds) - 1)
        assert counters.get("payload_decodes", 0) == counters.get("control_builds", 0)
    return {"rounds": rounds, "output": generated, "feedback_payloads": payloads,
            "timing_call_counts_only": timing, "draft_cache": counters, "committed_vs_consumed": boundaries}


def integrated_terminal_gate(dtype):
    """Exercise the actual proposed edge_request changes, including terminal crop."""
    spec = importlib.util.spec_from_file_location("task64_patched_smoke", REFERENCE / "latency_smoke.py")
    patched = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patched)
    results = []
    for terminal in ("max_new", "eos_correction", "eos_accepted"):
        for arm in smoke.ARMS:
            draft = make(172, dtype, "kv")
            torch.manual_seed(1172)
            target = HFTargetBackend(Qwen3ForCausalLM(draft.backend.model.config), object(), device="cpu", dtype=dtype)
            prompt = [1, 5, 7]
            if terminal.startswith("eos"):
                with torch.inference_mode():
                    hidden = target.model.model(input_ids=target._ids(prompt), use_cache=False).last_hidden_state[:, -1]
                    table = target.model.get_output_embeddings().weight
                    table.zero_(); table[96].copy_(hidden[0])
                    assert target.next(prompt).token_id == 96
                    if terminal == "eos_accepted":
                        hidden = draft.backend.teacher_hidden(prompt, [0])
                        table = draft.backend.model.get_output_embeddings().weight
                        table.zero_(); table[96].copy_(hidden[0])
            contract = {"window": 8, "max_new": 18, "eos_token_id": 96}
            expected = []
            for _ in range(18):
                token = target.next(prompt + expected).token_id
                expected.append(token)
                if token == 96:
                    break
            row = {"request_id": "integrated", "prompt_token_ids": prompt, "expected_target_token_ids": expected}
            def encode(hidden, accepted, rejected, answer):
                table = draft.backend.model.get_input_embeddings().weight
                return draft.head.message.encode_payload(hidden.reshape(1, -1), accepted,
                    None if rejected is None else table[rejected].reshape(1, -1), table[answer].reshape(1, -1))
            session = patched.CloudSession(target, encode, {row["request_id"]: row}, contract, "cpu")
            class DirectWire:
                up = down = 0
                def rpc(self, request):
                    if request["type"] == "start":
                        reply, payload = session.begin(request)
                    elif request["type"] == "verify":
                        reply, payload = session.verify(request)
                    elif request["type"] == "finish_request":
                        session.finish_request(); reply, payload = {"type": "finished"}, b""
                    else:
                        raise AssertionError(request)
                    return reply, payload, 0.
            record = patched.edge_request(DirectWire(), draft, row, arm, terminal + arm, False, contract)
            last = record["rounds"][-1]
            assert record["output_ids"] == expected
            assert last["draft_cache_after_commit"]["cache_is_prefix_of_committed"]
            assert "draft_cache_after_prepare" not in last, "terminal round ran feedback prepare"
            assert record["draft_cache_counters"]["verification_commits"] == record["verification_rounds"]
            if terminal.startswith("eos"):
                assert record["output_ids"] == [96] and len(record["rounds"]) == 1
                assert draft.stats["decoder_calls"] == len(last["candidate_ids"]), \
                    "terminal EOS caused a decoder forward after proposing the window"
                assert not draft.stats["feedback_prepares"]
                assert last["accepted"] == (1 if terminal == "eos_accepted" else 0)
            results.append({"dtype": str(dtype), "terminal": terminal, "arm": arm,
                            "rounds": len(record["rounds"]), "final_cache": last["draft_cache_after_commit"],
                            "decoder_calls": draft.stats["decoder_calls"], "terminal_no_prepare": True,
                            "target_output_exact": True})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "cpu_parity.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    branches, traces, terminals = [], [], []
    for dtype in (torch.float32, torch.bfloat16):
        for seed in (121, 172, 321):
            for variant in ("boundary", "kv"):
                branches.append(forced_branches(seed, dtype, variant))
            runs = {}
            for variant in ("reference", "boundary", "kv"):
                runs[variant] = {arm: actual_target_rounds(seed, dtype, variant, arm) for arm in smoke.ARMS}
            for arm in smoke.ARMS:
                for variant in ("boundary", "kv"):
                    for key in ("rounds", "output", "feedback_payloads"):
                        assert runs[variant][arm][key] == runs["reference"][arm][key], (seed, dtype, variant, arm, key)
            for variant in runs:
                assert runs[variant]["raw"]["rounds"] == runs[variant]["shadow"]["rounds"]
            traces.append({"seed": seed, "dtype": str(dtype), "runs": runs})
        terminals.extend(integrated_terminal_gate(dtype))
    result = {"status": "pass", "scope": "CPU tiny dense Qwen3 only; not production BF16 or two-host timing",
              "torch": torch.__version__, "transformers": transformers.__version__,
              "tolerances_locked_before_run": {"float32": {"rtol": 1e-5, "atol": 2e-6},
                                                "bfloat16": {"rtol": .02, "atol": .02}},
              "control_cache_same_input_requires_bitwise_logits": True,
              "native_and_repaired_greedy_and_target_decisions_require_exact_match": True,
              "branches": branches, "actual_target_closed_loops": traces, "integrated_terminal_gates": terminals,
              "source_sha256": {p.name: sha256(p.read_bytes()).hexdigest()
                                for p in (REFERENCE / "cached_draft.py", Path(__file__))}}
    smoke.write(args.out, result)
    print(json.dumps({"status": "pass", "branch_suites": len(branches),
                      "closed_loop_suites": len(traces), "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
