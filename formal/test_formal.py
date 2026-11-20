"""CPU regressions for the formal path, including a real random tiny-Qwen3 TCP run."""

import copy
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
import run
from run import Draft, init_head, train
from objective import fit_pca, logits, features
from raw_data import digest, partition, check_round
from verify503 import loopback
from feedback_codec.codec import CodecConfig
from feedback_codec.hf_backend import HFTargetBackend
from feedback_codec.v10_four_arm import V10FourArmHead, V10PreparedState
from h_vq64_head import MessageState
from message_codecs import METHODS, binding
from transformers import Qwen3Config, Qwen3ForCausalLM


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid input accepted")


def fixture():
    torch.manual_seed(117)
    cfg = Qwen3Config(
        vocab_size=41,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.0,
        eos_token_id=40,
        pad_token_id=0,
        bos_token_id=1,
    )
    cfg._attn_implementation = "eager"
    draft = Draft(Qwen3ForCausalLM(cfg), object(), device="cpu")
    target = HFTargetBackend(Qwen3ForCausalLM(cfg), object(), device="cpu")
    cc = CodecConfig(32, 32, 41)
    table = draft.model.get_input_embeddings()
    base = V10FourArmHead(cc, draft.model.get_output_embeddings(), mode="C", draft_embedding=table)
    init = base.trainable_state_dict()

    def data(n):
        labels = torch.randint(0, 41, (n, 8))
        labels[::2, 2:] = -100
        accepted = torch.arange(n) % 9
        return {
            "request": torch.arange(n) % 3,
            "hidden": torch.randn(n, 8, 32),
            "target": torch.randn(n, 32),
            "labels": labels,
            "accepted": accepted,
            "correction": torch.randint(41, (n,)),
            "rejected": torch.where(accepted.eq(8), -1, torch.randint(41, (n,))),
        }

    return draft, target, cc, init, {"train": data(19), "calibration": data(7)}


def scalar(head, table, b, idx):
    h = b["hidden"][idx, :1]
    n, rid, cid = (int(b[k][idx]) for k in ("accepted", "rejected", "correction"))
    r = None if rid == -1 else table.weight[rid : rid + 1]
    c = table.weight[cid : cid + 1]
    inputs = (b["target"][idx : idx + 1], n, r, c)
    state = V10PreparedState("C", (cid,), h, r, c, n, None, None, None)
    wire = head.message.encode_payload(*inputs)
    assert type(wire) is bytes and len(wire) == 8
    return head.forward_logits(
        b["hidden"][idx : idx + 1], MessageState(state, inputs, wire, (cid,)), torch.arange(1, 9)
    )


def main():
    torch.set_num_threads(1)
    draft, target, cfg, initial, data = fixture()
    table = draft.model.get_input_embeddings()
    roster = []
    for i in range(3):
        prompt = [1, 4 + i, 10]
        ids = []
        for _ in range(9):
            ids.append(target.next(prompt + ids).token_id)
            if ids[-1] == 40:
                break
        roster.append({"request_id": str(i), "prompt_token_ids": prompt, "expected_target_token_ids": ids})
    with tempfile.TemporaryDirectory() as tmp:
        for m in METHODS:
            out = Path(tmp) / m
            out.mkdir()

            def fresh():
                return init_head(draft, m, initial, cfg)

            head = fresh()
            if m == "pca_int8":
                rejects(lambda: fit_pca(head.message, data["calibration"], table, "calibration"))
                fit_pca(head.message, data["train"], table, "train", 5)
                vec = features(data["train"], table)[-1].double()
                _, _, vt = torch.linalg.svd(vec - vec.mean(0), full_matrices=False)
                vt = vt[:8]
                oriented = vt * vt.gather(1, vt.abs().argmax(1, keepdim=True)).sign()
                torch.testing.assert_close(head.message.projection, oriented.float(), atol=1e-5, rtol=1e-4)
                rejects(lambda: fit_pca(head.message, data["train"], table, "train"))
            head.eval()
            b = data["train"]
            actual = logits(head, table, b)
            for idx in (0, 8, 14):
                torch.testing.assert_close(
                    actual[idx : idx + 1], scalar(head, table, b, idx), atol=3e-6, rtol=2e-5
                )
            head = fresh()
            key = {"method": m}
            opts = dict(epochs=2, batch_size=8, n_requests=3, interval=2)
            before_message = copy.deepcopy(head.message.state_dict())
            assert not any(p.requires_grad for p in head.base.parameters())
            state = train(head, table, data, key, out, torch.device("cpu"), **opts)
            assert state["updates"] == 6 and state["tokens"] == 2 * int(b["labels"].ne(-100).sum())
            named = dict(head.base.named_parameters())
            assert all(torch.equal(named[k].cpu(), v) and named[k].grad is None for k, v in initial.items())
            assert any(not torch.equal(before_message[k], v) for k, v in head.message.named_parameters())
            assert all(p.grad is None for p in draft.model.parameters())
            restored = fresh()
            resume_dir = out / "restart"
            resume_dir.mkdir()

            def stop(*args, **kwargs):
                raise RuntimeError("intentional interruption")

            with patch.object(run, "calibrate", stop):
                try:
                    train(restored, table, data, key, resume_dir, torch.device("cpu"), **opts)
                except RuntimeError as exc:
                    assert str(exc) == "intentional interruption"
                else:
                    raise AssertionError("interruption did not occur")
            assert not (resume_dir / "selected.json").exists()
            train(restored, table, data, key, resume_dir, torch.device("cpu"), **opts)
            assert all(
                torch.equal(v, restored.trainable_state_dict()[k])
                for k, v in head.trainable_state_dict().items()
            )
            rejects(
                lambda: train(
                    restored, table, data, {"method": "wrong"}, resume_dir, torch.device("cpu"), **opts
                )
            )
            contract = {"max_new": 9, "window": 8, "eos_token_id": 40, "codec_binding": binding(head.message)}
            totals = loopback(
                roster, target, draft, head.requires_grad_(False).eval(), torch.device("cpu"), out, contract
            )
            assert totals["raw"]["accepted"] == totals["shadow"]["accepted"]
            print(m, "passed batch/single-wire logits, training/resume and 3-request x 3-arm tiny TCP")
    from existing import check_schema, load_exact, base_only

    check_schema("H-VQ64-TASK", "grouped_vq8")
    rejects(lambda: check_schema("H-PCAQ64-W", "pca_int8"))
    rejects(lambda: check_schema("H-VQ64-TASK", "mlp_int8"))
    import inspect

    rejects(lambda: base_only(Path(inspect.getfile(base_only))))
    model = run.HTaskHead(
        cfg, draft.model.get_output_embeddings(), draft_embedding=table, codec_type="grouped_vq8"
    )
    full = model.trainable_state_dict()
    load_exact(model, full)
    rejects(lambda: load_exact(model, initial))
    k = next(iter(full))
    rejects(lambda: load_exact(model, {**full, k: full[k].half()}))
    print("strict full state, base-only, foreign schema, source digest and dtype tests passed")
    # A nonterminal decision with a tied logit still uses the archived greedy token.
    h = {
        "same_effective_target_forward_as_decision": True,
        "available_before_receipt": True,
        "layer": "model.norm",
        "present": True,
        "absolute_position": 0,
        "source_prefix_sha256": digest([1]),
        "predicted_token_id": 448,
    }
    w = dict(
        round=0,
        offset=1,
        labels=[3],
        length=1,
        observation_key="r:0",
        hidden_key="r0.hidden",
        correction=448,
        accepted=0,
        rejected=714,
    )
    r = dict(prompt_ids=[1], output_ids=[448, 3], windows=[w], split="train", family_id="fam", request_id="x")
    rr = dict(
        hidden=h,
        observation_key="r:0",
        accepted_count=0,
        candidate_ids=[714],
        committed_ids=[448],
        prefix_after_commit_length=2,
        stop_reason="continue",
        tensor_prefix="r0",
        correction_id=448,
        bonus_id=None,
        rejected_id=714,
        top32={"first": 714, "scores": [27.625, 27.625]},
    )
    check_round(r, w, rr)
    for change in (
        dict(hidden={**h, "absolute_position": 1}),
        dict(hidden={**h, "predicted_token_id": 714}),
        dict(stop_reason="eos"),
        dict(committed_ids=[714]),
    ):
        rejects(lambda: check_round(r, w, {**rr, **change}))
    bad = copy.deepcopy(r)
    bad.update(split="calibration", request_id="y")
    rejects(lambda: partition([r, bad], [], {"train": (1, 1), "calibration": (1, 1)}))
    print("causal offset, selected greedy, terminal and family-leakage negatives passed")


if __name__ == "__main__":
    main()
