"""Train only the codec on causal tiny Qwen3 next-token labels and exercise the wire contract."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "demo"), str(ROOT / "demo/vendor")]
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM
from feedback_codec.codec import CodecConfig
from feedback_codec.fixed_probe import ProtocolView
from feedback_codec.hf_backend import HFDraftBackend, HFTargetBackend, _FrozenHFBackend
from h_vq64_head import HTaskHead, MessageState
from message_codecs import METHODS, binding, make_codec, restore_codec


class Draft(_FrozenHFBackend):
    teacher_hidden = HFDraftBackend.teacher_hidden


def setup(method):
    torch.manual_seed(121)
    cfg = Qwen3Config(vocab_size=41, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                      attention_dropout=0., pad_token_id=0, bos_token_id=1, eos_token_id=40)
    cfg._attn_implementation = "eager"
    target = HFTargetBackend(Qwen3ForCausalLM(cfg), object(), device="cpu")
    draft = Draft(Qwen3ForCausalLM(cfg), object(), device="cpu")
    head = HTaskHead(CodecConfig(32, 32, 41), draft.model.get_output_embeddings(),
                     draft_embedding=draft.model.get_input_embeddings(), codec_type=method)
    with torch.no_grad():
        head.base.core.repair.U.weight.normal_(0, .08, generator=torch.Generator().manual_seed(712))
    head.requires_grad_(False)
    head.message.requires_grad_(True)
    head.train()
    head.base.eval()
    return target, draft, head


def dataset(target, draft, head, split, prompts):
    rows = []
    table = draft.model.get_input_embeddings().weight
    with torch.no_grad():
        for index in prompts:
            prompt = [1, 8 + index // 24, 10 + index % 24]
            step = target.next(prompt)
            if step.token_id == 40:
                continue
            rejected = int(draft.next(prompt).token_id) if hasattr(draft, "next") else int(
                draft._forward_with_final_hidden(draft._ids(prompt))[0].logits[0, -1].argmax())
            if rejected == step.token_id:
                continue
            confirmed = prompt + [step.token_id]
            protocol = ProtocolView(tuple(confirmed), 8, 0, rejected, step.token_id)
            base = head.base.prepare_view(protocol, draft)
            inputs = (step.hidden.reshape(1, -1).clone(), 0, table[rejected].reshape(1, -1).clone(),
                      table[step.token_id].reshape(1, -1).clone())
            label = target.next(confirmed).token_id
            rows.append((split, str(prompt), inputs, MessageState(base, inputs, None, tuple(confirmed)),
                         draft.teacher_hidden(confirmed, [0]).clone(), torch.tensor([label])))
    return rows


def rejects(fn):
    try:
        fn()
    except (ValueError, TypeError, FloatingPointError, RuntimeError):
        return
    raise AssertionError("invalid input accepted")


def wire_checks(codec, inputs):
    results = {}
    for dtype in [torch.float32, torch.bfloat16]:
        m = copy.deepcopy(codec).to(dtype)
        x = tuple(v.to(dtype) if torch.is_tensor(v) else v for v in inputs)
        b = m.encode_payload(*x)
        control = m.decode_payload(b)
        error = (control - m.train_control(*x)).abs().max().item()
        assert error == 0 and len(b) == 8 and type(b) is bytes and b == m.encode_payload(*x)
        identity = binding(m)
        other = restore_codec(identity, copy.deepcopy(m.state_dict()))
        assert other.encode_payload(*x) == b and torch.equal(other.decode_payload(b), control)
        assert torch.equal(m.decode_payload(None), torch.zeros_like(control))
        for bad in [b"0", bytes(9), bytearray(b), 0]:
            rejects(lambda: m.decode_payload(bad))
        for val in [float("inf"), float("nan")]:
            rejects(lambda: m.encode_payload(torch.full_like(x[0], val), *x[1:]))
        for accepted, rejected in [(True, x[2]), (9, None), (0, None), (8, x[2])]:
            rejects(lambda: m.encode_payload(x[0], accepted, rejected, x[3]))
        assert len(m.encode_payload(x[0], 8, None, x[3])) == 8
        bad_state = copy.deepcopy(m.state_dict())
        key = "codebook" if m.codec_type == "grouped_vq8" else "scale"
        bad_state[key].reshape(-1)[0] += 1
        rejects(lambda: restore_codec(identity, bad_state))
        rejects(lambda: restore_codec(dict(identity, version=2), m.state_dict()))
        if m.codec_type == "grouped_vq8":
            ids = torch.tensor([0, 1, 127, 128, 200, 250, 254, 255])
            assert m._wire(ids) == bytes(ids.tolist())
            assert torch.equal(m.decode_payload(bytes(ids.tolist())), m.codebook[torch.arange(8), ids].reshape(1, 128))
        else:
            units = torch.tensor([[-200., -127., -2.5, -.5, .5, 1.5, 127., 200.]])
            assert m.pack(units * m.scale.float()) == bytes([129, 129, 254, 0, 0, 2, 127, 127])
            rejects(lambda: m.decode_payload(bytes([128]) * 8))
        results[str(dtype)] = {"wire_hex": b.hex(), "wire_vs_train_control_max_abs": error,
                               "checkpoint_bytes_identical": True}
    results["fp32_bf16_bytes_equal"] = results["torch.float32"]["wire_hex"] == results["torch.bfloat16"]["wire_hex"]
    return results


def loss(head, row):
    return F.cross_entropy(head.forward_logits(row[4], row[3], 1), row[5])


def run(method, out):
    target, draft, head = setup(method)
    train = dataset(target, draft, head, "train", range(32))
    valid = dataset(target, draft, head, "validation", range(40, 48))
    assert not {r[1] for r in train} & {r[1] for r in valid}
    m = head.message
    fit_ms = None
    if method == "pca_int8":
        rejects(lambda: m.encode_payload(*valid[0][2]))
        rejects(lambda: m.fit(r[:3] for r in valid))
        rejects(lambda: m.fit([train[0][:3]] * 12))
        rejects(lambda: m.fit([train[0][:3]] * 7))
        start = time.perf_counter()
        m.fit(r[:3] for r in train)
        fit_ms = 1000 * (time.perf_counter() - start)
        rejects(lambda: m.fit(r[:3] for r in train))
    frozen = {n: p.detach().clone() for n, p in head.named_parameters() if not p.requires_grad}
    buf = copy.deepcopy(m.state_dict())
    optimizer = torch.optim.AdamW(m.parameters(), lr=.002, weight_decay=0.)
    losses, gradients = [], {}
    start = time.perf_counter()
    for step in range(24):
        optimizer.zero_grad(set_to_none=True)
        value = loss(head, train[step % len(train)])
        value.backward()
        losses.append(float(value.detach()))
        for n, p in m.named_parameters():
            assert p.grad is not None and p.grad.isfinite().all()
            gradients[n] = max(gradients.get(n, 0.), float(p.grad.norm()))
        optimizer.step()
    train_ms = 1000 * (time.perf_counter() - start)
    assert all(v > 0 for v in gradients.values())
    for n, p in head.named_parameters():
        if n in frozen:
            assert p.grad is None and torch.equal(frozen[n], p)
    for n, p in m.named_parameters():
        assert p.isfinite().all() and not torch.equal(p, buf[n])
    assert all(torch.equal(v, buf[n]) for n, v in m.named_buffers())
    checks = wire_checks(m, valid[0][2])
    with torch.no_grad():
        heldout_ce = sum(float(loss(head, r)) for r in valid) / len(valid)
        clipped = [int((m.normalized(m.coordinates(*r[2])).abs() > 127).sum())
                   for r in valid] if method != "grouped_vq8" else []
    identity = binding(m)
    path = out / (method + ".pt")
    torch.save({"binding": identity, "message": m.state_dict(), "repair_U": head.base.core.repair.U.weight}, path)
    data = torch.load(path, weights_only=True)
    restored = restore_codec(identity, data["message"])
    assert restored.encode_payload(*valid[0][2]) == m.encode_payload(*valid[0][2])
    return {"binding": identity, "frozen_parameters": len(frozen), "train_requests": [r[1] for r in train],
            "heldout_requests": [r[1] for r in valid], "training_steps": 24, "losses": losses,
            "heldout_ce_smoke_only": heldout_ce, "max_gradient_norm": gradients, "wire": checks,
            "heldout_clipped_fraction": sum(clipped)/(8*len(clipped)) if clipped else None,
            "fit_ms": fit_ms, "optimizer_smoke_ms": train_ms}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    results = {m: run(m, args.out) for m in METHODS}
    (args.out / "training.json").write_text(json.dumps(results, indent=2))
    print("passed", list(results))
