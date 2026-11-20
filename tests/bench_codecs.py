"""Single-thread CPU timings. Latencies are host wall microseconds and overlap by design."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from unittest.mock import patch
import time

import cpu_parity as fx
from train_codecs import METHODS, ROOT, binding, make_codec, restore_codec, setup, torch
import cpu_test
from feedback_codec.fixed_probe import ProtocolView
from h_vq64_head import MessageState


def stats(values):
    ordered = sorted(values)
    index = .95 * (len(ordered) - 1)
    low = int(index)
    return {"n": len(values), "p50_us": statistics.median(ordered), "mean_us": statistics.mean(ordered),
            "p95_us": ordered[low] + (ordered[min(low + 1, len(ordered)-1)] - ordered[low]) * (index-low)}


def measured(fn, *args):
    start = time.perf_counter_ns()
    value = fn(*args)
    return value, (time.perf_counter_ns() - start) / 1000.


def sample(h, d, i, generator):
    def normal(width):
        return torch.randn(1, width, generator=generator)
    return normal(h), i % 9, None if i % 9 == 8 else normal(d), normal(d)


def config(path):
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "hidden_size": json.loads(raw)["hidden_size"]}


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(32)
    model = [config(args.target_config), config(args.draft_config)]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    report = {"environment": {"cpu": platform.processor(), "torch": str(torch.__version__),
                              "transformers": fx.transformers.__version__, "threads": torch.get_num_threads(),
                              "dtype": "float32", "gpu_event_us": None}, "model_configs": model,
              "warmup_per_method": args.warmup, "repeats_each_round": args.iterations, "rounds": 3,
              "scope": "tiny Qwen3 CPU localhost; production widths use random inputs and codec only",
              "stages_overlap": True, "shapes": {}}
    for scope, h, d in [("tiny", 32, 32), ("production_shape", *(m["hidden_size"] for m in model))]:
        gen = torch.Generator().manual_seed(913)
        fit = [("train", str(i), sample(h, d, i, gen)) for i in range(32)]
        inputs = [sample(h, d, i, gen) for i in range(64)]
        codecs, meta, cached = {}, {}, {}
        for method in METHODS:
            start = time.perf_counter_ns()
            if scope == "tiny":
                ck = torch.load(args.test_root / (method + ".pt"), weights_only=True)
                codec = restore_codec(ck["binding"], ck["message"])
            else:
                codec = make_codec(method, h, d)
            construct_us = (time.perf_counter_ns() - start)/1000
            fit_us = None
            if scope == "production_shape" and method == "pca_int8":
                _, fit_us = measured(codec.fit, fit)
            payload, first = measured(codec.encode_payload, *inputs[0])
            _, cold_decode = measured(codec.decode_payload, payload)
            codecs[method] = codec
            meta[method] = {"binding": binding(codec), "payload_bytes": 8, "construct_or_load_us": construct_us,
                            "first_encode_us": first, "first_decode_us": cold_decode, "offline_fit_us": fit_us,
                            "parameters": sum(p.numel() for p in codec.parameters()),
                            "parameter_bytes": sum(p.numel()*p.element_size() for p in codec.parameters()),
                            "buffers_bytes": sum(p.numel()*p.element_size() for p in codec.buffers()),
                            "receiver_bytes": sum(v.numel()*v.element_size() for n,v in codec.state_dict().items()
                                                  if n == "scale" or n == "codebook" or n.startswith("decoder."))}
            if scope == "tiny":
                _, backend, head = setup(method)
                head.message = codec
                head.base.core.repair.U.weight.copy_(ck["repair_U"])
                head.requires_grad_(False).eval()
                draft = fx.CachedDraft(backend, head)
                draft.reset(fx.smoke.Meter("cpu"))
                reply = ProtocolView((1, 3, 4), 8, 0, 12, 4)
                base = draft.head.base.prepare_view(reply, draft.backend)
                cached[method] = (draft, base)
        raw = {m: defaultdict(list) for m in METHODS}
        clipped = {m: [0, 0] for m in METHODS}
        for repeat in range(3):
            for i in range(-args.warmup if repeat == 0 else 0, args.iterations):
                x = inputs[(i + repeat * args.iterations) % len(inputs)]
                offset = (i + repeat) % 3
                for method in METHODS[offset:] + METHODS[:offset]:
                    m = codecs[method]
                    times = {}
                    if method == "grouped_vq8":
                        z, times["encoder"] = measured(m.logits, *x)
                        b, times["quantize_pack"] = measured(lambda: m._wire(z.argmax(-1)))
                    else:
                        z, times["encoder"] = measured(m.coordinates, *x)
                        b, times["quantize_pack"] = measured(m.pack, z)
                        if i >= 0:
                            clipped[method][0] += int((m.normalized(z).abs() > 127).sum())
                            clipped[method][1] += 8
                    _, times["decoder"] = measured(m.decode_payload, b)
                    _, times["complete_codec"] = measured(lambda: m.decode_payload(m.encode_payload(*x)))
                    if scope == "tiny":
                        draft, base = cached[method]
                        state = MessageState(base, (), b, base.corrected_prefix)
                        ctrl, times["control_including_decode"] = measured(draft.build_control, state)
                        _, times["repair_projection"] = measured(draft.head.base.core.repair.repair_from_control,
                                                                  base.local_hidden, ctrl, 1)
                    if i >= 0:
                        for k, val in times.items():
                            raw[method][k].append(val)
        for method, (num, den) in clipped.items():
            meta[method]["heldout_clipped_coordinate_fraction"] = num / den if den else None
        report["shapes"][scope] = {"target_hidden": h, "draft_hidden": d, "input_dim": h+2*d+1, "codecs": meta,
                                   "timings": {m: {k: stats(v) for k,v in t.items()} for m,t in raw.items()}}
        (out / (scope + "-samples-us.json")).write_text(json.dumps(raw))
    report["localhost_tcp"] = {}
    # Each local connection warms all arms, then measures three paired requests.
    # Rotate method order across independent connections to avoid a fixed startup order.
    for rotation in range(3):
        order = METHODS[rotation:] + METHODS[:rotation]
        for method in order:
            root = out / f"tcp-{rotation}-{method}"
            options = ["cpu_test.py", "--codec", method, "--implementation", "kv",
                       "--checkpoint", str(args.test_root / (method + ".pt")),
                       "--out", str(root), "--warmup-rounds", "1", "--repeats", "1"]
            with patch.object(sys, "argv", options), torch.inference_mode(False):
                cpu_test.main()
    for method in METHODS:
        rows = [json.loads(p.read_bytes()) for p in out.glob(f"tcp-*-{method}/edge/rows/*.json")]
        measured_rows = [r for r in rows if not r["warmup"]]
        report["localhost_tcp"][method] = {}
        for arm in fx.smoke.ARMS:
            subset = [r for r in measured_rows if r["arm"] == arm]
            assert len(subset) == 9
            walls = [r["wall_seconds"] * 1e6 for r in subset]
            report["localhost_tcp"][method][arm] = {**stats(walls), "samples_us": walls,
                "framed_up_bytes": sum(r["framed_application_bytes_up"] for r in subset),
                "framed_down_bytes": sum(r["framed_application_bytes_down"] for r in subset),
                "feedback_bytes": sum(i["feedback_bytes"] for r in subset for i in r["rounds"])}
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print("saved", out / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--target-config", type=Path, default=ROOT / "tests/model_configs/qwen3_8b.json")
    parser.add_argument("--draft-config", type=Path, default=ROOT / "tests/model_configs/qwen3_0_6b.json")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    run(parser.parse_args())
