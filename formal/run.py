"""Formal GPU1 runner. Only evaluation constructs the frozen Target."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "demo"), str(ROOT / "demo/vendor")]

import torch
from safetensors.torch import load_file
from feedback_codec.codec import CodecConfig
from feedback_codec.hf_backend import _FrozenHFBackend, HFDraftBackend
from h_vq64_head import HTaskHead
from message_codecs import METHODS, binding
from raw_data import (
    COUNTS,
    INDEX,
    MODELS,
    ORIGINAL_ROSTER,
    ROSTER,
    TRAIN_TOKENS,
    digest,
    prepare,
    read,
    require,
    sha,
    write,
)
from objective import fit_pca, losses, take

from existing import BASE_C, base_only, load_exact


class Draft(_FrozenHFBackend):
    teacher_hidden = HFDraftBackend.teacher_hidden


def check_models(args, names):
    require(sha(args.manifest) == MODELS, "model manifest SHA differs")
    manifest = read(args.manifest)
    for name in names:
        for file, entry in manifest[name].items():
            require(sha(args.models / name / file) == entry["sha256"], "model file differs: " + file)


def init_head(draft, method, initial, cfg):
    torch.manual_seed(42)
    head = HTaskHead(
        cfg,
        draft.model.get_output_embeddings(),
        draft_embedding=draft.model.get_input_embeddings(),
        codec_type=method,
    )
    require(not any(p.requires_grad for p in draft.model.parameters()), "Draft must be frozen")
    load_exact(head.base, initial)
    head.base.requires_grad_(False).eval()
    return head.to(draft.device)


def save_pt(path, state):
    tmp = path.with_suffix(".pending")
    torch.save(state, tmp)
    tmp.replace(path)


def code_binding():
    files = list((ROOT / "formal").glob("*.py"))
    files += [
        ROOT / "demo" / name
        for name in (
            "message_codecs.py",
            "h_vq64_head.py",
            "h_vq64_codec.py",
            "causal_operands.py",
            "latency_smoke.py",
            "cached_draft.py",
        )
    ]
    files += list((ROOT / "demo/vendor/feedback_codec").glob("*.py"))
    return digest({str(p.relative_to(ROOT).as_posix()): sha(p) for p in files})


def read_data(root):
    meta = read(root / "binding.json")
    require(
        meta["index_sha256"] == INDEX and meta["roster_sha256"] in (ROSTER, ORIGINAL_ROSTER),
        "prepared source differs",
    )
    data = {}
    for split, (requests, windows) in COUNTS.items():
        path = root / (split + ".safetensors")
        info = meta["splits"][split]
        require(
            info["requests"] == requests
            and info["windows"] == windows
            and sha(path) == info["packed_sha256"],
            "packed split binding differs",
        )
        data[split] = load_file(str(path))
    require(meta["splits"]["train"]["tokens"] == TRAIN_TOKENS, "training tokens differ")
    return data, digest(meta)


@torch.no_grad()
def calibrate(head, table, data, n_requests, device, batch_size):
    head.eval()
    sums = torch.zeros(n_requests, dtype=torch.float64)
    counts = torch.zeros(n_requests, dtype=torch.long)
    for off in range(0, len(data["request"]), batch_size):
        batch = take(data, slice(off, off + batch_size), device)
        values = losses(head, table, batch).cpu().double()
        requests = batch["request"].cpu()
        sums.index_add_(0, requests, values)
        counts.index_add_(0, requests, torch.ones_like(requests))
    require((counts > 0).all(), "missing calibration request")
    return float((sums / counts).mean())


def train(head, table, data, key, out, device, *, epochs=5, batch_size=32, n_requests=503, interval=128):
    train_data = data["train"]
    n = len(train_data["accepted"])
    params = [p for p in head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.01)
    resume = out / "resume.pt"
    state = {"key": key, "epoch": 1, "position": 0, "updates": 0, "tokens": 0, "history": []}

    def checkpoint():
        save_pt(
            resume,
            {
                "state": state,
                "model": head.trainable_state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.random.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            },
        )

    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        require(saved["state"]["key"] == key, "resume belongs to different method, data or source")
        head.load_trainable_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        torch.random.set_rng_state(saved["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
    else:
        if head.message.codec_type == "pca_int8":
            write(out / "pca.json", fit_pca(head.message, train_data, table, "train"))
        torch.manual_seed(42)
        checkpoint()
    tokens_per_epoch = int(train_data["labels"].ne(-100).sum())
    for epoch in range(state["epoch"], epochs + 1):
        head.train()
        head.base.eval()
        order = torch.randperm(n, generator=torch.Generator().manual_seed(42 + epoch))
        for start in range(state["position"], n, batch_size):
            batch = take(train_data, order[start : start + batch_size], device)
            optimizer.zero_grad(set_to_none=True)
            loss = losses(head, table, batch).mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, float("inf"), error_if_nonfinite=True)
            optimizer.step()
            state["tokens"] += int(batch["labels"].ne(-100).sum())
            state["updates"] += 1
            state["position"] = start + len(batch["accepted"])
            if state["updates"] == 1 or state["updates"] % interval == 0:
                checkpoint()
                write(
                    out / "progress.json",
                    {**state, "loss": float(loss.detach()), "gradient_norm": float(norm)},
                )
        require(
            state["tokens"] == epoch * tokens_per_epoch
            and state["updates"] == epoch * ((n + batch_size - 1) // batch_size),
            "unequal training budget",
        )
        metric = calibrate(head, table, data["calibration"], n_requests, device, batch_size)
        epoch_path = out / f"epoch{epoch}.pt"
        save_pt(
            epoch_path,
            {
                "key": key,
                "epoch": epoch,
                "model": head.trainable_state_dict(),
                "codec_binding": binding(head.message),
            },
        )
        state["history"].append({"epoch": epoch, "request_equal_ce": metric, "sha256": sha(epoch_path)})
        state.update(epoch=epoch + 1, position=0)
        checkpoint()
        write(out / "history.json", state["history"])
    best = min(state["history"], key=lambda r: (r["request_equal_ce"], r["epoch"]))
    require(sha(out / f"epoch{best['epoch']}.pt") == best["sha256"], "selected weights changed")
    write(
        out / "selected.json",
        {
            **best,
            "key": key,
            "updates": state["updates"],
            "tokens": state["tokens"],
            "epochs": epochs,
            "new_teacher_labels": 0,
        },
    )
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("prepare", "train", "eval", "inspect-existing", "eval-existing", "inspect-base")
    )
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--index", type=Path, default=Path("/data/a100-repair-raw5000-20261113-v1/index.jsonl")
    )
    parser.add_argument("--cache", type=Path, default=Path("/data/a100-repair-raw5000-20261113-v1/cache"))
    parser.add_argument(
        "--shards", type=Path, default=Path("/data/a100-ultrachat5000-vanilla-20261111-7858dc6a/shards")
    )
    parser.add_argument(
        "--roster",
        type=Path,
        default=Path(
            "/data/autoresearch/classic-261114-task59-fdbf2f8a/wave2-dev503-20261114/roster503.json"
        ),
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(
            "/data/autoresearch/classic-261114-task59-fdbf2f8a/matrix-fc65585b/continuation-v5/runs/C/train/epoch1.pt"
        ),
    )
    parser.add_argument("--models", type=Path, default=Path("/data/transfer_from_4090/models"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("/data/task59-a100-closedloop-20261114-v1/models.json")
    )
    parser.add_argument("--existing", type=Path, help="original A100 continuation-v4/v5 root")
    parser.add_argument("--group", default="H-VQ64-TASK")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (args.out / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == "prepare":
            torch.set_num_threads(2)
            prepare(args)
            return
        require(args.method is not None, "method required")
        if args.phase == "inspect-base":
            initial = base_only(args.base)
            emb = torch.nn.Embedding(151936, 1024, device="meta").requires_grad_(False)
            output = torch.nn.Linear(1024, 151936, bias=False, device="meta")
            output.weight = emb.weight
            probe = HTaskHead(
                CodecConfig(4096, 1024, 151936), output, draft_embedding=emb, codec_type=args.method
            )
            load_exact(probe.base, initial)
            result = {
                "status": "base_only",
                "checkpoint_sha256": BASE_C,
                "parameters": sum(t.numel() for t in initial.values()),
                "tensors": len(initial),
                "trained_codec_included": False,
                "new_training_updates": 0,
            }
            write(args.out / "C-base-only.json", result)
            print(result, flush=True)
            return
        if args.phase in ("inspect-existing", "eval-existing"):
            from existing import check_schema, historical

            require(args.existing is not None, "--existing is required")
            check_schema(args.group, args.method)
            require(sha(args.index) == INDEX, "original index/503 roster mismatch")
            from raw_data import lines, partition, read_roster

            partition(lines(args.index), read_roster(args.roster), COUNTS)
            if args.phase == "inspect-existing":
                # The base vocabulary tensors are frozen and absent from a trainable checkpoint.
                emb = torch.nn.Embedding(151936, 1024, device="meta").requires_grad_(False)
                output = torch.nn.Linear(1024, 151936, bias=False, device="meta")
                output.weight = emb.weight
                head = HTaskHead(
                    CodecConfig(4096, 1024, 151936), output, draft_embedding=emb, codec_type=args.method
                )
                result = historical(args.existing, args.group, args.method, head)
                write(args.out / (args.method + "-existing.json"), result)
                print(result, flush=True)
                return
        require(torch.cuda.device_count() == 1, "launcher must expose only the assigned GPU")
        device = torch.device("cuda:0")

        names = ["Qwen3-0.6B"] + (["Qwen3-8B"] if args.phase in ("eval", "eval-existing") else [])
        check_models(args, names)
        root = args.out / args.method
        root.mkdir(exist_ok=True)
        draft = Draft(str(args.models / names[0]), None, device=str(device), dtype=torch.bfloat16)
        cfg = CodecConfig(4096, 1024, draft.vocab_size)
        if args.phase == "eval-existing":
            from verify503 import evaluate_selected

            head = HTaskHead(
                cfg,
                draft.model.get_output_embeddings(),
                draft_embedding=draft.model.get_input_embeddings(),
                codec_type=args.method,
            ).to(device)
            source = historical(args.existing, args.group, args.method, head)
            out = root / ("existing-" + source["checkpoint_sha256"][:16])
            out.mkdir(exist_ok=True)
            write(out / "source.json", source)
            evaluate_selected(
                args,
                out,
                {**source, "eval_code": code_binding()},
                head,
                draft,
                device,
                source["checkpoint_sha256"],
            )
            return
        head = init_head(draft, args.method, base_only(args.base), cfg)
        meta = read(args.out / "binding.json")
        require(
            meta["index_sha256"] == INDEX and meta["roster_sha256"] in (ROSTER, ORIGINAL_ROSTER),
            "data binding mismatch",
        )
        key = {
            "method": args.method,
            "data": digest(meta),
            "frozen_C_secondary": BASE_C,
            "models": MODELS,
            "code": code_binding(),
            "seed": 42,
            "epochs": 5,
            "batch": 32,
            "lr": 0.0001,
            "wd": 0.01,
        }
        if args.phase == "train":
            data, source = read_data(args.out)
            require(source == key["data"], "binding changed during load")
            train(head, draft.model.get_input_embeddings(), data, key, root, device)
        else:
            from verify503 import evaluate

            evaluate(args, root, key, head, draft, device)


if __name__ == "__main__":
    main()
