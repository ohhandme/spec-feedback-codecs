"""Evaluate the fixed first 200 development requests using a dc6-selected codec."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "formal"))
import run as parent
import torch
from existing import BASE_C, base_only, historical, load_exact
from feedback_codec.hf_backend import HFTargetBackend, assert_compatible_pair
from message_codecs import binding
from raw_data import (
    COUNTS,
    INDEX,
    MODELS,
    ORIGINAL_ROSTER,
    ROSTER,
    digest,
    lines,
    partition,
    read,
    read_roster,
    require,
    sha,
    write,
)
from verify503 import ARMS, check_row, loopback

TRAIN_COMMIT = "dc6ac64318906e0b5890af4752b377630fe6fc51"
TRAIN_CODE = "b87306345007499c386ecba201f047412bf07ec95b46ce19d72a1a2d1936baf7"
IDS_SHA = "8012363ce8a15d0b3330ecba6cae520877cdf335c58e389b9026d7f0fbc6aeb6"
WAVE = Path("/data/autoresearch/classic-261114-task59-fdbf2f8a")


def cohort(roster_path):
    ids_path = HERE / "request_ids.txt"
    require(sha(ids_path) == IDS_SHA, "preregistered ordered IDs changed")
    ids = ids_path.read_text("ascii").splitlines()
    rows = read_roster(roster_path)
    require(len(rows) == len({r["request_id"] for r in rows}) == 503, "original roster size differs")
    require(
        len(set(ids)) == len(ids) == 200 and ids == [r["request_id"] for r in rows[:200]],
        "first 200 IDs/order differ",
    )
    return rows[:200]


def load_selected(folder, key, head):
    best = read(folder / "selected.json")
    require(
        best["key"] == key
        and best["updates"] == 24485
        and best["tokens"] == 6087440
        and best["epochs"] == 5
        and best["epoch"] in range(1, 6),
        "dc6 training selection differs",
    )
    history = read(folder / "history.json")
    require([r["epoch"] for r in history] == list(range(1, 6)), "incomplete calibration history")
    winner = min(history, key=lambda r: (r["request_equal_ce"], r["epoch"]))
    require(
        all(winner[k] == best[k] for k in ("epoch", "request_equal_ce", "sha256")),
        "selected checkpoint is not the original calibration winner",
    )
    path = folder / f"epoch{best['epoch']}.pt"
    require(sha(path) == best["sha256"], "selected file digest differs")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    require(ckpt["key"] == key and ckpt["epoch"] == best["epoch"], "checkpoint/key mismatch")
    load_exact(head, ckpt["model"])
    require(binding(head.message) == ckpt["codec_binding"], "codec binding mismatch")
    return best["sha256"]


def check_trace(result, row, arm, contract):
    check_row(result, row, arm, contract)
    require(
        not result["warmup"]
        and result["trial_id"] == row["request_id"] + "-" + arm
        and result["codec_binding"] == contract["codec_binding"],
        "trial identity mismatch",
    )
    oracle = row["expected_target_token_ids"]
    offset = 0
    for r in result["rounds"]:
        p = r["candidate_ids"]
        remaining = contract["max_new"] - offset
        require(
            offset < len(oracle)
            and 1 <= len(p) <= min(contract["window"], remaining)
            and contract["eos_token_id"] not in p[:-1]
            and (len(p) == min(contract["window"], remaining) or p[-1] == contract["eos_token_id"]),
            "invalid candidate window",
        )
        accepted = 0
        for candidate, target in zip(p, oracle[offset:]):
            if candidate != target:
                break
            accepted += 1
        size = min(len(oracle) - offset, accepted + 1)
        require(
            accepted == r["accepted"] and r["committed_ids"] == oracle[offset : offset + size],
            "round differs from frozen greedy decisions",
        )
        offset += size
    require(offset == len(oracle), "incomplete request")


def verify_pair(trials):
    def trajectory(arm):
        return [(r["candidate_ids"], r["accepted"], r["committed_ids"]) for r in trials[arm]["rounds"]]

    require(trajectory("raw") == trajectory("shadow"), "RAW/SHADOW trajectories differ")


def recover_rows(out, rows, contract, original_contract, source=None, *, sealed_only=False):
    """Validate a request's three archived measurements together before copying any arm."""
    path = out / "contract.json"
    if path.exists():
        require(read(path) == contract, "eval200 contract/cohort changed")
    else:
        require(not any(out.iterdir()), "unbound eval200 output is nonempty")
        write(path, contract)
    ids = {r["request_id"] for r in rows}
    if source is not None:
        require(
            read(source / "contract.json") == original_contract, "503 source method/checkpoint/roster differs"
        )
        if not sealed_only:
            completion = read(source / "completion.json")
            require(
                completion["requests"] == 503 and completion["contract"] == digest(original_contract),
                "503 source completion differs",
            )
    for arm in ARMS:
        folder = out / arm
        folder.mkdir(exist_ok=True)
        require(all(p.stem in ids for p in folder.glob("*.json")), "extra request from a different cohort")

    imported = []
    for row in rows:
        name = row["request_id"] + ".json"
        present = {}
        for arm in ARMS:
            dest = out / arm / name
            if dest.exists():
                present[arm] = read(dest)
                check_trace(present[arm], row, arm, contract)
        if len(present) == len(ARMS):
            verify_pair(present)
            continue
        if source is None:
            continue
        paths = {arm: source / arm / name for arm in ARMS}
        if sealed_only and not all(path.is_file() for path in paths.values()):
            continue
        archive = {}
        for arm, path in paths.items():
            raw = path.read_bytes()
            trial = json.loads(raw)
            check_trace(trial, row, arm, original_contract)
            archive[arm] = {
                **trial,
                "contract_id": digest(contract),
                "source_row_sha256": hashlib.sha256(raw).hexdigest(),
                "source_contract_id": trial["contract_id"],
                "measurement_origin": "archived_eval503",
            }
        verify_pair(archive)
        verify_pair({**archive, **present})
        for arm in ARMS:
            if arm not in present:
                write(out / arm / name, archive[arm])
        imported.append(row["request_id"])
    return imported


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="same dc6 formal output root as training")
    p.add_argument("--method", choices=parent.METHODS, required=True)
    p.add_argument("--existing", type=Path, help="optional original grouped VQ production runtime root")
    p.add_argument("--index", type=Path, default=Path("/data/a100-repair-raw5000-20261113-v1/index.jsonl"))
    p.add_argument("--roster", type=Path, default=WAVE / "wave2-dev503-20261114/roster503.json")
    p.add_argument(
        "--base", type=Path, default=WAVE / "matrix-fc65585b/continuation-v5/runs/C/train/epoch1.pt"
    )
    p.add_argument(
        "--manifest", type=Path, default=Path("/data/task59-a100-closedloop-20261114-v1/models.json")
    )
    p.add_argument("--models", type=Path, default=Path("/data/transfer_from_4090/models"))
    reuse = p.add_mutually_exclusive_group()
    reuse.add_argument(
        "--reuse-complete-503",
        action="store_true",
        help="use verified rows of the matching completed eval503",
    )
    reuse.add_argument(
        "--reuse-sealed-503",
        action="store_true",
        help="import fully verified three-arm requests even while eval503 is incomplete",
    )
    args = p.parse_args()
    rows = cohort(args.roster)
    require(parent.code_binding() == TRAIN_CODE, "underlying dc6 training/evaluation source differs")
    import fcntl

    with (args.out / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(torch.cuda.device_count() == 1, "expose only the assigned GPU")
        device = torch.device("cuda:0")
        parent.check_models(args, ["Qwen3-0.6B", "Qwen3-8B"])
        draft = parent.Draft(str(args.models / "Qwen3-0.6B"), None, device=str(device), dtype=torch.bfloat16)
        cfg = parent.CodecConfig(4096, 1024, draft.vocab_size)
        folder = args.out / args.method
        if args.existing is None:
            meta = read(args.out / "binding.json")
            require(
                meta["index_sha256"] == INDEX and meta["roster_sha256"] in (ROSTER, ORIGINAL_ROSTER),
                "wrong data binding",
            )
            key = {
                "method": args.method,
                "data": digest(meta),
                "frozen_C_secondary": BASE_C,
                "models": MODELS,
                "code": parent.code_binding(),
                "seed": 42,
                "epochs": 5,
                "batch": 32,
                "lr": 0.0001,
                "wd": 0.01,
            }
            head = parent.init_head(draft, args.method, base_only(args.base), cfg)
            selected = load_selected(folder, key, head)
        else:
            require(sha(args.index) == INDEX, "wrong source index")
            partition(lines(args.index), read_roster(args.roster), COUNTS)
            head = parent.HTaskHead(
                cfg,
                draft.model.get_output_embeddings(),
                draft_embedding=draft.model.get_input_embeddings(),
                codec_type=args.method,
            ).to(device)
            src = historical(args.existing, "H-VQ64-TASK", args.method, head)
            selected = src["checkpoint_sha256"]
            key = {**src, "eval_code": parent.code_binding()}
            folder = folder / ("existing-" + selected[:16])
        original = {
            "max_new": 128,
            "window": 8,
            "eos_token_id": 151645,
            "roster": sha(args.roster),
            "checkpoint_sha256": selected,
            "codec_binding": binding(head.message),
            "training": key,
            "transport": "single-GPU local TCP",
            "draft_implementation": "reference",
        }
        contract = {
            **original,
            "scope": "eval200",
            "expected_requests": 200,
            "ordered_ids_sha256": IDS_SHA,
            "parent_code_commit": TRAIN_COMMIT,
            "parent_code_sha256": TRAIN_CODE,
            "evaluation_script_sha256": sha(Path(__file__)),
        }
        out = folder / "eval200"
        out.mkdir(parents=True, exist_ok=True)
        source = folder / "eval503" if args.reuse_complete_503 or args.reuse_sealed_503 else None
        imported = recover_rows(out, rows, contract, original, source, sealed_only=args.reuse_sealed_503)
        if source is not None:
            print(f"Imported archived measurements for {len(imported)} requests this invocation.", flush=True)
        head.requires_grad_(False).eval()
        target = HFTargetBackend(
            str(args.models / "Qwen3-8B"), None, device=str(device), dtype=torch.bfloat16
        )
        assert_compatible_pair(target, draft)
        totals = loopback(rows, target, draft, head, device, out, contract)
        for row in rows:
            for arm in ARMS:
                check_trace(read(out / arm / (row["request_id"] + ".json")), row, arm, contract)
        write(
            out / "completion.json",
            {
                "scope": "eval200",
                "expected_requests": 200,
                "requests": len(rows),
                "ordered_ids_sha256": IDS_SHA,
                "contract": digest(contract),
                "totals": totals,
                "completed_unix": time.time(),
            },
        )


if __name__ == "__main__":
    main()
