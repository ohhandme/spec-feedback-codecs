"""Bind the original 3986/503 index to the frozen validation roster and RAW caches."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

INDEX = "6b8797ea5155e4fd9bd01474fd3c669e6902e2025b5723e03e2a5faf311c3ff6"
ROSTER = "a75803ceae25fcf21a4ae42cb87470bca375096d9e0676ed52645a727ae4a836"
ORIGINAL_ROSTER = "576c46525d252bfeeb108206f0197b27d8aa27e923205cbec738c0e2f5b6326c"
MODELS = "3aa01f3c1f4b9bf488fbe696a374775e4ebf371be1a1836f5a1bda02256c3ee4"
COUNTS = {"train": (3986, 156688), "calibration": (503, 19736)}
TRAIN_TOKENS = 1217488


def require(test, reason):
    if not test:
        raise ValueError(reason)


def sha(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def lines(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_roster(path):
    source = sha(path)
    if source == ROSTER:
        return lines(path)
    require(source == ORIGINAL_ROSTER, "unrecognized 503 roster")
    return [
        {
            "request_id": r["request_id"],
            "family_id": r["family_id"],
            "prompt_token_ids": r["prompt_ids"],
            "expected_target_token_ids": r["frozen_target_ids"],
        }
        for r in read(path)
    ]


def partition(records, roster, counts):
    parts = {split: [r for r in records if r["split"] == split] for split in counts}
    ids, families = set(), set()
    for split, rows in parts.items():
        unique = {r["request_id"] for r in rows}
        fam = {r["family_id"] for r in rows}
        require(len(rows) == len(unique) and not unique & ids and not fam & families, "split leakage")
        ids.update(unique)
        families.update(fam)
        require((len(rows), sum(len(r["windows"]) for r in rows)) == counts[split], "split size differs")
        for r in rows:
            for w in r["windows"]:
                off, n = w["offset"], w["length"]
                require(
                    0 < off < len(r["output_ids"])
                    and 1 <= n <= 8
                    and len(w["labels"]) == n
                    and w["labels"] == r["output_ids"][off : off + n]
                    and r["output_ids"][off - 1] == w["correction"],
                    "saved Target labels/offset differ",
                )
    dev = {r["request_id"]: r for r in parts["calibration"]}
    require(len(dev) == len(roster) == len({r["request_id"] for r in roster}), "roster size/duplicates")
    for row in roster:
        r = dev[row["request_id"]]
        require(
            (r["prompt_ids"], r["output_ids"], r["family_id"])
            == (row["prompt_token_ids"], row["expected_target_token_ids"], row["family_id"]),
            "503 roster mismatch",
        )
    return parts


def check_round(r, w, rr):
    offset = w["offset"]
    corrected = r["prompt_ids"] + r["output_ids"][:offset]
    before_answer = corrected[:-1]
    count, proposal = w["accepted"], rr["candidate_ids"]
    require(type(count) is int and 0 <= count <= len(proposal) <= 8, "invalid offered count")
    answer = rr["correction_id"] if count < len(proposal) else rr["bonus_id"]
    rejected = proposal[count] if count < len(proposal) else None
    h = rr["hidden"]
    require(
        rr["observation_key"] == w["observation_key"]
        and rr["accepted_count"] == count
        and w["correction"] == answer == corrected[-1]
        and w["rejected"] == rr["rejected_id"] == rejected
        and rr["committed_ids"] == proposal[:count] + [answer]
        and corrected[-(count + 1) :] == rr["committed_ids"]
        and rr["stop_reason"] == "continue"
        and rr["prefix_after_commit_length"] == len(corrected)
        and h["present"]
        and h["same_effective_target_forward_as_decision"] is True
        and h["available_before_receipt"] is True
        and h["layer"] == "model.norm"
        and h["absolute_position"] == len(corrected) - 2
        and h["source_prefix_sha256"] == digest(before_answer)
        and h["predicted_token_id"] == answer
        and w["hidden_key"] == rr["tensor_prefix"] + ".hidden",
        "illegal RAW decision prefix/answer/H",
    )


def pack_split(split, records, cache_dir, shards, out, dims=(1024, 4096)):
    draft_d, target_d = dims
    total = sum(len(r["windows"]) for r in records)
    x = {
        "hidden": torch.zeros(total, 8, draft_d, dtype=torch.bfloat16),
        "target": torch.empty(total, target_d, dtype=torch.bfloat16),
        "labels": torch.full((total, 8), -100, dtype=torch.long),
    }
    for k in ("request", "accepted", "rejected", "correction"):
        x[k] = torch.empty(total, dtype=torch.long)
    hashes = {}
    pos = 0
    for ri, r in enumerate(records):
        rid = r["request_id"]
        require(Path(rid).name == rid and rid not in (".", ".."), "invalid request ID")
        folder = shards / rid
        trace = read(folder / "trace.json")
        require(
            trace["trajectory_policy"] == "RAW"
            and trace["repair_enabled"] is False
            and trace["prompt_ids"] == r["prompt_ids"]
            and trace["output_ids"] == r["output_ids"],
            "RAW differs",
        )
        cache_path = cache_dir / (rid + ".safetensors")
        cache = load_file(str(cache_path))
        d, h = cache["draft_sequence"], cache["target"]
        require(
            d.ndim == 2
            and d.shape[1] == draft_d
            and h.shape == (len(r["windows"]), target_d)
            and d.dtype == h.dtype == torch.bfloat16
            and d.isfinite().all()
            and h.isfinite().all(),
            "cache shape, dtype or values differ",
        )
        hashes[rid] = [sha(cache_path), sha(folder / "trace.json")]
        with safe_open(str(folder / "features.safetensors"), framework="pt") as source:
            for wi, w in enumerate(r["windows"]):
                check_round(r, w, trace["rounds"][w["round"]])
                original = source.get_tensor(w["hidden_key"])
                require(
                    original.dtype == h.dtype and torch.equal(original.reshape(-1), h[wi]),
                    "cached H differs from RAW",
                )
                h_bytes = original.contiguous().view(torch.uint8).numpy().tobytes()
                require(
                    hashlib.sha256(h_bytes).hexdigest() == trace["rounds"][w["round"]]["hidden"]["sha256"],
                    "RAW hidden digest differs",
                )
                off, n = w["offset"], w["length"]
                history = d[off : off + n]
                require(len(history) == n, "Draft cache too short")
                x["hidden"][pos, :n] = history
                x["target"][pos] = h[wi]
                x["labels"][pos, :n] = torch.tensor(w["labels"])
                for k, v in {
                    "request": ri,
                    "accepted": w["accepted"],
                    "correction": w["correction"],
                    "rejected": -1 if w["rejected"] is None else w["rejected"],
                }.items():
                    x[k][pos] = v
                pos += 1
    path = out / (split + ".safetensors")
    tmp = path.with_suffix(".pending")
    save_file(x, str(tmp))
    tmp.replace(path)
    return {
        "requests": len(records),
        "windows": pos,
        "tokens": int(x["labels"].ne(-100).sum()),
        "source_cache_trace_root": digest(hashes),
        "packed_sha256": sha(path),
    }


def prepare(args):
    require(sha(args.index) == INDEX, "index/roster SHA mismatch")
    parts = partition(lines(args.index), read_roster(args.roster), COUNTS)
    result = {"index_sha256": INDEX, "roster_sha256": sha(args.roster), "splits": {}}
    for split, records in parts.items():
        result["splits"][split] = pack_split(split, records, args.cache, args.shards, args.out)
    require(result["splits"]["train"]["tokens"] == TRAIN_TOKENS, "training label count differs")
    write(args.out / "binding.json", result)
