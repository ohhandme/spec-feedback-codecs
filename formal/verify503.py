"""Same-device TCP loopback. Target decides each token before the frozen oracle is checked."""

from concurrent.futures import ThreadPoolExecutor
import socket
import time

import torch
from feedback_codec.hf_backend import HFTargetBackend, assert_compatible_pair
from latency_smoke import ARMS, CloudSession, NativeDraft, Wire, edge_request
from message_codecs import binding
from raw_data import digest, read, read_roster, require, sha, write


def server(listener, session):
    with listener.accept()[0] as conn:
        wire = Wire(conn)
        while True:
            meta, payload = wire.recv()
            require(not payload, "unexpected upstream bytes")
            kind = meta["type"]
            if kind == "close":
                wire.send({"type": "closed"})
                return
            if kind == "start":
                reply, payload = session.begin(meta)
            elif kind == "verify":
                reply, payload = session.verify(meta)
            elif kind == "finish_request":
                session.finish_request()
                reply, payload = {"type": "finished"}, b""
            else:
                raise ValueError("unknown RPC")
            wire.send(reply, payload)


def check_row(result, row, arm, contract):
    require(
        result["request_id"] == row["request_id"]
        and result["arm"] == arm
        and result["contract_id"] == digest(contract)
        and result["output_ids"] == row["expected_target_token_ids"],
        "row/checkpoint/oracle mismatch",
    )
    rounds = result["rounds"]
    require(
        rounds and [t for r in rounds for t in r["committed_ids"]] == result["output_ids"],
        "commit trace differs",
    )
    for i, r in enumerate(rounds):
        count = r["accepted"]
        require(
            0 <= count <= len(r["candidate_ids"])
            and r["committed_ids"][:count] == r["candidate_ids"][:count]
            and r["feedback_bytes"] == (8 if arm != "raw" and i < len(rounds) - 1 else 0),
            "acceptance/wire mismatch",
        )
    require(
        result["accepted_draft_tokens"] == sum(r["accepted"] for r in rounds)
        and result["verification_rounds"] == len(rounds),
        "acceptance denominator differs",
    )


def loopback(roster, target, draft, head, device, out, contract):
    table = draft.model.get_input_embeddings().weight

    def encode(hidden, accepted, rejected, answer):
        return head.message.encode_payload(
            hidden.reshape(1, -1),
            accepted,
            None if rejected is None else table[rejected].reshape(1, -1),
            table[answer].reshape(1, -1),
        )

    session = CloudSession(target, encode, {r["request_id"]: r for r in roster}, contract, device)
    edge = NativeDraft(draft, head)
    totals = {arm: {"accepted": 0, "rounds": 0, "output_tokens": 0, "feedback_bytes": 0} for arm in ARMS}
    with socket.socket() as listener, ThreadPoolExecutor(max_workers=1) as pool:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        with socket.create_connection(listener.getsockname()) as conn:
            future = pool.submit(server, listener, session)
            wire = Wire(conn)
            for index, row in enumerate(roster):
                rid = row["request_id"]
                trajectories = {}
                for arm in ARMS:
                    dest = out / arm
                    dest.mkdir(exist_ok=True)
                    path = dest / (rid + ".json")
                    if path.exists():
                        result = read(path)
                    else:
                        result = edge_request(wire, edge, row, arm, rid + "-" + arm, False, contract)
                        check_row(result, row, arm, contract)
                        write(path, result)
                    check_row(result, row, arm, contract)
                    trajectories[arm] = [
                        (r["candidate_ids"], r["accepted"], r["committed_ids"]) for r in result["rounds"]
                    ]
                    total = totals[arm]
                    total["accepted"] += result["accepted_draft_tokens"]
                    total["rounds"] += result["verification_rounds"]
                    total["output_tokens"] += len(result["output_ids"])
                    total["feedback_bytes"] += sum(r["feedback_bytes"] for r in result["rounds"])
                require(trajectories["raw"] == trajectories["shadow"], "RAW/SHADOW trajectories differ")
                write(out / "progress.json", {"completed_requests": index + 1, "contract": digest(contract)})
            require(wire.rpc({"type": "close"})[0]["type"] == "closed", "cloud did not finish")
        future.result()
    for total in totals.values():
        total["accepted_per_verification_round"] = total["accepted"] / total["rounds"]
    return totals


def evaluate(args, root, key, head, draft, device):
    best = read(root / "selected.json")
    require(
        best["key"] == key and best["updates"] == 24485 and best["tokens"] == 6087440, "incomplete training"
    )
    path = root / f"epoch{best['epoch']}.pt"
    require(sha(path) == best["sha256"], "selected checkpoint SHA mismatch")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    require(ckpt["key"] == key and ckpt["epoch"] == best["epoch"], "checkpoint key mismatch")
    head.load_trainable_state_dict(ckpt["model"])
    require(binding(head.message) == ckpt["codec_binding"], "codec binding mismatch")
    evaluate_selected(args, root, key, head, draft, device, best["sha256"])


def evaluate_selected(args, root, key, head, draft, device, checkpoint_sha):
    roster = read_roster(args.roster)
    require(len(roster) == len({r["request_id"] for r in roster}) == 503, "expected the original 503")
    head.requires_grad_(False).eval()
    target = HFTargetBackend(str(args.models / "Qwen3-8B"), None, device=str(device), dtype=torch.bfloat16)
    assert_compatible_pair(target, draft)
    out = root / "eval503"
    out.mkdir(exist_ok=True)
    contract = {
        "max_new": 128,
        "window": 8,
        "eos_token_id": 151645,
        "roster": sha(args.roster),
        "checkpoint_sha256": checkpoint_sha,
        "codec_binding": binding(head.message),
        "training": key,
        "transport": "single-GPU local TCP",
        "draft_implementation": "reference",
    }
    if (out / "contract.json").exists():
        require(
            read(out / "contract.json") == contract,
            "existing evaluation directory belongs to another contract",
        )
    write(out / "contract.json", contract)
    totals = loopback(roster, target, draft, head, device, out, contract)
    write(
        out / "completion.json",
        {"requests": 503, "totals": totals, "contract": digest(contract), "completed_unix": time.time()},
    )
