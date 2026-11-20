"""Actual two-host TCP smoke; reference eager decoding, no training labels acquired."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "vendor")]
ARMS = ("raw", "shadow", "active")
MAX_FRAME = 4 * 1024 * 1024


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def checked(path, expected):
    from hashlib import file_digest
    with Path(path).open("rb") as stream:
        require(file_digest(stream, "sha256").hexdigest() == expected, "source hash differs: " + str(path))


def resource_snapshot():
    """Outside timed requests; retain concurrent process/memory evidence."""
    result = {"time_unix": time.time()}
    for name, fields in [("devices", "index,name,memory.total,memory.free,utilization.gpu"),
                          ("processes", "pid,process_name,used_memory")]:
        flag = "--query-gpu=" if name == "devices" else "--query-compute-apps="
        try:
            run = subprocess.run(["nvidia-smi", flag + fields, "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10)
            result[name] = {"exit_code": run.returncode, "output": run.stdout, "error": run.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            result[name] = {"error": str(error)}
    return result


class Wire:
    def __init__(self, channel):
        self.channel = channel
        self.up = self.down = 0

    def send(self, metadata, payload=b""):
        header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        body = struct.pack("!I", len(header)) + header + payload
        require(len(body) <= MAX_FRAME, "frame too large")
        frame = struct.pack("!I", len(body)) + body
        self.channel.sendall(frame)
        self.up += len(frame)

    def exact(self, size):
        blocks = []
        while size:
            block = self.channel.recv(size)
            if not block:
                raise EOFError("peer closed an incomplete frame")
            blocks.append(block)
            size -= len(block)
        return b"".join(blocks)

    def recv(self):
        count = struct.unpack("!I", self.exact(4))[0]
        require(4 <= count <= MAX_FRAME, "invalid frame length")
        frame = self.exact(count)
        head = struct.unpack("!I", frame[:4])[0]
        require(head <= count - 4, "invalid metadata length")
        meta = json.loads(frame[4:4 + head])
        require(isinstance(meta, dict), "metadata must be an object")
        self.down += count + 4
        return meta, frame[4 + head:]

    def rpc(self, metadata):
        start = time.perf_counter()
        self.send(metadata)
        reply, payload = self.recv()
        elapsed = time.perf_counter() - start
        require(reply.get("type") != "error", "cloud error: " + str(reply.get("message")))
        return reply, payload, elapsed


class Meter:
    """GPU events are read after the request; nested stages are not additive."""
    def __init__(self, device=None):
        self.device = device
        self.wall = defaultdict(float)
        self.calls = defaultdict(int)
        self.events = []

    @contextmanager
    def timed(self, name):
        import torch
        cuda = self.device is not None and str(self.device).startswith("cuda")
        if cuda:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.wall[name] += time.perf_counter() - start
            self.calls[name] += 1
            if cuda:
                b.record()
                self.events.append((name, a, b))

    def finish(self):
        import torch
        gpu = defaultdict(float)
        if self.events:
            torch.cuda.synchronize(self.device)
            for name, a, b in self.events:
                gpu[name] += a.elapsed_time(b) / 1000
        return {"host_wall_seconds": dict(self.wall), "gpu_event_seconds": dict(gpu), "calls": dict(self.calls),
                "nested_stages_are_not_additive": True}


class CloudSession:
    def __init__(self, target, encode, roster, contract, device=None):
        self.target, self.encode, self.roster = target, encode, roster
        self.cfg, self.device = contract, device
        self.current = None

    def begin(self, request):
        require(self.current is None, "unfinished cloud request")
        require(request["arm"] in ARMS and request["request_id"] in self.roster, "unknown arm/request")
        row = self.roster[request["request_id"]]
        require(request["prompt_ids"] == row["prompt_token_ids"], "prompt differs from frozen sample")
        self.current = {"trial_id": request["trial_id"], "request_id": request["request_id"],
                        "arm": request["arm"], "warmup": request["warmup"], "rounds": []}
        self.confirmed, self.generated = request["prompt_ids"].copy(), []
        self.expected = row["expected_target_token_ids"][:self.cfg["max_new"]]
        self.meter = Meter(self.device)
        return {"type": "started"}, b""

    def verify(self, request):
        require(self.current is not None, "verify without request")
        require(request["round"] == len(self.current["rounds"])
                and request["prefix_sha256"] == digest(self.confirmed), "edge/cloud prefix or round differs")
        proposal = request["proposals"]
        remaining = self.cfg["max_new"] - len(self.generated)
        eos, window = self.cfg["eos_token_id"], self.cfg["window"]
        require(isinstance(proposal, list) and proposal and all(type(x) is int for x in proposal)
                and len(proposal) <= min(window, remaining) and eos not in proposal[:-1], "invalid proposals")
        require(len(proposal) == min(window, remaining) or proposal[-1] == eos, "unexplained short window")
        start = time.perf_counter()
        before = self.confirmed.copy()
        accepted, answer, rejected, hidden = 0, None, None, None
        for token in proposal:
            with self.meter.timed("target_verify"):
                step = self.target.next(self.confirmed.copy())
            if token != int(step.token_id):
                rejected, answer, hidden = token, int(step.token_id), step.hidden
                self.confirmed.append(answer)
                self.generated.append(answer)
                break
            accepted += 1
            self.confirmed.append(token)
            self.generated.append(token)
            if token == eos:
                break
        done = self.generated[-1] == eos or len(self.generated) == self.cfg["max_new"]
        if rejected is None and not done:
            with self.meter.timed("target_verify"):
                step = self.target.next(self.confirmed.copy())
            answer, hidden = int(step.token_id), step.hidden
            self.confirmed.append(answer)
            self.generated.append(answer)
            done = answer == eos or len(self.generated) == self.cfg["max_new"]
        committed = proposal[:accepted] + ([] if answer is None else [answer])
        require(self.confirmed == before + committed, "invalid authoritative commit")
        # Frozen answers are post-decision assertions only, never a verifier input.
        require(self.generated == self.expected[:len(self.generated)], "actual Target diverged from frozen sample")
        feedback = b""
        if not done and self.current["arm"] != "raw":
            require(hidden is not None and answer is not None, "missing causal encoder boundary")
            with self.meter.timed("cloud_encode_and_pack"):
                feedback = self.encode(hidden, accepted, rejected, answer)
            require(type(feedback) is bytes and len(feedback) == 8, "codec did not produce 8 raw bytes")
        work = time.perf_counter() - start
        reply = {"type": "result", "accepted": accepted, "answer": answer, "rejected": rejected,
                 "done": done, "feedback_bytes": len(feedback), "cloud_work_seconds": work}
        self.current["rounds"].append({**reply, "candidate_ids": proposal, "committed_ids": committed,
                                       "prefix_before_sha256": digest(before), "output_tokens": len(self.generated)})
        if done:
            require(self.generated == self.expected, "Target terminal output differs")
        return reply, feedback

    def finish_request(self):
        require(self.current is not None and self.current["rounds"][-1]["done"], "unfinished trial")
        row = {**self.current, "output_ids": self.generated, "timing": self.meter.finish(),
               "contract_id": digest(self.cfg), "codec_binding": self.cfg.get("codec_binding")}
        self.current = None
        return row


class NativeDraft:
    def __init__(self, backend, head):
        self.backend, self.head = backend, head
        self.device = backend.device
        self.state = None

    def reset(self, meter):
        self.state, self.meter = None, meter
        original = self.head.message.decode_payload
        if hasattr(self, "original_decode"):
            original = self.original_decode
        self.original_decode = original
        def decode(payload):
            with self.meter.timed("edge_codebook_decode"):
                return original(payload)
        self.head.message.decode_payload = decode

    def token(self, prefix, position, arm):
        import torch
        with torch.inference_mode(), self.meter.timed("draft_native_forward"):
            output, hidden = self.backend._forward_with_final_hidden(self.backend._ids(prefix))
        logits = output.logits[0, -1]
        if self.state is not None:
            with torch.inference_mode(), self.meter.timed("edge_head_including_decode_and_projection"):
                repaired = self.head.forward_logits(hidden[:, -1], self.state, position)[0]
            if arm == "active":
                logits = repaired
        with self.meter.timed("draft_decision"):
            return int(logits.argmax().item())

    def prepare(self, confirmed, reply, payload):
        from feedback_codec.fixed_probe import ProtocolView
        from h_vq64_head import MessageState
        import torch
        with torch.inference_mode(), self.meter.timed("edge_feedback_prepare"):
            view = ProtocolView(tuple(confirmed), 8, reply["accepted"], reply["rejected"], reply["answer"])
            base = self.head.base.prepare_view(view, self.backend)
            self.state = MessageState(base, (), payload, tuple(confirmed))


def edge_request(wire, draft, row, arm, trial_id, warmup, contract):
    meter = Meter(getattr(draft, "device", None))
    draft.reset(meter)
    before_up, before_down = wire.up, wire.down
    started = time.perf_counter()
    reply, payload, start_rpc = wire.rpc({"type": "start", "trial_id": trial_id, "request_id": row["request_id"],
        "arm": arm, "warmup": warmup, "prompt_ids": row["prompt_token_ids"]})
    require(reply["type"] == "started" and not payload, "invalid start response")
    confirmed, generated, rounds = row["prompt_token_ids"].copy(), [], []
    first = None
    while len(generated) < contract["max_new"]:
        proposal = []
        for position in range(1, min(contract["window"], contract["max_new"] - len(generated)) + 1):
            proposal.append(draft.token(confirmed + proposal, position, arm))
            if proposal[-1] == contract["eos_token_id"]:
                break
        reply, payload, rpc_time = wire.rpc({"type": "verify", "round": len(rounds),
            "prefix_sha256": digest(confirmed), "proposals": proposal})
        require(reply["type"] == "result" and len(payload) == reply["feedback_bytes"]
                and len(payload) in (0, 8), "invalid verifier reply")
        count, answer = reply["accepted"], reply["answer"]
        require(type(count) is int and 0 <= count <= len(proposal), "invalid acceptance")
        require(reply["rejected"] == (proposal[count] if count < len(proposal) else None), "invalid rejected token")
        committed = proposal[:count] + ([] if answer is None else [answer])
        require(committed and len(generated) + len(committed) <= contract["max_new"], "invalid commit length")
        generated.extend(committed)
        confirmed.extend(committed)
        if hasattr(draft, "commit"):
            draft.commit(confirmed, reply)
        if first is None:
            first = time.perf_counter() - started
        done = generated[-1] == contract["eos_token_id"] or len(generated) == contract["max_new"]
        require(reply["done"] == done and not (done and payload), "terminal feedback is invalid")
        expected = row["expected_target_token_ids"][:contract["max_new"]]
        require(generated == expected[:len(generated)], "verified output differs from expected Target")
        rounds.append({"candidate_ids": proposal, "committed_ids": committed, "accepted": count,
                       "rpc_wall_seconds": rpc_time, "cloud_work_seconds": reply["cloud_work_seconds"],
                       "transport_framing_residual_seconds": rpc_time - reply["cloud_work_seconds"],
                       "feedback_bytes": len(payload)})
        if hasattr(draft, "snapshot"):
            rounds[-1]["draft_cache_after_commit"] = draft.snapshot()
        if done:
            break
        if arm != "raw":
            require(len(payload) == 8, "feedback missing")
            draft.prepare(confirmed, reply, payload)
            if hasattr(draft, "snapshot"):
                rounds[-1]["draft_cache_after_prepare"] = draft.snapshot()
        else:
            require(not payload, "RAW received extra feedback")
    elapsed = time.perf_counter() - started
    require(generated == expected, "terminal output differs")
    generation_up, generation_down = wire.up - before_up, wire.down - before_down
    # Finish/flush telemetry acknowledgements are outside generation time/bytes.
    terminal, extra, _ = wire.rpc({"type": "finish_request"})
    require(terminal["type"] == "finished" and not extra, "cloud trial did not finish")
    return {"trial_id": trial_id, "request_id": row["request_id"], "arm": arm, "warmup": warmup,
        "output_ids": generated, "rounds": rounds, "wall_seconds": elapsed, "ttft_seconds": first,
        "tpot_seconds": (elapsed - first) / (len(generated) - 1) if len(generated) > 1 else None,
        "accepted_draft_tokens": sum(r["accepted"] for r in rounds), "verification_rounds": len(rounds),
        "framed_application_bytes_up": generation_up, "framed_application_bytes_down": generation_down,
        "start_rpc_seconds": start_rpc, "timing": meter.finish(),
        "contract_id": digest(contract), "codec_binding": contract.get("codec_binding"),
        "draft_implementation": getattr(draft, "mode", "reference"),
        "draft_cache_counters": dict(draft.stats) if hasattr(draft, "stats") else {}}


def load_roster(contract):
    checked(HERE / "roster.jsonl", contract["roster_sha256"])
    rows = [json.loads(line) for line in (HERE / "roster.jsonl").read_text().splitlines()]
    require(len(rows) == 3 and len({r["request_id"] for r in rows}) == 3, "smoke sample differs")
    return {r["request_id"]: r for r in rows}


def checkpoint(config):
    import torch
    checked(config["checkpoint_path"], config["contract"]["checkpoint_sha256"])
    payload = torch.load(config["checkpoint_path"], map_location="cpu", weights_only=True)
    state = payload[config.get("checkpoint_state_key", "model")]
    require(all(isinstance(x, torch.Tensor) and bool(x.isfinite().all()) for x in state.values()), "invalid trained weights")
    return state


def load_components(config, role):
    import torch
    from feedback_codec.hf_backend import _FrozenHFBackend, HFTargetBackend, HFDraftBackend
    from feedback_codec.codec import CodecConfig
    from message_codecs import restore_codec
    from h_vq64_head import HTaskHead
    from safetensors import safe_open
    state = checkpoint(config)
    root, device = Path(config[role + "_models_root"]), config[role + "_device"]
    require(("A100" if role == "cloud" else "4090") in torch.cuda.get_device_name(device),
            "actual GPU does not match requested cloud/edge role")
    checked(config[role + "_model_manifest"], config["contract"]["model_manifest_sha256"])
    manifest = json.loads(Path(config[role + "_model_manifest"]).read_text())
    required = ("Qwen3-8B", "Qwen3-0.6B") if role == "cloud" else ("Qwen3-0.6B",)
    for model in required:
        for name, entry in manifest[model].items():
            checked(root / model / name, entry["sha256"])
    if role == "cloud":
        free, total = torch.cuda.mem_get_info(device)
        require(free >= config.get("cloud_min_free_gib", 20) * 1024 ** 3,
                f"parallel cloud memory insufficient: free={free},total={total}")
        target = HFTargetBackend(str(root / "Qwen3-8B"), None, device=device, dtype=torch.bfloat16)
        message = restore_codec(config["contract"]["codec_binding"],
            {name.removeprefix("message."): value for name, value in state.items() if name.startswith("message.")}).to(device).eval()
        message.requires_grad_(False)
        table = None
        for path in sorted((root / "Qwen3-0.6B").glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as handle:
                if "model.embed_tokens.weight" in handle.keys():
                    require(table is None, "duplicate Draft embedding table")
                    table = handle.get_tensor("model.embed_tokens.weight").to(device)
        require(table is not None and table.shape == (151936, 1024), "cloud Draft protocol embedding missing")
        def encode(hidden, accepted, rejected, answer):
            with torch.inference_mode():
                return message.encode_payload(hidden.reshape(1, -1), accepted,
                    None if rejected is None else table[rejected].reshape(1, -1), table[answer].reshape(1, -1))
        return target, encode
    class DraftBackend(_FrozenHFBackend):
        teacher_hidden = HFDraftBackend.teacher_hidden
    backend = DraftBackend(str(root / "Qwen3-0.6B"), None, device=device, dtype=torch.bfloat16)
    head = HTaskHead(CodecConfig(4096, 1024, backend.vocab_size), backend.model.get_output_embeddings(),
                     draft_embedding=backend.model.get_input_embeddings(), codec_type=config["contract"]["codec_binding"]["codec_type"]).to(device)
    head.load_trainable_state_dict(state)
    from message_codecs import binding
    require(binding(head.message) == config["contract"]["codec_binding"], "edge codec state differs")
    head.requires_grad_(False).eval()
    implementation = config["contract"].get("draft_implementation", "reference")
    if implementation != "reference":
        from cached_draft import CachedDraft
        return CachedDraft(backend, head, mode=implementation)
    return NativeDraft(backend, head)


def cloud(config, out):
    write(out / "resources_before_load.json", resource_snapshot())
    target, encode = load_components(config, "cloud")
    write(out / "resources_ready.json", resource_snapshot())
    session = CloudSession(target, encode, load_roster(config["contract"]), config["contract"], config["cloud_device"])
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((config.get("bind_host", "127.0.0.1"), config["port"]))
        listener.settimeout(config.get("timeout_seconds", 300))
        listener.listen(1)
        write(out / "ready.json", {"status": "ready", "contract_id": digest(config["contract"]), "time_unix": time.time()})
        channel, peer = listener.accept()
        with channel:
            channel.settimeout(config.get("timeout_seconds", 300))
            channel.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            wire = Wire(channel)
            while True:
                request, payload = wire.recv()
                require(not payload, "unexpected upstream raw payload")
                kind = request.get("type")
                if kind == "hello":
                    require(request["contract_id"] == digest(config["contract"]), "edge/cloud contract differs")
                    reply, feedback = {"type": "hello", "contract_id": digest(config["contract"])}, b""
                elif kind == "ping":
                    reply, feedback = {"type": "pong"}, b""
                elif kind == "start":
                    reply, feedback = session.begin(request)
                elif kind == "verify":
                    reply, feedback = session.verify(request)
                elif kind == "finish_request":
                    row = session.finish_request()
                    write(out / "rows" / (digest(row["trial_id"]) + ".json"), row)
                    reply, feedback = {"type": "finished"}, b""
                elif kind == "shutdown":
                    require(session.current is None, "unfinished shutdown")
                    wire.send({"type": "shutdown"})
                    break
                else:
                    raise ValueError("unknown RPC command")
                wire.send(reply, feedback)
        write(out / "completion.json", {"status": "complete", "contract_id": digest(config["contract"]),
                                         "framed_bytes_sent": wire.up, "framed_bytes_received": wire.down})
    write(out / "resources_after.json", resource_snapshot())


def edge(config, out):
    write(out / "resources_before_load.json", resource_snapshot())
    draft = load_components(config, "edge")
    write(out / "resources_ready.json", resource_snapshot())
    roster = list(load_roster(config["contract"]).values())
    connected = time.perf_counter()
    with socket.create_connection((config["cloud_host"], config["port"]), timeout=config.get("timeout_seconds", 300)) as channel:
        connect_seconds = time.perf_counter() - connected
        channel.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        wire = Wire(channel)
        reply, _, _ = wire.rpc({"type": "hello", "contract_id": digest(config["contract"])})
        require(reply.get("contract_id") == digest(config["contract"]), "handshake contract differs")
        pings = [wire.rpc({"type": "ping"})[2] for _ in range(16)]
        jobs = [(roster[0], arm, "warmup-" + arm, True) for arm in ARMS]
        for repeat in range(config["contract"]["repeats"]):
            for i, row in enumerate(roster):
                rotate = (repeat + i) % len(ARMS)
                for arm in ARMS[rotate:] + ARMS[:rotate]:
                    jobs.append((row, arm, f"repeat{repeat}-request{row['request_id']}-{arm}", False))
        for i, (row, arm, trial_id, warmup) in enumerate(jobs):
            record = edge_request(wire, draft, row, arm, trial_id, warmup, config["contract"])
            write(out / "rows" / (digest(trial_id) + ".json"), record)
            write(out / "status.json", {"completed": i + 1, "total": len(jobs), "last_trial": trial_id})
        wire.rpc({"type": "shutdown"})
    write(out / "completion.json", {"status": "complete", "contract_id": digest(config["contract"]),
        "trials": len(jobs), "connect_seconds_excluded_from_generation": connect_seconds, "idle_ping_rpc_seconds": pings,
        "scope": "Actual edge/cloud link; reference full-prefix eager verifier; A100 may concurrently train. Small smoke, no isolated speed claim."})
    write(out / "resources_after.json", resource_snapshot())


def summarize(edge_root, cloud_root, out, *, expected_contract=None, expected_trials=None):
    client = [json.loads(p.read_text()) for p in (edge_root / "rows").glob("*.json")]
    cloud_rows = [json.loads(p.read_text()) for p in (cloud_root / "rows").glob("*.json")]
    server = {r["trial_id"]: r for r in cloud_rows}
    require(client and len(client) == len(cloud_rows) == len(server) and
            {r["trial_id"] for r in client} == set(server), "duplicate or incomplete edge/cloud trials")
    if expected_trials is not None:
        require(len(client) == expected_trials, "row count differs from completed trials")
    implementations = {r.get("draft_implementation", "reference") for r in client}
    require(len(implementations) == 1, "mixed Draft implementations")
    contracts = {r.get("contract_id") for r in client + cloud_rows}
    require(len(contracts) == 1, "mixed row contracts")
    if expected_contract is not None:
        require(contracts == {digest(expected_contract)} and
                implementations == {expected_contract.get("draft_implementation", "reference")},
                "rows differ from completed contract or implementation")
    identities = [r.get("codec_binding") for r in client + cloud_rows]
    require(all(item == identities[0] for item in identities), "mixed codec schema or state")
    identity = identities[0]
    require(isinstance(identity, dict) and identity.get("codec_type") in ("grouped_vq8", "pca_int8", "mlp_int8")
            and identity.get("bits") == 64 and all(isinstance(identity.get(k), str) and len(identity[k]) == 64
            and all(c in "0123456789abcdef" for c in identity[k]) for k in ("state_sha256", "schema_sha256")),
            "missing codec schema, budget or state binding")
    if expected_contract is not None:
        require(identities[0] == expected_contract.get("codec_binding"), "codec differs from contract")
    paired = defaultdict(dict)
    for row in client:
        remote = server[row["trial_id"]]
        require(all(row[key] == remote[key] for key in ("request_id", "arm", "warmup")),
                "edge/cloud trial metadata differs")
        require(row["output_ids"] == remote["output_ids"] and len(row["rounds"]) == len(remote["rounds"]), "cross-host output differs")
        require(remote["timing"]["calls"]["target_verify"] == len(row["output_ids"]), "sequential Target forward count differs")
        for a, b in zip(row["rounds"], remote["rounds"]):
            require(a["candidate_ids"] == b["candidate_ids"] and a["committed_ids"] == b["committed_ids"]
                    and a["accepted"] == b["accepted"], "cross-host round differs")
        if not row["warmup"]:
            group = paired[row["trial_id"].rsplit("-", 1)[0]]
            require(row["arm"] not in group, "duplicate paired arm")
            group[row["arm"]] = row
    require(paired, "no measured trials")
    for group in paired.values():
        require(set(group) == set(ARMS), "missing paired arm")
        require(all(r["request_id"] == group["raw"]["request_id"] and
                    r["output_ids"] == group["raw"]["output_ids"] for r in group.values()),
                "paired request or Target output differs")
        require([r["accepted"] for r in group["raw"]["rounds"]] == [r["accepted"] for r in group["shadow"]["rounds"]], "shadow changed accept counts")
        require([r["candidate_ids"] for r in group["raw"]["rounds"]]
                == [r["candidate_ids"] for r in group["shadow"]["rounds"]], "shadow changed Draft decisions")
    from statistics import median
    def q95(values):
        values = sorted(values)
        offset = .95 * (len(values) - 1)
        lo = int(offset)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (offset - lo)
    def stage(rows, key, side):
        pool = [server[r["trial_id"]] if side == "cloud" else r for r in rows]
        count = sum(r["timing"]["calls"].get(key, 0) for r in pool)
        wall = sum(r["timing"]["host_wall_seconds"].get(key, 0) for r in pool)
        gpu = sum(r["timing"]["gpu_event_seconds"].get(key, 0) for r in pool)
        return {"calls": count, "total_host_seconds": wall,
                "mean_host_ms": wall * 1000 / count if count else None,
                "mean_gpu_event_ms": gpu * 1000 / count if count and all(
                    not r["timing"]["calls"].get(key) or key in r["timing"]["gpu_event_seconds"] for r in pool
                ) else None}
    reports = {}
    for arm in ARMS:
        rows = [group[arm] for group in paired.values()]
        tpot = [r["tpot_seconds"] for r in rows if r["tpot_seconds"] is not None]
        reports[arm] = {"requests": len(rows), "total_wall_seconds": sum(r["wall_seconds"] for r in rows),
            "p50_request_seconds": median(r["wall_seconds"] for r in rows),
            "p95_request_seconds_descriptive_only": q95([r["wall_seconds"] for r in rows]),
            "p50_ttft_seconds": median(r["ttft_seconds"] for r in rows),
            "p50_tpot_seconds": median(tpot) if tpot else None,
            "confirmed_tokens_per_second": sum(len(r["output_ids"]) for r in rows) / sum(r["wall_seconds"] for r in rows),
            "accepted_draft_tokens": sum(r["accepted_draft_tokens"] for r in rows),
            "verification_rounds": sum(r["verification_rounds"] for r in rows),
            "framed_application_bytes_up": sum(r["framed_application_bytes_up"] for r in rows),
            "framed_application_bytes_down": sum(r["framed_application_bytes_down"] for r in rows),
            "raw_feedback_payload_bytes": sum(x["feedback_bytes"] for r in rows for x in r["rounds"]),
            "draft_cache_counter_totals": {key: sum(r.get("draft_cache_counters", {}).get(key, 0) for r in rows)
                for key in sorted({key for r in rows for key in r.get("draft_cache_counters", {})})},
            "stages": {key: stage(rows, key, side) for side, key in [
                ("cloud", "cloud_encode_and_pack"), ("cloud", "target_verify"),
                ("edge", "edge_codebook_decode"), ("edge", "edge_feedback_prepare"),
                ("edge", "edge_head_including_decode_and_projection"), ("edge", "draft_native_forward"),
                ("edge", "draft_decoder_forward"), ("edge", "draft_kv_crop"),
                ("edge", "draft_native_projection"), ("edge", "edge_control_prepare_including_decode"),
                ("edge", "edge_head_including_projection")]}}
    differences = {arm + "_minus_raw_seconds": [g[arm]["wall_seconds"] - g["raw"]["wall_seconds"] for g in paired.values()]
                   for arm in ("shadow", "active")}
    write(out, {"status": "complete", "codec_binding": identities[0], "draft_implementation": next(iter(implementations)), "contract_id": next(iter(contracts)), "arms": reports, "paired_differences": differences,
        "scope": "Small concurrent-load smoke. Shadow adds codec+repair work but keeps exact RAW proposals. Nested decode/head timings must not be added twice; RPC residual includes framing/serialization/transport, not pure network RTT. Byte counts are application frames, not TCP/IP capture."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("cloud", "edge", "compare"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cloud-results", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    for name, expected in cfg["contract"]["source_sha256"].items():
        checked(HERE / name, expected)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        if args.phase == "compare":
            terminals = [json.loads((root / "completion.json").read_text()) for root in (args.out, args.cloud_results)]
            require(all(r["status"] == "complete" and r["contract_id"] == digest(cfg["contract"]) for r in terminals),
                    "edge/cloud complete contract differs")
            environments = [json.loads((root / "environment.json").read_text()) for root in (args.out, args.cloud_results)]
            require(environments[0]["hostname"] != environments[1]["hostname"], "same-host run cannot establish requested cross-host timing")
            summarize(args.out, args.cloud_results, args.out / "comparison.json",
                      expected_contract=cfg["contract"], expected_trials=terminals[0]["trials"])
        else:
            import fcntl, os, platform, torch, transformers
            write(args.out / "environment.json", {"hostname": socket.gethostname(), "role": args.phase,
                "gpu": torch.cuda.get_device_name(cfg[args.phase + "_device"]), "python": platform.python_version(),
                "torch": torch.__version__, "transformers": transformers.__version__, "CUDA": torch.version.cuda,
                "contract_id": digest(cfg["contract"]), "config": cfg, "time_unix": time.time()})
            with open(cfg[args.phase + "_smoke_lock"], "a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                write(args.out / "lease.json", {"pid": os.getpid(), "status": "held", "parallel_training_allowed": True})
                try:
                    globals()[args.phase](cfg, args.out)
                finally:
                    write(args.out / "lease.json", {"pid": os.getpid(), "status": "released"})
    except BaseException:
        if not (args.out / "first_error.json").exists():
            write(args.out / "first_error.json", {"traceback": traceback.format_exc(), "time_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
