"""Tiny actual Qwen3 over a real local TCP socket; correctness, not latency evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import tempfile
from collections import Counter
from hashlib import sha256
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
import latency_smoke as smoke
from feedback_codec.hf_backend import HFTargetBackend, HFDraftBackend, _FrozenHFBackend
from feedback_codec.codec import CodecConfig
from h_vq64_head import HTaskHead
from message_codecs import METHODS, binding, restore_codec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("reference", "boundary", "kv"), default="reference")
    parser.add_argument("--out", type=Path, help="new directory for this run's receipts and rows")
    parser.add_argument("--codec", choices=METHODS, default="grouped_vq8")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--warmup-rounds", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.warmup_rounds < 0 or args.repeats < 1:
        parser.error("invalid TCP repetition count")
    torch.set_num_threads(1)
    torch.manual_seed(121)
    config = Qwen3Config(vocab_size=41, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=128, attention_dropout=0.,
                        pad_token_id=0, bos_token_id=1, eos_token_id=40)
    config._attn_implementation = "eager"
    target = HFTargetBackend(Qwen3ForCausalLM(config), object(), device="cpu")
    class Draft(_FrozenHFBackend):
        teacher_hidden = HFDraftBackend.teacher_hidden
    draft = Draft(Qwen3ForCausalLM(config), object(), device="cpu")
    head = HTaskHead(CodecConfig(32, 32, 41), draft.model.get_output_embeddings(),
                     draft_embedding=draft.model.get_input_embeddings(), codec_type=args.codec).eval()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["binding"]["codec_type"] != args.codec:
            raise ValueError("checkpoint method differs")
        head.message = restore_codec(checkpoint["binding"], checkpoint["message"])
        head.base.core.repair.U.weight.data.copy_(checkpoint["repair_U"])
    elif args.codec != "grouped_vq8":
        raise ValueError("new codecs require a fitted smoke checkpoint")
    head.message.eval()
    head.requires_grad_(False)
    if args.implementation == "reference":
        native = smoke.NativeDraft(draft, head)
    else:
        from cached_draft import CachedDraft
        native = CachedDraft(draft, head, mode=args.implementation)
    decoder_inputs = []
    def record_decoder_input(module, inputs, kwargs):
        decoder_inputs.append(kwargs["input_ids"].shape[-1])
    draft.model.model.register_forward_pre_hook(record_decoder_input, with_kwargs=True)
    contract = {"window": 8, "max_new": 16, "eos_token_id": 40, "draft_implementation": args.implementation}
    contract["codec_binding"] = binding(head.message)
    contract["source_sha256"] = json.loads((Path(__file__).parent / "variant_config.json").read_text())["contract"]["source_sha256"]
    contract["checkpoint_sha256"] = (sha256(args.checkpoint.read_bytes()).hexdigest()
                                     if args.checkpoint else "legacy-tiny-seed121")
    cloud_message = restore_codec(contract["codec_binding"], head.message.state_dict()).requires_grad_(False).eval()
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(head.message.parameters(), cloud_message.parameters()))
    roster = {}
    for i, prompt in enumerate(([1, 2], [1, 3, 4], [1, 6])):
        output = []
        for _ in range(contract["max_new"]):
            token = target.next(list(prompt) + output).token_id
            output.append(token)
            if token == 40:
                break
        roster[str(i)] = {"request_id": str(i), "prompt_token_ids": list(prompt),
                           "expected_target_token_ids": output}
    hooks = Counter()
    wire_hex = []
    def counted(owner, name):
        original = getattr(owner, name)
        def call(*args, **kwargs):
            hooks[name] += 1
            return original(*args, **kwargs)
        setattr(owner, name, call)
    for obj, method in [(head.message, "decode_payload"), (head.base.core.repair, "repair_from_control")]:
        counted(obj, method)
    if args.implementation != "reference":
        counted(native, "build_control")
    def encode(hidden, accepted, rejected, answer):
        table = draft.model.get_input_embeddings().weight
        hooks["encode_payload"] += 1
        payload = cloud_message.encode_payload(hidden.reshape(1, -1), accepted,
            None if rejected is None else table[rejected].reshape(1, -1), table[answer].reshape(1, -1))
        wire_hex.append(payload.hex())
        return payload
    session = smoke.CloudSession(target, encode, roster, contract, "cpu")
    session.begin({"arm": "raw", "request_id": "0", "trial_id": "bad", "warmup": True, "prompt_ids": [1, 2]})
    try:
        session.verify({"round": 0, "prefix_sha256": "tampered", "proposals": [1] * 8})
    except ValueError as error:
        assert str(error) == "edge/cloud prefix or round differs"
    else:
        raise AssertionError("tampered boundary accepted")
    session.current = None
    if args.out is None:
        root = Path(tempfile.mkdtemp(prefix="cpu-", dir=Path(__file__).parent))
    else:
        root = args.out.resolve()
        root.mkdir(parents=True, exist_ok=False)
    edge_root, cloud_root = root / "edge", root / "cloud"
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0));listener.listen(1);listener.settimeout(10)
    port = listener.getsockname()[1]
    def serve():
        channel, _ = listener.accept()
        with channel:
            channel.settimeout(10)
            wire = smoke.Wire(channel)
            while True:
                request, payload = wire.recv()
                assert not payload
                kind = request["type"]
                if kind == "hello":
                    assert request["contract_id"] == smoke.digest(contract)
                    assert binding(cloud_message) == contract["codec_binding"]
                    reply, feedback = {"contract_id": smoke.digest(contract)}, b""
                elif kind == "start":reply, feedback = session.begin(request)
                elif kind == "verify":reply, feedback = session.verify(request)
                elif kind == "finish_request":
                    row = session.finish_request()
                    smoke.write(cloud_root / "rows" / (smoke.digest(row["trial_id"]) + ".json"), row)
                    reply, feedback = {"type": "finished"}, b""
                elif kind == "shutdown":wire.send({"type": "shutdown"});break
                else:raise AssertionError(kind)
                wire.send(reply, feedback)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(serve)
        with socket.create_connection(("127.0.0.1", port), timeout=10) as channel:
            wire = smoke.Wire(channel)
            hello, _, _ = wire.rpc({"type": "hello", "contract_id": smoke.digest(contract)})
            assert hello["contract_id"] == smoke.digest(contract)
            for repeat in range(-args.warmup_rounds, args.repeats):
                for row in roster.values():
                    for arm in smoke.ARMS:
                        trial = f"repeat{repeat}-request{row['request_id']}-{arm}"
                        decoder_inputs.clear()
                        hooks.clear()
                        wire_hex.clear()
                        result = smoke.edge_request(wire, native, row, arm, trial, repeat < 0, contract)
                        assert result["output_ids"] == row["expected_target_token_ids"]
                        if args.implementation != "reference":
                            counters = result["draft_cache_counters"]
                            assert counters["decoder_calls"] == len(decoder_inputs) == result["timing"]["calls"]["draft_decoder_forward"]
                            assert counters["decoder_input_tokens"] == sum(decoder_inputs)
                        else:
                            calls = result["timing"]["calls"]
                            assert len(decoder_inputs) == calls["draft_native_forward"] + calls.get("edge_feedback_prepare", 0)
                        expected_feedback = (result["verification_rounds"] - 1) if arm != "raw" else 0
                        assert hooks["encode_payload"] == expected_feedback
                        if args.implementation != "reference":
                            assert hooks["decode_payload"] == hooks["build_control"] == expected_feedback
                            assert hooks["repair_from_control"] == counters.get("repair_projections", 0)
                        result["observed_calls"] = dict(hooks)
                        result["feedback_hex"] = wire_hex.copy()
                        result["observed_decoder_input_lengths"] = decoder_inputs.copy()
                        smoke.write(edge_root / "rows" / (smoke.digest(trial) + ".json"), result)
            wire.rpc({"type": "shutdown"})
        future.result(timeout=10)
    listener.close()
    smoke.summarize(edge_root, cloud_root, root / "summary.json", expected_contract=contract, expected_trials=9 * (args.warmup_rounds + args.repeats))
    result = {"status": "pass", "checks": ["real TCP framing", "tampered causal prefix rejected",
        "actual tiny Qwen3 and configured message codec", "Target greedy sequence matches independent AR",
        "SHADOW and RAW proposals identical", "cross-host round equality and paired summary"],
        "scope": "CPU tiny models over local TCP; no 4090/A100 timing claim", "artifacts": str(root)}
    smoke.write((root if args.out else Path(__file__).parent) / "cpu_receipt.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
