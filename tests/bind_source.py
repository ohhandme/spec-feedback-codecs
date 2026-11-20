"""Seal the reviewed legacy production checkpoint to the multi-codec source tree."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
sys.path.insert(0, str(DEMO))
from message_codecs import make_codec, binding
import torch


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    args = ap.parse_args()
    config_path = DEMO / "variant_config.json"
    config = json.loads(config_path.read_bytes())
    contract = config["contract"]
    if sha(args.checkpoint) != contract["checkpoint_sha256"]:
        raise ValueError("legacy checkpoint changed")
    state = torch.load(args.checkpoint, weights_only=True, map_location="cpu")["model"]
    m = make_codec("grouped_vq8", 4096, 1024)
    m.load_state_dict({k.removeprefix("message."): v for k, v in state.items() if k.startswith("message.")})
    contract["codec_binding"] = binding(m)
    contract["source_sha256"] = {p.relative_to(DEMO).as_posix(): sha(p) for p in sorted(DEMO.rglob("*.py"))}
    contract["parent_review_contract"] = "e99bc4710666ea02b4400bdcb3434d931085418c15b850fd71e6c8593664409b"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (DEMO / "variant_manifest.json").write_text(json.dumps({"variant_contract_id": digest, "contract": contract}, indent=2))
    print(digest)
