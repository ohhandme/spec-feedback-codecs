"""Bind existing weights and source bytes on each host; no model/GPU execution."""
import argparse
from hashlib import sha256, file_digest
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", choices=("cloud", "edge"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    contract = config["contract"]
    actual = digest(config["checkpoint_path"])
    if contract.get("checkpoint_sha256"):
        assert actual == contract["checkpoint_sha256"], "trained checkpoint differs across hosts"
    contract["checkpoint_sha256"] = actual
    actual_manifest = digest(config[args.role + "_model_manifest"])
    assert actual_manifest == contract["model_manifest_sha256"], "model manifest differs from original A100/4090 pair"
    assert contract["roster_sha256"] == digest(HERE / "roster.jsonl")
    sources = {str(p.relative_to(HERE)).replace("\\", "/"): digest(p)
               for p in sorted(HERE.rglob("*.py")) if not p.relative_to(HERE).parts[0].startswith("cpu-")}
    if contract.get("source_sha256"):
        assert sources == contract["source_sha256"], "source differs across hosts"
    contract["source_sha256"] = sources
    args.out.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "sealed", "checkpoint_sha256": actual,
        "contract_sha256": sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "source_files": len(sources)}))


if __name__ == "__main__":
    main()
