"""Compare saved CPU FP32 and BF16 codec behavior across every held-out request."""
import argparse
import copy
import json
from pathlib import Path

from train_codecs import METHODS, dataset, setup, restore_codec, torch


def check(method, root):
    target, draft, head = setup(method)
    data = torch.load(root / (method + ".pt"), weights_only=True)
    fp32 = restore_codec(data["binding"], data["message"])
    bf16 = copy.deepcopy(fp32).bfloat16()
    results = []
    for _, key, x, *_ in dataset(target, draft, head, "validation", range(40, 48)):
        b0 = fp32.encode_payload(*x)
        b1 = bf16.encode_payload(*x)
        controls = [m.decode_payload(b) for m, b in [(fp32, b0), (bf16, b1)]]
        errors = [float((m.train_control(*x).float() - c.float()).abs().max())
                  for m, c in zip([fp32, bf16], controls)]
        assert errors == [0., 0.]
        results.append({"request": key, "fp32_bytes": b0.hex(), "bf16_bytes": b1.hex(),
                        "identical_coordinates": sum(i == j for i, j in zip(b0, b1)),
                        "wire_control_error_fp32_bf16": errors,
                        "max_cross_dtype_control_difference": float((controls[0]-controls[1].float()).abs().max())})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root
    torch.set_num_threads(1)
    with torch.no_grad():
        results = {m: check(m, root) for m in METHODS}
    (root / "precision.json").write_text(json.dumps(results, indent=2))
    print("precision passed", {m: len(rows) for m, rows in results.items()})
