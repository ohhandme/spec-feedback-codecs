"""Load the original A100 matrix checkpoint without renaming or casting trained tensors."""

from pathlib import Path

import torch
from raw_data import INDEX, MODELS, read, require, sha

BASE_C = "6edc4cd191daf7123b8407ee2876f2164dd1c4038bc20ae00268225157eaa91e"
C_CONFIG = "66d9858d62cc8178c1e450ae207c908d987a62f493293e7b6ee94df838acc397"
C_FREEZE = "0cc024f62d40d1c6786a6b71d8e38bc8930dfbe7e098f007076021adb75ff728"
C_COMPLETION = "adce98d3f306ed0115cd7a5fef663b5e7966ff4902608124dce855f56981df36"


def base_only(path):
    require(sha(path) == BASE_C, "not the original A100 C-secondary epoch1 checkpoint")
    folder = path.parent
    require(sha(folder / "completion.json") == C_COMPLETION, "original C completion differs")
    terminal = read(folder / "completion.json")
    require(
        terminal["selected"] == read(folder / "selected.json")
        and terminal["updates"] == 783440
        and terminal["config_sha256"] == C_CONFIG
        and terminal["runtime_freeze_sha256"] == C_FREEZE
        and terminal["selected"]["secondary"]["sha256"] == BASE_C,
        "C training provenance differs",
    )
    obj = torch.load(path, map_location="cpu", weights_only=True)
    require(
        obj["config_sha256"] == C_CONFIG
        and obj["runtime_freeze_sha256"] == C_FREEZE
        and obj["group"] == "C"
        and obj["epoch"] == 1
        and obj["position"] == 0
        and obj["updates"] == 156688,
        "wrong common C training state",
    )
    state = obj["model"]
    require(not any(k.startswith("message.") for k in state), "expected base-only C weights")
    return state


HISTORICAL = {
    "9fc8d7f1b72bf07c067fd4604aa823bd15f61e7f1197cc84463380a1e36ace96": "88cc5aef0cf8ff8864b2086580947c70fedd6258c10001d238ad1bb2941f717a",
    "0cc024f62d40d1c6786a6b71d8e38bc8930dfbe7e098f007076021adb75ff728": "0676abc2ed63afdbc94b67f00a76b0f6145a8ddacad04a5cd8f71ef9933907cf",
}


def check_schema(group, method):
    if group == "H-PCAQ64-W":
        raise ValueError(
            "historical PCA uses 4096 H-only features, unsigned bins and a codebook; "
            "pca_int8 uses conditioned 6145 features, signed scales and a learned linear decoder"
        )
    require(
        (group, method) == ("H-VQ64-TASK", "grouped_vq8"),
        "unsupported historical source architecture/objective for the requested codec",
    )


def load_exact(head, state):
    expected = head.trainable_state_dict()
    require(set(state) == set(expected), "source tensor membership differs")
    for key, dest in expected.items():
        value = state[key]
        require(
            isinstance(value, torch.Tensor)
            and value.dtype == dest.dtype
            and value.shape == dest.shape
            and value.isfinite().all(),
            "source dtype/shape/value differs: " + key,
        )
    head.load_trainable_state_dict(state)
    require(
        all(torch.equal(v, state[k].cpu()) for k, v in head.trainable_state_dict().items()),
        "trained weights changed during import",
    )


def historical(root, group, method, head):
    check_schema(group, method)
    freeze_path = root / "runtime_freeze.json"
    cfg_path = root / "configs" / (group + ".json")
    freeze_sha, cfg_sha = sha(freeze_path), sha(cfg_path)
    require(HISTORICAL.get(freeze_sha) == cfg_sha, "unrecognized production runtime/config pairing")
    cfg = read(cfg_path)
    files = read(freeze_path)["files"].values()
    require(
        INDEX in files
        and MODELS in files
        and cfg_sha in files
        and cfg["index_sha256"] == INDEX
        and cfg["CPU_fixture"] is False
        and cfg["evaluation_only"] is False
        and cfg["checkpoint_owner"] == cfg["group"] == group
        and cfg["loss"] == "CE"
        and cfg["position_weights"] == [1] * 8
        and cfg["training_windows"] == 156688
        and cfg["calibration_windows"] == 19736
        and cfg["epochs"] == 5
        and cfg["batch_size"] == 1
        and cfg["updates"] == 783440,
        "historical production training contract differs",
    )
    train_dir = root / "runs" / group / "train"
    terminal = read(train_dir / "completion.json")
    selected = read(train_dir / "selected.json")
    require(
        terminal["status"] == "complete"
        and terminal["CPU_fixture"] is False
        and terminal["config_sha256"] == cfg_sha
        and terminal["runtime_freeze_sha256"] == freeze_sha
        and terminal["group"] == group
        and terminal["updates"] == cfg["updates"]
        and terminal["selected"] == selected,
        "no matching completed A100 training",
    )
    best = selected["secondary"]
    require(
        best["rule"] == "minimum_calibration_request_equal_CE"
        and 1 <= best["epoch"] <= 5
        and best["updates"] == best["epoch"] * cfg["training_windows"]
        and best["position"] == 0,
        "not a complete validation-selected epoch",
    )
    path = train_dir / "checkpoints" / Path(best["path"]).name
    require(
        sha(path) == best["sha256"]
        and best["sha256"] != "8be64061de63ee7283d3de0d320843669019f636c96ff71667461b15e0eda563",
        "checkpoint SHA/source mismatch",
    )
    obj = torch.load(path, map_location="cpu", weights_only=True)
    require(
        obj["config_sha256"] == cfg_sha
        and obj["runtime_freeze_sha256"] == freeze_sha
        and obj["group"] == group
        and all(obj[k] == best[k] for k in ("epoch", "position", "updates")),
        "checkpoint provenance mismatch",
    )
    load_exact(head, obj["model"])
    return {
        "origin": "historical A100 matrix",
        "group": group,
        "selection": "secondary",
        "epoch": best["epoch"],
        "checkpoint_sha256": best["sha256"],
        "runtime_sha256": freeze_sha,
        "config_sha256": cfg_sha,
        "completion_sha256": sha(train_dir / "completion.json"),
        "selected_sha256": sha(train_dir / "selected.json"),
        "train_index": INDEX,
        "model_manifest": MODELS,
        "windows_per_epoch": 156688,
        "epochs": 5,
        "batch_windows": 1,
        "total_updates": 783440,
        "new_training_updates": 0,
    }
