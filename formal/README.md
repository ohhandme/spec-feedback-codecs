# Full RAW training and original 503-request verification

The phases below run independently. Inspect the checkpoints already on the copied A100 disk first. Nothing starts training automatically. The commands assume Python 3.11+, PyTorch, transformers with Qwen3 support, safetensors and Linux. Paths default to the original /data layout. Use --help to override their locations.

```sh
out=/data/spec-feedback-formal76
source_root=/data/autoresearch/classic-261114-task59-fdbf2f8a/matrix-fc65585b/continuation-v5
python formal/run.py inspect-existing --existing "$source_root" --group H-VQ64-TASK --method grouped_vq8 --out "$out"
python formal/run.py inspect-base --method grouped_vq8 --out "$out"
```

| Archived candidate | What the loader verifies |
| --- | --- |
| H-VQ64-TASK | Recognized v4/v5 A100 config and runtime hashes, completed 3986-request training, original 503 selection, exact parameter keys, shapes, dtypes, values and selected file hash. The historical batch size was one window and the training terminal has 783440 updates. The production checkpoint must exist on the execution machine. |
| H-PCAQ64-W | Incompatible with pca_int8. The old encoder consumes H alone and produces unsigned bin IDs for a codebook. The requested method consumes H plus protocol features and uses symmetric signed coordinates with a linear decoder. |
| C-secondary epoch 1 | Common repair only. The native checkpoint contains a model dictionary and optimizer/RNG metadata. inspect-base validates the original file and training provenance and loads its repair tensors. |
| Old 4090 grouped-VQ smoke checkpoint | Different origin. It cannot satisfy the A100 production checkpoint binding. |
| MLP-INT8 | Requires a matching trained encoder, scales and decoder. A grouped codebook or an untrained initialization cannot supply those parameters. |

For an admitted historical full checkpoint, evaluate the original 503 directly. No preparation or training phase is needed. Set CUDA_VISIBLE_DEVICES to the assigned physical device or UUID. On the designated new server GPU1 is assigned to this job.

```sh
CUDA_VISIBLE_DEVICES=1 python formal/run.py eval-existing --existing "$source_root" --group H-VQ64-TASK --method grouped_vq8 --out "$out"
```

Only for a method confirmed missing or incompatible, run prepare once and explicitly choose that method for supplemental training. The example uses mlp_int8. The other supported names are pca_int8 and grouped_vq8.

```sh
python formal/run.py prepare --out "$out"
CUDA_VISIBLE_DEVICES=1 python formal/run.py train --out "$out" --method mlp_int8
CUDA_VISIBLE_DEVICES=1 python formal/run.py eval --out "$out" --method mlp_int8
```

Preparation uses the original index, Draft cache, RAW shards and 503 roster. It checks each cached H against the recorded final-norm bytes and the prefix that produced the correction or bonus. It accepts either original wave2 roster503.json or the pinned development503_existing.jsonl. It reads shard contents only for training and calibration requests. Expect about 4.5 GB of packed BF16 data and enough working RAM for serialization. The repair head and Draft stay frozen. Each supplemental codec sees 156688 windows and 1217488 valid labels per epoch, with seed 42, five epochs, 4897 updates per epoch, 32 windows per update and 16 in the last update. CE averages tokens within a window and then averages windows. PCA fits mean, axes and max-absolute scales on all training rows using blocked FP64 centered covariance, then freezes them. The 503-request calibration metric averages within each request before averaging requests. Ties select the earliest epoch.

Evaluation runs an actual local TCP connection, eager native Draft and Target in one process on the assigned GPU. It keeps each request's RAW, SHADOW and ACTIVE records, compares every emitted token to the frozen oracle and requires identical RAW/SHADOW proposals and commits. The acceptance measure is total accepted Draft tokens divided by verification rounds. The output records feedback bytes, framed bytes and timing. Timing describes same-device loopback. On interruption reissue the same command and output directory. Supplemental epochs save codec parameters and buffers and refer to the immutable C checkpoint for their frozen repair weights.

## CPU verification

```sh
python formal/test_formal.py
```

Local tests covered three methods, deterministic restart, frozen repair tensors, batched versus actual wire-decoded logits, full-train PCA versus SVD on a small fixture, 27 tiny Qwen3 TCP trials, incorrect source and tensor types, and teacher token ties. The original sealed index and roster were also checked locally, along with 28 real RAW windows using original H and a separately regenerated 0.6B CPU Draft cache. These checks do not measure production acceptance or GPU speed. The actual remote trained files and the entire GPU path need verification on the owning machine.
