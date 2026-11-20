# Codec development screening on 200 requests

Run from this checkout on the assigned GPU at a phase boundary. Use the same output root and any model, manifest, base or roster path overrides from the completed dc6 training command.

```sh
CUDA_VISIBLE_DEVICES=1 python eval200/evaluate.py --method pca_int8 --out "$TRAIN_OUT"
```

Use mlp_int8 or grouped_vq8 for the other methods. For a validated historical grouped checkpoint, pass the original runtime directory with --existing. The default evaluates all three arms on the first 200 requests of the original, ordered 503-request development roster. Training and checkpoint selection retain the full dc6 splits and budgets.

The ordered IDs are fixed in request_ids.txt. Its SHA-256 is 8012363ce8a15d0b3330ecba6cae520877cdf335c58e389b9026d7f0fbc6aeb6 (ASCII, one ID per line, LF including the final line). Each method writes a separate eval200 directory. contract.json binds the cohort, checkpoint, parent code digest and evaluator SHA-256. completion.json reports 200 requests plus token and verification counts by arm. Repeating a command validates and resumes its own rows. To import matching measurements from the sibling completed eval503 archive, explicitly add --reuse-complete-503. For an unfinished sibling eval503 run, use --reuse-sealed-503 instead. A request qualifies when its RAW, SHADOW and ACTIVE files pass token, round and feedback checks and the RAW and SHADOW trajectories match. Missing arms leave the request for normal evaluation. Progress counters do not establish eligibility. Imported rows retain the original file and contract digests. The original 503 files remain available for a later full comparison.

Run the focused CPU verification with python eval200/test_eval.py /path/to/roster503.json. The test uses a real tiny Qwen3 TCP trial and a dc6-format fixture checkpoint. Formal model outcomes require the assigned GPU run.
