# Validation record

Validation completed on 2026-09-24 using the existing `retfound` environment, Python 3.12. Package versions are recorded in `provenance/validation_environment.json`.

Passed:

1. Python compilation and configuration dry runs.
2. Independent copy at a separate temporary path: binary and multilabel one-epoch training with a small timm ViT and synthetic images; checkpoint save/reload; five-step episodic TENT evaluation for val/test/external/REFUGE2; finite predictions, expected row counts, and ordered 12-label CSV headers.
3. Full-size original binary epoch-7 and multilabel epoch-10 checkpoints: strict state-dictionary loading with window attention, CPU forward on a synthetic 224x224 input, expected output shapes `[1,2]` and `[1,12]`, finite logits. See `provenance/weight_validation.json`.
4. Synthetic binary and multilabel statistical calculations including 1000-resample binary and 500-resample multilabel confidence intervals; perfect separation yields 100% AUC/sensitivity/specificity; paired identical scores yield DeLong p=1.
5. AST comparison confirms the four historical numerical/statistical functions and the merged-label analysis loader are unchanged from the source. See `provenance/statistics_validation.json`.
6. Distribution checks: regular files only, no weights/images/private label sheets, no original machine paths in Python sources, and a SHA256 inventory.

Commands to rerun:

```bash
python -m compileall -q src scripts analysis tests
python tests/smoke.py
python tests/metrics_smoke.py
python scripts/check_checkpoint.py /path/to/binary/checkpoint-best_7.pth --classes 2
python scripts/check_checkpoint.py /path/to/multilabel/checkpoint-best_10.pth --classes 12
```

Not performed: GPU training, full-size model backward/TENT benchmarking, complete 100-epoch training, full-dataset inference, paper-table reproduction, a clean pip environment installation, or GitHub upload. The current session could not access the NVIDIA driver. The small-model integration test validates control flow, not numerical equivalence of a new full-size training run. Checkpoint loading validates compatibility, not historical attention semantics. Public checkpoint URLs and paper citation details remain to be supplied.

The selected multilabel checkpoint is now `multilabel_bagvit_7_mae_224_WA_loadsingle_tent_abnormal_head10/checkpoint-best_10.pth`. The synthetic integration test uses separate GON and NGON annotation files and includes positive 12th-concept labels in NGON images.

After final-model pruning, the synthetic training and TENT evaluation tests and metric tests were rerun. Retained architecture functions were checked for AST equivalence against the pre-pruning implementation.
