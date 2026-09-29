# CIGAS: final BagViT training and evaluation

Final-model training and evaluation module for **Concept-based artificial intelligence for transparent glaucoma detection from color fundus photographs**. CIGAS denotes the overall system; BagViT is its visual model. This repository covers the two selected experiments:

- `binarylabel_bagvit_7_mae_224_WA_loadsingle_tent`
- `multilabel_bagvit_7_mae_224_WA_loadsingle_tent_abnormal_head10` (12 labels)

This package includes model definitions, data loading, binary training, multilabel training initialized from the binary model, final TENT evaluation, and historical metric analysis. It does not include images, private annotations, trained weights or per-image research predictions.

## Setup

Use Python 3.12 (the locally validated interpreter). Install PyTorch for your hardware and the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python tests/smoke.py
```

The pinned versions describe the current validation environment, not a recovered historical training environment. CUDA training and a fresh dependency installation have not been validated in this packaging session. See [validation](docs/VALIDATION.md).

## Data and checkpoints

See [data format](docs/DATA.md) and [experiment provenance](docs/PROVENANCE.md). Obtain the original RETFound initialization checkpoint separately and provide its local filename. Publication/download URLs for the two trained BagViT checkpoints have not yet been supplied; their SHA256 hashes are recorded in `docs/provenance/checkpoints.json`.

All commands below run from this directory. Replace placeholder paths with local paths.

```bash
# Stage 1: binary training (local-mask attention, as in the available training source)
python scripts/run.py binary train --data-root /path/to/datasets \
  --pretrained /path/to/RETFound_mae_natureCFP.pth

# Stage 2: multilabel training, initialized from the selected binary checkpoint
python scripts/run.py multilabel train --data-root /path/to/datasets \
  --binary-checkpoint /path/to/checkpoint-best_7.pth

# Stage 3: TENT evaluation, window attention in both experiments
python scripts/run.py binary evaluate --data-root /path/to/datasets \
  --checkpoint /path/to/binary/checkpoint-best_7.pth
python scripts/run.py multilabel evaluate --data-root /path/to/datasets \
  --checkpoint /path/to/multilabel/checkpoint-best_10.pth
```

Outputs are stored under `outputs/<experiment>/`. `--output-dir`, `--num-workers`, `--device`, `--batch-size`, and training `--epochs` are configurable. Add `--dry-run` to inspect the resolved configuration without importing PyTorch. Set `CUDA_VISIBLE_DEVICES` to select one GPU. The supported wrapper runs one process, matching the recorded world size of 1.

TENT uses batch size 128, Adam learning rate 0.001, 5 steps, and per-batch episodic reset. Batch size/order affect adaptation and must be retained for result reproduction. Checkpoint epochs are stored zero-based; training selection may select different epochs on a different environment. The historical checkpoint files are model snapshots without optimizer state; `--checkpoint` during training restores those weights but cannot exactly resume their optimizer trajectory.

## Metric analysis

The scripts preserve the source analysis methods: each evaluated set selects its own Youden threshold; binary AUPRC uses trapezoidal integration and multilabel uses average precision. These are retrospective metrics, not validation-set-fixed operating thresholds. Bootstrap confidence intervals and optional paired DeLong statistics against explicitly provided predictions are included. See limitations in [provenance](docs/PROVENANCE.md).

```bash
export DATA_ROOT=/path/to/datasets
export PREDICTIONS_ROOT="$PWD/outputs"
EPOCH=7 python analysis/binary_metrics.py binarylabel_bagvit_7_mae_224_WA_loadsingle_tent
EPOCH=10 python analysis/multilabel_metrics.py multilabel_bagvit_7_mae_224_WA_loadsingle_tent_abnormal_head10
```

Optional `REFERENCE_TASK` and `REFERENCE_EPOCH` select a reference experiment under `PREDICTIONS_ROOT`. With no reference predictions, comparison columns are unavailable. Summaries are saved in `delong/`. The multilabel analysis includes per-label, macro, ACC and external subgroup results; binary analysis also includes external subgroups. No reference experiment is selected automatically; reference model training and ablation runners are not included. Binary uses 1000 bootstrap resamples and multilabel uses 500, as in their respective source workflows. Training-engine metrics and these post-hoc statistics have different threshold/aggregation rules.

## Contents and attribution

- `configs/`: checkpoint-derived hyperparameters and the verified 12-label order.
- `src/`: training, model, TENT, datasets and local utility dependencies.
- `analysis/`: portable historical statistical scripts.
- `examples/`: synthetic label schema only.
- `docs/provenance/`: source hashes, checkpoint hashes, training logs and validation evidence.
- `tests/`: synthetic CPU integration test.

The existing repository license is preserved in `LICENSE`; original RETFound/MAE copyright headers are retained. See `THIRD_PARTY_NOTICES.md`. Author citation metadata and public checkpoint locations remain to be supplied.

## Release scope

Only the final binary and 12-concept training/evaluation paths are included. LoRA, alternative backbones, model-depth variants, the old 11-concept loader, flip augmentation at test time and no-TENT evaluation commands are excluded. Both local-mask attention (binary training) and non-overlapping window attention (concept training and final TENT evaluation) are retained because the selected experiments require them. Their presence does not expose an ablation workflow.

The package is the BagViT module of CIGAS, not the full reporting/annotation system: Qwen report generation, expert annotation tools, full box-map processing and reader-study/baseline experiments are outside this release. Standard evaluation without TENT remains internally necessary for selecting checkpoints during training. Existing manuscript/method discrepancies are documented in `docs/PROVENANCE.md`; pruning does not resolve or silently change them.
