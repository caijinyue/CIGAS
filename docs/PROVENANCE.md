# Paper experiment provenance

The selected multilabel paper experiment is `multilabel_bagvit_7_mae_224_WA_loadsingle_tent_abnormal_head10`, using `checkpoint-best_10.pth` (epoch 10; head `[12,1024]`). Binary evaluation remains `binarylabel_bagvit_7_mae_224_WA_loadsingle_tent`, using `bagvit_7_20251014_dovs/checkpoint-best_7.pth` (head `[2,1024]`). The former 11-label experiment is not the paper's main multilabel configuration and is no longer selected by this package.

The 12th concept is `macula & background retinal lesion`. It is a separate concept, not the binary GON output. The submitted manuscript Table 3 defines 12 concepts. All 38 available concept/macro AUC point estimates in Table 2 match the local `delong_2026_0405` summary for the selected 12-label experiment. This identifies the matching experiment; it does not certify a full rerun or exact historical source revision. Submission documents are not distributed in this repository.

The checkpoint records `gon_ngon=True`, `win_attn=True`, `multi_label=True`, `train_head_only_epochs=10`. Training merges GON and NGON annotation tables under `label_multilabel_GON-NGON`; both groups can have positive concept labels. The selected checkpoint has no training TENT (`tta=None`). Evaluation uses window attention and episodic TENT (Adam, learning rate 0.001, 5 steps, batch 128). Binary training uses local-mask attention in the available source, while its TENT evaluation uses window attention. Weight shape compatibility cannot establish historical attention semantics.

Training configs are extracted from checkpoint arguments; both record 100 epochs, seed 0, input 224, batch 128, base LR 0.005, actual LR 0.0025, layer decay 0.65, weight decay 0.05, drop path 0.2. Current validation versions are not claimed to be historical training versions. Model snapshots lack optimizer state, preventing exact optimizer continuation.

## Portability changes

- Configurable data/output/checkpoint paths; explicit evaluation mode; local RETFound initialization.
- Current arguments are not overwritten wholesale by checkpoint arguments; external dataset arguments are copied.
- Evaluation skips redundant binary initialization; epoch suffix comes from checkpoint metadata.
- Fixed training prediction suffix formatting and numbered best-checkpoint lookup; zero best scores can save a checkpoint.
- Resume head freezing respects start epoch; forward errors propagate.
- CPU smoke support; ordered 12-label prediction headers and schema/duplicate validation for merged annotations.
- Matching 12-label statistical workflow comes from `2_multilabel_subset_youden_all_external_VF3.py`; numerical functions are preserved. Paths are configurable; the no-TENT analysis mode and automatic reference selection have been removed. Optional reference predictions must be supplied explicitly.

## Historical behaviors and limitations

Missing annotation columns/images default to zero in the inherited loader. Multilabel training uses resize/center-crop because the Boolean passed to the transform builder does not equal the string `train`; this behavior is preserved. Binary training uses its augmentation pipeline. Models and data paths have not been silently changed to fix historical methodology.

Post-hoc statistics choose a Youden threshold on each evaluated set and bootstrap individual images, not patients. Binary uses 1000 resamples; the selected multilabel source uses 500. Binary AUPRC is trapezoidal PR area; multilabel AUPRC is average precision. Multilabel analysis includes ACC, external subgroups, optional severity analysis, and optional paired DeLong against explicitly supplied predictions. Severity analysis and full paper-table reproduction have not been validated.

The submission audit found F1/AUPRC column swaps and external specificity row offsets in the manuscript table. The software keeps correctly named metric columns rather than reproducing document transcription errors. The current GON training annotation table has all 18,343 rows positive for the 12th concept; the NGON table has 3,126/16,006 positive. These local observations warrant checking annotation construction and do not prove historical data were identical. No labels are changed or distributed by this package.

Hashes, checkpoint metadata and training logs are in `provenance/`.

## Final-model-only release

Removed unused LoRA code, alternative backbone/model factories (DINOv2, BagNet/ResNet branches, BagViT36), the shadowed unused VisionTransformer class, the legacy 11-label loader, unused folder-inference branch, flip TTA and no-TENT evaluation/analysis modes. The fixed supported architecture is RETFound-initialized ViT-Large, window size 7. Both required attention implementations and standard training-time validation remain. Core retained architecture and numerical metric functions were compared against the pre-pruning source using AST equivalence. Optional paired statistical comparisons consume external predictions; no reference model is bundled.
