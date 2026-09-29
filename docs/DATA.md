# Data format

`--data-root` is the parent of `DOVS`, `SMDG`, and `REFUGE2`:

```text
datasets/
  DOVS/
    train/{0_NGON,1_GON}/
    val/{0_NGON,1_GON}/
    test/{0_NGON,1_GON}/
    external/                  # multilabel external images (recursive)
    REFUGE2/                   # multilabel REFUGE2 images (recursive)
    label_multilabel_GON-NGON/  # paired tables listed below
  SMDG/test/{0_NGON,1_GON}/      # binary external images
  REFUGE2/test/{0_NGON,1_GON}/   # binary REFUGE2 images
```

| Split | GON table stem | NGON table stem |
|---|---|---|
| train | train_GON_RY_20260105 | train_NGON_RY_20260211 |
| val | val_GON_RY_20260105 | val_NGON_RY20260211 |
| test | test_GON_RY_20260105 | test_NGON_RY20260211 |
| external | SMDG_GON_RY_20260105 | external_NGON_RY20260211 |
| REFUGE2 | REFUGE2_GON_RY_20260105 | REFUGE2_NGON_RY20260211 |

CSV is preferred, then XLSX, then XLS. XLS requires the optional `xlrd` dependency. Each table contains `image` and binary concept columns in `configs/labels.json`; the loader normalizes column names, aligns the 12 concepts, fills absent columns with zero, and concatenates the two tables. Both GON and NGON can have positive concepts. The synthetic schema is `examples/multilabel_labels.csv`.

Matching uses case-sensitive basenames, recursive sorted image order, and all-zero labels for unlisted images. Only use unlisted images if confirmed negative. Duplicate annotation/image basenames are rejected. Statistical analysis matches stems, so stems must also be unique. Keep the original splits and image order for reproduction; no splitting is performed here.

Binary labels are ImageFolder indices: `0_NGON`=0, `1_GON`=1. The entry points construct train/val/test/external datasets even in evaluation mode. All these directories must exist. No real images or labels are distributed.

Evaluation uses resize 256, center-crop 224 and ImageNet normalization. Historical multilabel training also uses this transform (see PROVENANCE.md). TENT depends on batch composition: retain batch size 128 for comparisons with original results.

Optional severity analysis expects `DOVS/DOVS_test_VF.csv`. This branch is inherited and has not been validated for release.
