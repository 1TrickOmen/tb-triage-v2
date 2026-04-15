# Colab baseline workflow

This is the smallest practical path to train the current MobileNetV2 TB baseline on a Colab GPU.

## What you need
Upload or make available **one of these two data layouts** inside the repo:

### Option A — easiest: already-prepared processed data
Use this if you already have the repo's repaired data plumbing locally and want the fastest Colab run.

Required paths:
- `data/processed/merged_metadata.csv`
- `data/processed/extracted/chest_xray/...`
- `data/processed/extracted/tbx11k/...`

### Option B — raw archives only
Use this if you only have the original archives and want Colab to rebuild metadata before training.

Required paths:
- `data/raw/chest-xray/chest-xray-masks-and-labels-DatasetNinja.tar`
- `data/raw/tbx11k/tbx11k-DatasetNinja.tar`

The script will extract images into:
- `data/processed/extracted/chest_xray/...`
- `data/processed/extracted/tbx11k/...`

## Colab setup
Open a GPU runtime in Colab, then run:

```bash
%cd /content
!git clone <YOUR-REPO-URL> tb-triage-v2
%cd /content/tb-triage-v2
!pip install -r requirements-colab.txt
```

If your data already lives in Google Drive, mount it and copy it into the repo layout:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Path 1 — train from prepared processed data
Copy your prepared `data/processed` directory into the cloned repo so these exist:
- `/content/tb-triage-v2/data/processed/merged_metadata.csv`
- `/content/tb-triage-v2/data/processed/extracted/...`

Then run:

```bash
!python scripts/colab_train_baseline.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/merged_metadata.csv \
  --output-dir experiments/colab-baseline \
  --epochs 15 \
  --batch-size 32 \
  --image-size 256 \
  --class-weight none
```

## Path 2 — train from raw tar archives
Copy the two tar files into:
- `/content/tb-triage-v2/data/raw/chest-xray/chest-xray-masks-and-labels-DatasetNinja.tar`
- `/content/tb-triage-v2/data/raw/tbx11k/tbx11k-DatasetNinja.tar`

Then run:

```bash
!python scripts/colab_train_baseline.py \
  --repo-root /content/tb-triage-v2 \
  --rebuild-metadata \
  --tbx11k-tar data/raw/tbx11k/tbx11k-DatasetNinja.tar \
  --chest-xray-tar data/raw/chest-xray/chest-xray-masks-and-labels-DatasetNinja.tar \
  --metadata-csv data/processed/merged_metadata.csv \
  --output-dir experiments/colab-baseline \
  --epochs 15 \
  --batch-size 32 \
  --image-size 256 \
  --class-weight none
```

## Outputs
After training, Colab writes:
- `experiments/colab-baseline/mobilenetv2_baseline.keras`
- `experiments/colab-baseline/metrics.json`
- `experiments/colab-baseline/history.json`

## Notes
- The baseline trainer is still **classification only**, which is fine for the segmentation value test: the masked variant is created by writing a second metadata CSV whose `image_path` points at lung-masked image copies.
- `merged_metadata.csv` may contain repo-relative image paths. The trainer now resolves those relative to the repo root automatically, which is what makes the Colab path sane.
- This baseline still loads images into memory before training. It is fine for the current baseline scale on normal Colab RAM, but it is not yet the final streaming pipeline.
- The training loop now relies on the Keras generator's native length instead of a manual `steps_per_epoch`, which avoids the intermittent "input ran out of data" warning seen in Colab.
- To bias the loss toward TB recall on imbalanced data, rerun with `--class-weight balanced`.
- If you hit RAM pressure, first reduce `--batch-size` to `16`.

## Next experiment: class-weighted rerun
From `/content/tb-triage-v2`, run:

```bash
!python scripts/colab_train_baseline.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/merged_metadata.csv \
  --output-dir experiments/colab-baseline-class-weighted \
  --epochs 15 \
  --batch-size 32 \
  --image-size 256 \
  --class-weight balanced
```

## Threshold analysis for TB recall tradeoffs
After the class-weighted run finishes, analyze TB probability thresholds from the saved artifacts:

```bash
!python scripts/colab_analyze_thresholds.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/merged_metadata.csv \
  --run-dir experiments/colab-baseline-class-weighted
```

That writes:
- `experiments/colab-baseline-class-weighted/threshold_analysis/threshold_metrics.csv`
- `experiments/colab-baseline-class-weighted/threshold_analysis/test_predictions.csv` (only regenerated if an older run did not already save predictions)

To pick the most precise threshold that still reaches a recall target, for example `0.90`:

```bash
!python scripts/colab_analyze_thresholds.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/merged_metadata.csv \
  --run-dir experiments/colab-baseline-class-weighted \
  --target-recall 0.90
```

To sweep a smaller custom threshold set:

```bash
!python scripts/colab_analyze_thresholds.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/merged_metadata.csv \
  --run-dir experiments/colab-baseline-class-weighted \
  --thresholds 0.10,0.20,0.30,0.40,0.50,0.60
```

## Segmentation value test — masked-input variant
The clean comparison is:
1. train the current class-weighted raw-image baseline
2. generate lung masks for the same metadata rows
3. write a masked-image metadata CSV
4. rerun the same classifier recipe against that masked metadata
5. rerun the same threshold analysis on the masked run

### What is assumed here
- You already have or will provide a saved lung segmentation model, for example `artifacts/lung_segmentation.keras` or a Kaggle-exported `best_model.keras`.
- This repo does **not** yet train that segmentation model in Colab.
- `scripts/colab_predict_lung_masks.py` now loads common Kaggle-style custom segmentation objects automatically (`dice_coefficient`, `dice_coef`, `dice_loss`, `bce_dice_loss`, `jaccard_index`, `iou`, `iou_score`, `jaccard_loss`), so a typical U-Net `.keras` export should load without manual notebook surgery.
- The new scripts here only handle:
  - predicting/consuming masks
  - materializing masked classifier inputs
  - keeping the classifier and threshold-analysis flow identical

### Step 1 — predict lung masks for the merged dataset
From `/content/tb-triage-v2`, run:

```bash
!python scripts/colab_predict_lung_masks.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/merged_metadata.csv \
  --segmentation-model artifacts/lung_segmentation.keras \
  --output-masks-dir data/processed/predicted_lung_masks \
  --output-metadata-csv data/processed/merged_metadata_with_predicted_masks.csv \
  --image-size 512 \
  --threshold 0.5
```

If your segmentation artifact is the Kaggle U-Net export named `best_model.keras`, just point `--segmentation-model` at that file instead.

That writes:
- `data/processed/predicted_lung_masks/*.png`
- `data/processed/merged_metadata_with_predicted_masks.csv`

If you already have masks from somewhere else and they are named by `image_id` stem, you can skip mask prediction and point the next step at `--masks-dir` instead.

### Step 2 — create masked classifier inputs
If you used the prediction step above:

```bash
!python scripts/colab_prepare_masked_metadata.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/merged_metadata_with_predicted_masks.csv \
  --output-images-dir data/processed/masked_images \
  --output-metadata-csv data/processed/merged_metadata_masked.csv
```

If you already have masks in a directory and want to consume them directly:

```bash
!python scripts/colab_prepare_masked_metadata.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/merged_metadata.csv \
  --masks-dir data/processed/predicted_lung_masks \
  --output-images-dir data/processed/masked_images \
  --output-metadata-csv data/processed/merged_metadata_masked.csv
```

That writes:
- `data/processed/masked_images/*.png`
- `data/processed/merged_metadata_masked.csv`

By default this step **fails** if any row is missing a usable mask, because otherwise the masked experiment quietly turns into a mixed raw+masked mess. If you intentionally want fallback behavior for debugging only, add `--allow-missing-masks`.

### Step 3 — train the masked-input classifier with the same recipe

```bash
!python scripts/colab_train_baseline.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/merged_metadata_masked.csv \
  --output-dir experiments/colab-baseline-masked-class-weighted \
  --epochs 15 \
  --batch-size 32 \
  --image-size 256 \
  --class-weight balanced
```

### Step 4 — run the same threshold analysis on the masked run

```bash
!python scripts/colab_analyze_thresholds.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/merged_metadata_masked.csv \
  --run-dir experiments/colab-baseline-masked-class-weighted \
  --target-recall 0.90
```

### Minimal apples-to-apples comparison set
Compare these artifacts side by side:
- raw baseline metrics: `experiments/colab-baseline-class-weighted/metrics.json`
- raw threshold sweep: `experiments/colab-baseline-class-weighted/threshold_analysis/threshold_metrics.csv`
- masked baseline metrics: `experiments/colab-baseline-masked-class-weighted/metrics.json`
- masked threshold sweep: `experiments/colab-baseline-masked-class-weighted/threshold_analysis/threshold_metrics.csv`

## Source-held-out validation across the built-in sources
This is the next disciplined experiment if you want a real generalization read instead of another comfy in-distribution score.

### What this means here
For each run, choose one source as the **true unseen test source** and train only on the remaining sources:
- hold out `montgomery`
- hold out `shenzhen`
- hold out `tbx11k`

The new helper writes two CSVs:
- an **experiment metadata CSV** containing only seen-source `train` / `val` / `test` rows plus held-out rows marked `external_eval`
- a **holdout-only CSV** for the true unseen source

The trainer now honors explicit metadata splits when `experiment_split` is present, so it no longer re-randomizes the data and quietly wrecks the source-holdout design.

If your repo clone still has the repaired merged CSV at `data/merged_metadata.csv` instead of `data/processed/merged_metadata.csv`, the new prep helper will fall back to that legacy path automatically. Tiny mercy.

### Important assumptions
- `source_dataset=montgomery` and `source_dataset=shenzhen` are inferred from the DatasetNinja chest X-ray archive filename prefixes (`MCUCXR_` and `CHNCXR_`).
- `source_dataset=tbx11k` is a single source bucket. This ignores any hidden site/hospital heterogeneity inside TBX11K because the current repo metadata does not expose finer-grained provenance.
- For the source-held-out experiment, only rows with `include_for_training=true` and `label_final in {Normal, TB}` are eligible. That means TBX11K rows already excluded from the binary baseline (`sick_but_non-tb`, `uncertain_tb`, and TBX11K's own archive `test` split) stay excluded.
- The seen-source `test` split is still useful as a sanity check, but the real headline number should come from the held-out-source evaluation.

### Step 1 — prepare one source-held-out split
Example: hold out `shenzhen` as the true unseen source.

```bash
!python scripts/colab_prepare_source_holdout.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/merged_metadata.csv \
  --holdout-source shenzhen
```

That writes:
- `data/processed/source_holdout/shenzhen_experiment_metadata.csv`
- `data/processed/source_holdout/shenzhen_holdout_only.csv`
- `data/processed/source_holdout/shenzhen_summary.json`

### Step 2 — train the current class-weighted MobileNetV2 baseline on seen sources only

```bash
!python scripts/colab_train_baseline.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/source_holdout/shenzhen_experiment_metadata.csv \
  --output-dir experiments/source_holdout/shenzhen_class_weighted \
  --epochs 15 \
  --batch-size 32 \
  --image-size 256 \
  --class-weight balanced
```

This run uses only the explicit seen-source `train` / `val` / `test` rows from the prepared metadata.

## Recovery run after the collapsed Shenzhen DenseNet result
This is the clean next ablation the repo now supports directly. It keeps the honest Shenzhen source-held-out protocol, drops aggressive sample weighting, uses DenseNet121, and unfreezes only the tail of the backbone instead of freezing everything or flinging the whole network wide open.

### Why this is the next sane run
- same **Shenzhen holdout** truth-serum split
- **DenseNet121** stays, because MobileNet already looked too flimsy
- **no source-balanced sample weights** for the first recovery run
- **partial unfreezing** via `--trainable-fraction` instead of a binary frozen/unfrozen toggle
- **low learning rate**
- **mild augmentation** only

### Step 1 — prepare the Shenzhen held-out metadata
```bash
!python scripts/colab_prepare_source_holdout.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/merged_metadata.csv \
  --holdout-source shenzhen
```

### Step 2 — run the recovery experiment
```bash
!python scripts/colab_train_baseline.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/source_holdout/shenzhen_experiment_metadata.csv \
  --output-dir experiments/source_holdout/shenzhen_densenet121_partial_unfreeze_lr3e5 \
  --architecture densenet121 \
  --epochs 20 \
  --batch-size 32 \
  --image-size 256 \
  --learning-rate 3e-5 \
  --trainable-fraction 0.25 \
  --class-weight none \
  --augmentation mild
```

Recommended reading of this setup:
- `--trainable-fraction 0.25` = unfreeze roughly the last quarter of the DenseNet backbone
- `--class-weight none` = do **not** repeat the aggressive weighting path yet
- `--augmentation mild` = preserve the rollback away from the bad strong-augmentation recipe

### Step 3 — analyze the seen-source internal test split
```bash
!python scripts/colab_analyze_thresholds.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/source_holdout/shenzhen_experiment_metadata.csv \
  --run-dir experiments/source_holdout/shenzhen_densenet121_partial_unfreeze_lr3e5 \
  --architecture densenet121 \
  --target-recall 0.90
```

### Step 4 — evaluate on the true unseen Shenzhen holdout
Start with `0.40` for continuity, then compare against the threshold selected in Step 3.

```bash
!python scripts/colab_eval_external.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/source_holdout/shenzhen_holdout_only.csv \
  --run-dir experiments/source_holdout/shenzhen_densenet121_partial_unfreeze_lr3e5 \
  --architecture densenet121 \
  --threshold 0.40
```

### Step 3 — analyze the seen-source internal test split

```bash
!python scripts/colab_analyze_thresholds.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/source_holdout/shenzhen_experiment_metadata.csv \
  --run-dir experiments/source_holdout/shenzhen_class_weighted \
  --target-recall 0.90
```

### Step 4 — evaluate on the true unseen held-out source
Use the selected operating threshold from Step 3. If you want to stay consistent with the current baseline recommendation until you resweep, use `0.40` first.

```bash
!python scripts/colab_eval_external.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/source_holdout/shenzhen_holdout_only.csv \
  --run-dir experiments/source_holdout/shenzhen_class_weighted \
  --threshold 0.40
```

That writes the true unseen-source outputs under:
- `experiments/source_holdout/shenzhen_class_weighted/external_eval/shenzhen_holdout_only/`

### Next disciplined experiment: source-balanced normal-robustness variant
This is the smallest practical change for the current failure mode: keep MobileNetV2 fixed, keep the source-held-out protocol fixed, but rebalance the **training loss mass** so dominant source/class groups stop drowning out minority normal groups.

The helper below writes a new metadata CSV with a `sample_weight` column. In the recommended `source_label` mode, every `(source_dataset, label_final)` training group gets equal total weight mass. That means, for example, a rare `montgomery::Normal` group no longer counts for far less than a giant `tbx11k::Normal` group.

### Recommended run: hold out one source, then weight the seen-source training rows
Example: hold out `shenzhen`, then train the source-balanced variant.

```bash
!python scripts/colab_prepare_source_holdout.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/merged_metadata.csv \
  --holdout-source shenzhen

!python scripts/colab_prepare_source_balanced_metadata.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/source_holdout/shenzhen_experiment_metadata.csv \
  --output-metadata-csv data/processed/source_holdout/shenzhen_experiment_metadata_source_balanced.csv \
  --balance-mode source_label

!python scripts/colab_train_baseline.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/source_holdout/shenzhen_experiment_metadata_source_balanced.csv \
  --output-dir experiments/source_holdout/shenzhen_source_balanced \
  --epochs 15 \
  --batch-size 32 \
  --image-size 256 \
  --class-weight none

!python scripts/colab_analyze_thresholds.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/source_holdout/shenzhen_experiment_metadata_source_balanced.csv \
  --run-dir experiments/source_holdout/shenzhen_source_balanced \
  --target-recall 0.90

!python scripts/colab_eval_external.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/source_holdout/shenzhen_holdout_only.csv \
  --run-dir experiments/source_holdout/shenzhen_source_balanced \
  --threshold 0.40
```

Notes:
- Use `--class-weight none` here. The trainer now accepts metadata `sample_weight`, and stacking class weights on top would double-count the biasing.
- The training metrics JSON now records `sample_weight_summary`, including total loss mass per `(source,label)` group, so you can verify the rebalance actually happened instead of trusting vibes.
- If you want a narrower ablation later, `--balance-mode normal_by_source` only rebalances Normal rows across sources and leaves TB rows at weight 1.0.

### Repeat for the other two sources
#### Hold out Montgomery
```bash
!python scripts/colab_prepare_source_holdout.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/merged_metadata.csv \
  --holdout-source montgomery

!python scripts/colab_prepare_source_balanced_metadata.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/source_holdout/montgomery_experiment_metadata.csv \
  --output-metadata-csv data/processed/source_holdout/montgomery_experiment_metadata_source_balanced.csv \
  --balance-mode source_label

!python scripts/colab_train_baseline.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/source_holdout/montgomery_experiment_metadata_source_balanced.csv \
  --output-dir experiments/source_holdout/montgomery_source_balanced \
  --epochs 15 \
  --batch-size 32 \
  --image-size 256 \
  --class-weight none

!python scripts/colab_eval_external.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/source_holdout/montgomery_holdout_only.csv \
  --run-dir experiments/source_holdout/montgomery_source_balanced \
  --threshold 0.40
```

#### Hold out TBX11K
```bash
!python scripts/colab_prepare_source_holdout.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/merged_metadata.csv \
  --holdout-source tbx11k

!python scripts/colab_prepare_source_balanced_metadata.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/source_holdout/tbx11k_experiment_metadata.csv \
  --output-metadata-csv data/processed/source_holdout/tbx11k_experiment_metadata_source_balanced.csv \
  --balance-mode source_label

!python scripts/colab_train_baseline.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/source_holdout/tbx11k_experiment_metadata_source_balanced.csv \
  --output-dir experiments/source_holdout/tbx11k_source_balanced \
  --epochs 15 \
  --batch-size 32 \
  --image-size 256 \
  --class-weight none

!python scripts/colab_eval_external.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/source_holdout/tbx11k_holdout_only.csv \
  --run-dir experiments/source_holdout/tbx11k_source_balanced \
  --threshold 0.40
```

### Recommended comparison table to fill in after the runs
Do not average this away too early. Keep per-source results visible.

| Held-out source | Seen-source train rows | Held-out test rows | Threshold used | Held-out recall (TB) | Held-out precision (TB) | Held-out specificity | AUROC | PR AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| montgomery | 5327 | 138 | 0.40 or selected | TODO | TODO | TODO | TODO | TODO |
| shenzhen | 4707 | 758 | 0.40 or selected | TODO | TODO | TODO | TODO | TODO |
| tbx11k | 896 | 4569 | 0.40 or selected | TODO | TODO | TODO | TODO | TODO |

## External validation on the Pakistan Mendeley dataset
This repo now has the minimal path for **external-only** evaluation of a saved classifier. It does **not** mix the external dataset into training: the generated metadata marks every row with `include_for_training=false` and `is_external_test=true`.

### Acquisition assumption
The Mendeley page is visible, but direct scripted download details were not reliably exposed from the page fetch. So the code assumes you will manually download and extract the dataset into the repo, for example under:

- `/content/tb-triage-v2/data/external/mendeley_pakistan/`

If the extracted dataset has obvious class folders such as `TB/` and `Normal/`, the metadata builder should infer labels automatically. If the naming is weird, pass `--tb-dir` and `--normal-dir` explicitly.

### Step 1 — prepare external metadata
If the extracted folder already contains clearly named TB and Normal subfolders:

```bash
!python scripts/colab_prepare_external_mendeley_metadata.py \
  --repo-root /content/tb-triage-v2 \
  --dataset-root data/external/mendeley_pakistan \
  --output-csv data/processed/mendeley_pakistan_metadata.csv
```

If the dataset uses different folder names, be explicit, for example:

```bash
!python scripts/colab_prepare_external_mendeley_metadata.py \
  --repo-root /content/tb-triage-v2 \
  --dataset-root data/external/mendeley_pakistan \
  --tb-dir Tuberculosis \
  --normal-dir Normal \
  --output-csv data/processed/mendeley_pakistan_metadata.csv
```

### Step 2 — evaluate the saved class-weighted run at threshold 0.40
Assuming your best current run lives at `experiments/colab-baseline-class-weighted`:

```bash
!python scripts/colab_eval_external.py \
  --repo-root /content/tb-triage-v2 \
  --metadata-csv data/processed/mendeley_pakistan_metadata.csv \
  --run-dir experiments/colab-baseline-class-weighted \
  --threshold 0.40
```

That writes:
- `experiments/colab-baseline-class-weighted/external_eval/mendeley_pakistan_metadata/predictions.csv`
- `experiments/colab-baseline-class-weighted/external_eval/mendeley_pakistan_metadata/metrics.json`
- `experiments/colab-baseline-class-weighted/external_eval/mendeley_pakistan_metadata/confusion_at_threshold.csv`
- `experiments/colab-baseline-class-weighted/external_eval/mendeley_pakistan_metadata/threshold_metrics.csv`

### What this external eval reports
- full-dataset probability inference from the saved `.keras` model
- AUROC and PR AUC
- thresholded TB precision / recall / F1 at `0.40`
- thresholded confusion counts (`tn`, `fp`, `fn`, `tp`)
- optional threshold sweep using the same grid as the internal threshold analysis
