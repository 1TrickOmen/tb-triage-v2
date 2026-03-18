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
