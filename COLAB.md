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
- The current baseline is **classification only**. U-Net segmentation is scaffolded in the repo but not yet wired into a Colab training flow here.
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
