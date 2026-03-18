# TB Triage V2

A reproducible tuberculosis chest X-ray triage prototype evolving from the MSc dissertation baseline into a trustworthy, externally validated decision-support system.

## Goal
Build a clinically credible prototype for TB triage from chest X-rays using:
- lung segmentation
- image classification
- explainability
- external validation
- disciplined experiment tracking

## Principles
- Reproducible over notebook chaos
- External validation over pretty internal metrics
- Recall/sensitivity over vanity accuracy
- Explainability for audit, not decoration
- Decision support, not autonomous diagnosis

## Initial Scope
### Datasets
- Montgomery
- Shenzhen
- TBX11K

### Models
- U-Net for lung segmentation
- MobileNetV2 baseline classifier
- DenseNet121 baseline classifier
- EfficientNet-B0 baseline classifier
- ResNet50 baseline classifier

### Core Experiments
1. Raw-image classifiers
2. Lung-masked / segmented classifiers
3. External evaluation on held-out source data
4. Grad-CAM review on correct and incorrect predictions
5. Calibration and threshold analysis

Current Colab baseline tooling now supports post-hoc TB threshold sweeps from saved probabilities so precision/recall/F1 and confusion tradeoffs can be reviewed explicitly, instead of pretending `0.50` is sacred.

It also now supports the minimal segmentation value test path: predict or consume lung masks, materialize masked classifier inputs into a second metadata CSV, then rerun the exact same MobileNetV2 training + threshold analysis flow on the masked variant.

And, crucially, it now has a minimal **source-held-out** path: prepare metadata that holds out one source (`montgomery`, `shenzhen`, or `tbx11k`) as a true unseen external test source, train only on the remaining sources, then evaluate the saved model on the held-out source separately. The trainer now respects explicit metadata split columns instead of silently reshuffling everything.

## Milestones
### M1 — Data Foundation
- [ ] Define metadata schema
- [ ] Ingest Montgomery
- [ ] Ingest Shenzhen
- [ ] Inspect and ingest TBX11K
- [ ] Deduplicate images
- [ ] Create train/val/test + external holdout splits

### M2 — Reproducible Baseline
- [ ] Reproduce dissertation MobileNetV2 baseline
- [ ] Reproduce U-Net baseline
- [ ] Save metrics, plots, and confusion matrices automatically

### M3 — Segmentation Value Test
- [ ] Train raw-image baselines
- [ ] Train lung-masked baselines
- [ ] Compare performance and explainability directly

### M4 — Trust Layer
- [ ] External validation
- [ ] Grad-CAM audit set
- [ ] Calibration analysis
- [ ] Misclassification review

### M5 — Prototype App
- [ ] Upload chest X-ray
- [ ] Predict TB suspicion band
- [ ] Show lung mask
- [ ] Show Grad-CAM heatmap
- [ ] Show model version + disclaimer

## Success Criteria
The first trustworthy prototype should:
- beat the dissertation baseline on TB recall or generalization
- hold up on an external dataset
- show measurable value from segmentation or prove it unnecessary
- produce explainability outputs that mostly focus on medically relevant regions
- have reproducible metrics and saved experiment history

## Current Status
Scaffold created. Next actions:
1. create metadata schema and split policy
2. extract notebook logic into modules
3. inspect TBX11K before using it for training

## Colab-ready baseline path
A minimal Colab training path is now included for the current MobileNetV2 baseline.

Files added for this:
- `requirements-colab.txt`
- `scripts/colab_train_baseline.py`
- `COLAB.md`

Quick start on Colab:
1. open a **GPU** runtime
2. clone this repo into `/content/tb-triage-v2`
3. `pip install -r requirements-colab.txt`
4. either:
   - copy in prepared `data/processed/...`, or
   - upload the two raw tar archives under `data/raw/...`
5. run `python scripts/colab_train_baseline.py ...`

Exact commands and expected paths are in `COLAB.md`.
