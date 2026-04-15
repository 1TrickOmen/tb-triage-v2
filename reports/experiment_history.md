# TB Triage V2 — Experiment History

## Purpose

This file is the clean project ledger for `tb-triage-v2`, reconstructed from:
- repo history
- committed docs
- run summaries captured in Telegram topic `13768`
- later Lightning/remote-training discussion

It exists because the scientific history of the project ended up split across:
- code commits
- Colab outputs
- Lightning runs
- chat messages

That is a terrible way to remember experiment results.

---

## 1. Project framing

The project began as a dissertation-era TB chest X-ray prototype and was reframed into a more serious goal:

- **not** “AI diagnoses TB”
- **yes** “TB chest X-ray triage / second-reader prototype”

Core V2 priorities became:
- use **multi-source data**
- prefer **external validation** over flattering internal splits
- evaluate **recall, specificity, ROC-AUC, PR-AUC**
- keep **Grad-CAM** for explainability
- treat **segmentation as optional unless it proves value**

Target datasets discussed and used:
- Shenzhen
- Montgomery
- TBX11K
- later external Pakistan Mendeley dataset

---

## 2. Repo / pipeline build

The old notebook was ported into a real repo: `tb-triage-v2`.

Major repo capabilities added over time:
- project scaffold / modular structure
- metadata-driven ingestion
- baseline classifier training
- threshold analysis
- external validation workflow
- source-held-out evaluation
- source-balanced weighting
- DenseNet121 support
- segmentation-prep path
- Grad-CAM / heatmap tooling
- Colab helper scripts

### Important commit landmarks

- `3b558b9` — initial scaffold / baseline structure
- `0433981` — class-weighted rerun + Keras training-loop fix
- `c03c230` — threshold analysis workflow
- `6837b53` — segmentation-prep workflow
- `2af02b5` — external validation workflow (Pakistan)
- `f5dcbdc` — source-held-out workflow
- `d0941bd` — source-balanced weighting workflow
- `3086104` — DenseNet121 support
- `1d828cf` — CLAHE + stronger augmentation experiment prep
- `f561df7` — mild augmentation + disable CLAHE

---

## 3. Data preparation history

### What was done
- Downloaded / parsed TBX11K
- Wired chest-xray masks/labels data
- Built merged metadata from raw archives
- Fixed multiple path-resolution / extraction issues
- Learned not to stuff generated/extracted bulk into Git

### Important practical lessons
- Generated data belongs outside Git
- Processed metadata/artifacts need to be reproducible
- The Pi/workspace machine is fine for orchestration/prep, not for heavy training

---

## 4. Old notebook findings

Recovered dissertation-era notebook characteristics:
- Kaggle chest-xray segmentation dataset
- `MetaData.csv`-driven loading
- MobileNetV2 classifier
- separate U-Net segmentation path
- Grad-CAM
- 70/15/15 split style

### Key technical conclusion
The old notebook was a real academic prototype, but segmentation was **not tightly integrated into classifier training**.

In practice:
- classification was the real engine
- segmentation mostly supported visualization / focus / narrative
- dissertation framing made it feel more integrated than it really was

---

## 5. Colab baseline phase

Training moved to **Colab T4** for actual GPU runs.

### 5.1 MobileNetV2 baseline
Setup:
- MobileNetV2
- frozen base
- 256x256
- 15 epochs

Result:
- accuracy: **0.922**
- ROC-AUC: **0.964**
- PR-AUC: **0.920**
- TB precision: **0.904**
- TB recall: **0.725**

### Conclusion
A strong internal baseline, but TB recall was still not ideal for triage.

---

## 6. Class-weighted rerun + threshold analysis

### Why
The baseline looked good internally, but recall needed improvement.

### Changes
- fixed flaky Keras training-loop behavior
- added class weighting
- added threshold sweeping instead of assuming threshold 0.5 was optimal

### Important threshold conclusion
Preferred operating point became approximately:
- **threshold = 0.40**

### Internal triage-style takeaway
This gave a better recall-oriented operating point and became the best internal MobileNetV2 reference run.

---

## 7. External validation on Pakistan Mendeley dataset

### Why
Internal metrics were not enough; we needed out-of-distribution testing.

### MobileNetV2 external result
At the chosen threshold, the model showed:
- TB recall transferred well: **~92.6%**
- specificity collapsed: **~5.4%**

### Conclusion
The model became extremely trigger-happy on external data.

Interpretation:
- it could still catch TB
- but it massively overcalled Normal images as TB
- internal success did **not** imply safe external behavior

This was one of the biggest project wake-up calls.

---

## 8. Source-held-out evaluation

We then tested a better form of generalization:
- train on seen sources
- hold one source out completely

### 8.1 Shenzhen holdout — MobileNetV2
At threshold 0.40:
- TB recall: **82.9%**
- specificity: **35.7%**
- precision TB: **57.1%**
- ROC-AUC: **0.728**
- PR-AUC: **0.760**

### 8.2 Montgomery holdout — MobileNetV2
At threshold 0.40:
- TB recall: **93.1%**
- specificity: **26.3%**
- precision TB: **47.8%**
- ROC-AUC: **0.766**

### Conclusion
Pattern became clear:
- TB detection sensitivity transferred better than specificity
- the real weakness was false positives on unseen-source Normal images

That directly motivated the next experiment family: **normal/source robustness**.

---

## 9. Source-balanced training

### Why
To reduce false positives on unseen-source Normal images.

### Shenzhen source-balanced MobileNetV2
Documented result:
- TB recall: **69.1%**
- specificity: **70.5%**

### Comparison to earlier Shenzhen holdout MobileNetV2
Earlier non-source-balanced result:
- recall: **82.9%**
- specificity: **35.7%**

### Conclusion
Source balancing improved specificity a lot, but recall dropped substantially.

Interpretation:
- weighting helped reduce false positives
- but it also cost too much TB sensitivity to be an obvious win

---

## 10. DenseNet121 phase

### Why
MobileNetV2 looked too limited for the cross-source robustness problem.

### What changed
- added DenseNet121 as a selectable architecture
- extended eval / script support to handle architecture-specific model artifacts

### Important summary result captured in chat
For a Shenzhen holdout DenseNet-based source-balanced run, later thread summaries described a more respectable tradeoff around:
- TB recall: **79.7%**
- TB precision: **63.6%**
- specificity: **52.8%**

### Conclusion
DenseNet looked promising as a stronger backbone candidate, though not yet a solved answer.

---

## 11. Segmentation value investigation

### Goal
Determine whether lung-masked input actually improves the classifier, rather than just making the dissertation story prettier.

### What was built
- segmentation-prep workflow
- mask prediction path
- masked metadata generation path
- scripts to compare masked vs raw experiments

### Technical fixes made
- Kaggle custom-object model loading fixes
- grayscale-vs-RGB fixes
- input-size handling fixes

### What went wrong
The Kaggle U-Net checkpoint path turned into a VRAM / inference pain point:
- model loaded only after multiple compatibility fixes
- fixed 512x512 behavior was heavy
- inference became impractical on the current Colab path

### Scientific conclusion
Segmentation was **not clearly adding core classifier value** in the current pipeline.

What remained true:
- Grad-CAM still works without segmentation
- segmentation may still be useful later
- but it stopped being a required pillar of the immediate model roadmap

---

## 12. Heatmaps / Grad-CAM analysis

Heatmaps were generated on:
- Pakistan external cases
- Shenzhen holdout cases

### What they suggested
- On bad domain-shift behavior, the model sometimes seemed to rely on global/domain cues rather than robust pathology signal
- On better-performing cases, attention looked more plausible

### Conclusion
Heatmaps were useful as a sanity lens, but not proof that the model was clinically meaningful.

---

## 13. Robustness patch that backfired

A robustness-oriented patch introduced:
- CLAHE preprocessing
- stronger augmentation

### Result
This path collapsed badly.

Later summary of the bad DenseNet run:
- accuracy looked superficially okay: **0.8246**
- but at threshold 0.5 it predicted **everything as Normal**
- confusion matrix: `[[583, 0], [124, 0]]`
- TB precision / recall / F1: **0 / 0 / 0**
- ROC-AUC: **0.478**
- PR-AUC: **0.209**

### Conclusion
The “robustness” patch overcorrected and made the recipe worse.

This led to the next repo rollback direction:
- disable CLAHE
- use **mild augmentation** instead of heavy augmentation

---

## 14. Move from Colab to Lightning

### Why the procedure changed
Originally, the workflow was:
- edit repo locally
- push to GitHub
- give Colab commands
- user runs them

That changed because the bottleneck changed.

New bottlenecks became:
- remote environment reliability
- GPU visibility / TensorFlow compatibility
- corrupted archives
- verifying training on the actual target box

### New workflow
- orchestrate locally
- execute remotely over Lightning SSH
- fix env/data on-box
- launch training there
- monitor/report results

### Practical environment lessons
- Lightning handles expired / refreshed frequently
- a clean env using `tensorflow[and-cuda]` was eventually the fix path
- TensorFlow **2.17.1** eventually saw the **T4 GPU** correctly

---

## 15. Lightning infrastructure recovery

### Problems encountered
- broken / missing envs after refresh
- SSH handle/auth churn
- remote archive corruption
- studio sleep / inactivity behavior
- watcher auth inheritance failures

### Eventually fixed
- SSH working again
- remote archive download directly on Lightning
- archive validation with `tar -tf`
- rebuilt extracted data and merged metadata
- TensorFlow GPU stack working on the remote T4

### Important conclusion
By this point, the project was **not blocked by tooling anymore**.

The bottleneck became experiment quality, not pipeline plumbing.

---

## 16. Lightning baseline recovery run

### Why it was run
This was mainly an end-to-end remote execution proof run after all the Lightning setup nonsense.

### Setup
- MobileNetV2
- rebuilt merged metadata
- baseline training on Lightning

### Result
Finished cleanly, but was only mediocre:
- roughly **~0.59 accuracy / ~0.59 ROC-AUC** (as described in chat)

### Conclusion
This run mattered mostly because it proved:
- remote dataset rebuild works
- remote training works
- GPU path works

Scientifically, it was not strong.

---

## 17. Shenzhen DenseNet mild-augmentation run on Lightning

### Setup
- Shenzhen fully held out
- source-balanced seen-source training
- DenseNet121
- mild augmentation
- no CLAHE

### Operationally
- run launched successfully
- watcher behavior became a process problem and exposed monitoring design issues
- run later confirmed **finished**

### Final result
Poor / collapsed for actual TB triage:
- accuracy: **0.8246**
- confusion matrix: `[[583, 0], [124, 0]]`
- TB precision / recall / F1: **0 / 0 / 0**
- ROC-AUC: **0.478**
- PR-AUC: **0.209**
- predicted all cases as Normal at the default threshold

### Conclusion
This was a scientifically bad run despite the fake-good accuracy.

---

## 18. Audit of the collapsed Shenzhen DenseNet run

### Audit goal
Check whether the result was caused by:
- label flip
- wrong positive-class probability
- evaluation bug
- Shenzhen-holdout bookkeeping bug

### Audit finding
No structural bug was found.

Confirmed:
- Normal = 0
- TB = 1
- 2-way softmax semantics consistent
- thresholding / saved probabilities / reports lined up

### Audit conclusion
The collapse was **not** caused by a bookkeeping or eval bug.

Likely explanation:
- the **training recipe failed**
- aggressive weighting and/or frozen/training setup likely sabotaged learning

---

## 19. Seen-sources sanity run on Lightning

### Why
To test whether the source-held-out collapse was just a hard split problem or whether the recipe itself was broken.

### Setup
- seen sources only
- fresh internal split
- unweighted
- MobileNetV2

### Result
- accuracy: **0.7783**
- val_accuracy: **0.7768**
- ROC-AUC: **0.490**
- PR-AUC: **0.213**
- confusion approximately: `TN 636, FP 2, FN 182, TP 0`

### Conclusion
This was a prettier failure, but still a failure.

Key implication:
- the problem was **not only** the harsh Shenzhen holdout
- even a cleaner seen-source setup showed weak TB-vs-Normal separation

---

## 20. Comparison conclusion from Lightning runs

Explicit comparison later concluded:
- collapsed Shenzhen DenseNet run: bad
- seen-sources unweighted MobileNet sanity run: also bad

### Key takeaway
Both runs failed at actual TB detection.

Interpretation:
- source balancing may have hurt one run badly
- but weighting is **not the only villain**
- model/data recipe appears to be learning little useful ranking signal in these later runs

---

## 21. Shenzhen DenseNet121 recovery run on Colab

### Why
This was the explicit recovery ablation after the collapsed robustness paths. The goal was to test whether a cleaner DenseNet recipe could recover useful Shenzhen source-held-out performance without class weighting or CLAHE.

### Setup
- true unseen source holdout: **Shenzhen**
- training sources: **TBX11K + Montgomery** only
- architecture: **DenseNet121**
- trainable fraction: **0.25**
- learning rate: **3e-5**
- class weighting: **none**
- augmentation: **mild**
- image size: **256x256**
- requested epochs: **20**
- runtime: **Colab T4 GPU**

Split sizes:
- train: **3294**
- val: **706**
- test: **707**

### Default-threshold result (0.5 on seen-source internal test split)
- accuracy: **0.8218**
- loss: **0.4214**
- ROC-AUC: **0.7405**
- PR-AUC: **0.3705**
- confusion matrix: `[[579, 4], [122, 2]]`
- TB precision: **0.3333**
- TB recall: **0.0161**

### Interpretation
This run was **not a full collapse**, because the ROC-AUC shows some ranking signal survived. But as an operating classifier it was still bad: the default decision boundary behaved like “predict almost everything as Normal.”

### Threshold sanity check from saved test predictions
Lower thresholds improved recall, but not enough to make the model genuinely good:

- threshold **0.40** → recall **0.0726**, specificity **0.9863**, precision **0.5294**
- threshold **0.30** → recall **0.2258**, specificity **0.9588**, precision **0.5385**
- threshold **0.25** → recall **0.3629**, specificity **0.9005**, precision **0.4369**
- threshold **0.20** → recall **0.5565**, specificity **0.7633**, precision **0.3333**

### Training dynamics
Training history looked unstable rather than convincingly healthy:
- best validation accuracy occurred early, around **epoch 5** (`0.8286`)
- lowest validation loss occurred even earlier, around **epoch 3** (`0.4309`)
- later validation behavior became erratic and degraded badly

### Conclusion
This was **not a recovery run** in the scientific sense.

It was better than the totally dead all-Normal recipe because it retained some score separation, but it still failed as a TB triage model. Threshold tuning could expose some signal, yet the model was still far from a trustworthy operating point.

### Best next move
Keep the same disciplined Shenzhen holdout setup and change **one variable only**. The next ablation should be:
- **DenseNet121**
- same holdout / split logic
- same learning rate (`3e-5`)
- same trainable fraction (`0.25`)
- same `class-weight none`
- but change **augmentation from `mild` to `none`**

Rationale:
- this run suggests the model may have weak ranking signal but a poor decision boundary
- validation instability suggests the recipe is still not healthy
- removing augmentation is the cleanest next single-variable test before changing backbone or unfreeze depth again

---

## 22. Monitoring / process lessons

The Telegram thread captured several important process lessons, independent of model science.

### Lessons learned
1. **box-side watcher != user-facing notification**
2. `timeout=0` does **not** solve missing SSH auth inheritance
3. SSH handle alone is useless without the actual key/auth context
4. Lightning studio sleep can invalidate naïve watcher assumptions
5. local-only repo edits are not “done” if Colab/remote uses GitHub state
6. one long-lived worker is often cleaner than a separate launch worker + finish watcher

### Behavioral commitment that emerged
For long jobs, the better pattern is:
- launch
- monitor
- report only when there is a real result or blocker

Not vague “done” semantics.

---

## 23. What the project achieved

### Tooling / infrastructure achievements
- real modular repo
- reproducible metadata rebuild paths
- Colab training path
- Lightning remote-training path
- external evaluation scripts
- source-held-out evaluation scripts
- DenseNet support
- Grad-CAM tooling

### Scientific / project achievements
- established that internal results alone are misleading
- identified specificity on unseen-source normals as a major bottleneck
- proved source-held-out validation is essential
- showed segmentation is not currently proven to add classifier value
- identified that some late training recipes collapse completely

---

## 24. What currently looks true

### Strongest project-level conclusions
1. **The pipeline is mature enough.**  
   The repo is not the problem anymore.

2. **Generalization is the real problem.**  
   Especially specificity / false positives on unseen-source Normal images.

3. **Threshold tuning helps only when the model has signal.**  
   It does not rescue a model with near-zero separation.

4. **Weighting must be treated carefully.**  
   It may improve some tradeoffs but can also sabotage learning.

5. **Segmentation should be considered optional until it proves value.**

6. **Accuracy is a liar here.**  
   Several runs looked acceptable by accuracy while being clinically useless.

---

## 25. Best next steps

### A. Documentation / discipline
- Keep this file updated after every meaningful experiment
- Add finished-run artifacts/metrics references whenever possible
- Stop leaving key results only in chat

### B. Experimental strategy
Run disciplined ablations only:
- weighting on/off
- frozen vs trainable base
- MobileNetV2 vs DenseNet121
- mild aug vs none

Change one thing at a time.

### C. Evaluation strategy
Keep **source-held-out evaluation** as the main truth serum.

Do not regress to pretty mixed-source-only metrics.

### D. Segmentation strategy
Only continue segmentation if doing a narrow, disciplined:
- raw vs masked comparison

No giant side quest unless it shows clear value.

### E. Operational strategy
Prefer one long-lived worker for:
- launch
- monitor
- final report

rather than splitting launch and watcher semantics across multiple loosely coupled workers.

### F. Immediate next run
Run the direct follow-up ablation:
- Shenzhen holdout
- DenseNet121
- trainable fraction `0.25`
- learning rate `3e-5`
- class weight `none`
- **augmentation `none`**

Keep everything else the same so the comparison against the failed mild-augmentation recovery run stays clean.

---

## 26. Known gaps in this history

Some exact final metrics/results are still not preserved as local repo artifacts.

In particular, the local repo did **not** contain all Lightning experiment outputs when this history was reconstructed. Some parts of this ledger therefore depend on Telegram-thread evidence rather than committed metrics files.

That means this file is currently:
- **better than memory alone**
- **better than chat archaeology**
- but still not as good as a properly maintained experiment registry

---

## 27. One-line project summary

`tb-triage-v2` evolved from a convincing academic prototype into a real experimental pipeline, and the current blocker is no longer plumbing — it is finding a training recipe that learns a genuinely generalizable TB-vs-Normal signal instead of producing flattering but brittle results.
