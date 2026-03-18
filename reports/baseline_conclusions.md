# Baseline Conclusions: MobileNetV2

## Executive Summary
After repairing the data pipeline and implementing a rigorous source-held-out validation protocol, we have exhausted the capacity of the MobileNetV2 architecture for the TB triage task.

While MobileNetV2 is capable of high sensitivity (catching TB), it fundamentally struggles with specificity (rejecting normal X-rays) when exposed to unseen hospital sources. Balancing the training data by source mitigated the false positive rate, but at a severe cost to recall, proving the model lacks the representational capacity to generalize the subtle radiological differences between TB and varied normal scans.

## The Journey & Metrics

### 1. Plain Class-Weighted Baseline (The "Trigger-Happy" Model)
*Trained on TBX11K + Montgomery, evaluated on held-out Shenzhen*
* **TB Recall:** 82.9%
* **Specificity:** 35.7% (Terrible)
* **Conclusion:** The model learned to over-predict TB on unfamiliar images. It was highly sensitive but clinically unusable due to an unacceptable false positive rate.

### 2. Source-Balanced Baseline (The "Conservative" Model)
*Trained on TBX11K + Montgomery (equal weight per source/label), evaluated on held-out Shenzhen*
* **TB Recall:** 69.1% (Dropped hard)
* **Specificity:** 70.5% (Much better)
* **Conclusion:** Forcing the model to respect underrepresented normal cases fixed the over-calling issue, but the model did not have enough capacity to maintain TB recall.

### The Threshold Tradeoff (Source-Balanced Model)
To get acceptable TB recall from the balanced model, we have to lower the threshold drastically, which destroys specificity again:
* **Threshold 0.15:** Recall 92.7%, Specificity 25.7%
* **Threshold 0.40:** Recall 69.1%, Specificity 70.5%
* **Threshold 0.60:** Recall 43.3%, Specificity 89.2%

## Final Decision
We cannot calibrate our way out of this tradeoff with MobileNetV2. The architecture is too small for cross-domain medical imaging.

**Next Technical Step:**
Swap the backbone from MobileNetV2 to **DenseNet121** (the industry standard for Chest X-ray analysis, famously used in CheXNet). We will run the exact same source-held-out, source-balanced experiment to see if DenseNet121 provides the capacity needed to maintain both >90% recall and >80% specificity on unseen data.
