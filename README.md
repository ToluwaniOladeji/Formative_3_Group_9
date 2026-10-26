# Formative_3_Group_9
# PART 1: Expectation-Maximization (EM) Algorithm for Gaussian Mixture Models
### Probability distribution

## Overview

This project implements the **Expectation-Maximization (EM) algorithm from scratch in Python** to separate a mixed dataset of **father and child heights** into two Gaussian distributions **without using labels**.

The model assumes that every height belongs to one of two hidden groups:

- Distribution 1 (**Children**, the group with the smaller mean)
- Distribution 2 (**Fathers**)

Since the labels are unknown, EM repeatedly estimates which distribution each height most likely belongs to and updates the Gaussian parameters until the log-likelihood converges.

---

# Dataset

Dataset used: **GaltonFamilies.csv**

Columns used:

- `father`
- `childHeight`

The program combines both columns into one unlabeled dataset (a father with several children appears once per child, giving 1,868 heights).

| Height |
|---------|
| 70 |
| 67 |
| 71 |
| 64 |
| ... |

The EM algorithm has **no knowledge** of which values belong to fathers or children.

---

## 2. Why NOT split at the global mean? Why use soft (probabilistic) assignment?

**Why not split at the global mean?**
- **Wrong boundary:** the global mean blends both groups, so it is not where one group ends and the other starts.
- **Misclassification:** children's and fathers' heights overlap, so a hard line forces short fathers and tall children into the wrong pile, and they are never corrected.
- **Inaccurate estimates:** cutting the data truncates both bell curves, so the pile means and variances are distorted.
- **No uncertainty:** each height gets one fixed label, whether it is next to the line or far from it.
- **No improvement:** the split is done once and is never updated.

**Why use soft assignment?**
- **Probabilities:** each height gets a probability for both groups.
- **Handles uncertainty:** borderline heights can belong to either group (about 50/50).
- **Weighted estimates:** heights contribute to the parameters according to their probabilities.
- **Iterative improvement:** EM updates the parameters to improve the log-likelihood.

**Slide: global-mean split on our data**

| | Global-mean split | EM (soft) |
|---|---|---|
| Cut / output | Line at the global mean = 67.97 in | A probability for each group |
| Pile means | 64.73 / 70.23 (distorted by the cut) | Learned from weighted data |
| Misclassified | **600 of 1,868 heights (32%)**: fathers below the line, children above it | No forced label, uncertainty is kept |
| Uncertainty | None | e.g. 67 in = 59.10% Child / 40.90% Father |

---

# Gaussian Distribution

Each hidden group is modeled using a Gaussian (Normal) distribution.

P(x) = (1 / sqrt(2πσ²)) * exp(-(x-μ)² / 2σ²)

Where **μ** = mean, **σ²** = variance, **x** = observed height.

Each group therefore needs three parameters: **mean (μ)**, **variance (σ²)** and **mixing coefficient (π)**, the share of the data that group explains (π₁ + π₂ = 1).

---

# EM Algorithm

The EM algorithm starts from initial guesses and then alternates between two steps.

---

## Step 1 — Initialization

The *variance* and *mean* are calculated from the data itself, not hard-coded:

```python
n = len(data)
average = sum(data) / n
data_variance = sum((x - average) ** 2 for x in data) / n
spread = math.sqrt(data_variance)

mean1 = average - spread
mean2 = average + spread

variance1 = data_variance
variance2 = data_variance

pi1 = 0.5
pi2 = 0.5
```

Reasons for this initialization:
- **Adaptable:** works with different datasets without manually changing starting values.
- **Better starting estimates:** uses the data's average and standard deviation to position the two initial means.
- **Iteration 0:** the initial log-likelihood is calculated from these values before any update.

---

## Step 2 — Expectation (E-Step)

For every height, using the current means, variances and mixing coefficients:

- Compute the probability that it came from Gaussian 1.
- Compute the probability that it came from Gaussian 2.
- Weight each by its mixing coefficient.

```
prob1 = pi1 × Gaussian1
prob2 = pi2 × Gaussian2
```

Normalize them:

```
responsibilities1 = prob1 / (prob1 + prob2)
responsibilities2 = prob2 / (prob1 + prob2)
```

These are called **responsibilities**. Each one tells us how much that data point belongs to each Gaussian.

### Mathematical example of the E-step

Assume one height observation `x = 68` and the initial parameters:

| Parameter | Gaussian 1 | Gaussian 2 |
|-----------|-----------:|-----------:|
| μ         | 64.66      | 71.28 |
| σ²        | 10.96      | 10.96 |
| π         | 0.5        | 0.5 |

Gaussian probabilities:

```
P1 ≈ 0.0724
P2 ≈ 0.0738
```

Multiply by the mixing coefficients:

```
0.5 × 0.0724 = 0.0362
0.5 × 0.0738 = 0.0369
```

Normalize to obtain the responsibilities:

```
Total = 0.0362 + 0.0369 = 0.0731

r1 = 0.0362 / 0.0731 ≈ 0.495
r2 = 0.0369 / 0.0731 ≈ 0.505
```

**Interpretation:** height **68** belongs **49.5%** to Gaussian 1 and **50.5%** to Gaussian 2. This is a **soft assignment**: a borderline height is shared between both groups.

---

## Step 3 — Maximization (M-Step)

After calculating the responsibilities for every height, the Gaussian parameters are updated **using the responsibilities as weights**:

- **Means:** each height contributes according to its responsibility instead of counting equally.
- **Variances:** the weighted spread around the new mean.
- **Mixing coefficients:** `π = effective number of points / total observations`, the proportion of the dataset belonging to each Gaussian.

### Mathematical example of the M-step

Suppose the responsibilities for Gaussian 1 are:

| Height | Responsibility |
|-------:|---------------:|
| 65 | 0.90 |
| 68 | 0.57 |
| 72 | 0.10 |

```
N1 = 0.90 + 0.57 + 0.10 = 1.57                       (effective number of observations)

μ  = (0.90×65 + 0.57×68 + 0.10×72) / 1.57
   = (58.50 + 38.76 + 7.20) / 1.57
   = 104.46 / 1.57 ≈ 66.54

σ² = [0.90(65−66.54)² + 0.57(68−66.54)² + 0.10(72−66.54)²] / 1.57

π  = 1.57 / 3 = 0.523
```

**Interpretation:** about **52.3%** of the dataset currently belongs to Gaussian 1.

## One EM Iteration

```
Initialize μ, σ² and π
        ↓
Compute Gaussian probabilities
        ↓
Calculate responsibilities (E-Step)
        ↓
Update μ, σ² and π (M-Step)
        ↓
Compute log-likelihood
        ↓
Repeat until log-likelihood converges
```

---

## Step 4 — Log-Likelihood and Stopping Condition

After each update we calculate the **log-likelihood**, which measures how well the Gaussian mixture explains the data. A higher value means a better fit, and it improves with every iteration.

**Stopping condition:** stop when the **log-likelihood changes negligibly** between two iterations (the change is smaller than `tolerance`). The number of iterations is therefore not fixed.

```python
tolerance = 0.01
max_iterations = 10000      # safety limit only
```

In our run, the change drops below the tolerance at **iteration 7**.

---

# Optimization Tracking Table

Iteration 0 is the initialization (before any update).

| Iteration | μ₁ (Children) | μ₂ (Fathers) | σ₁² | σ₂² | π₁ | π₂ | Log-Likelihood |
|:---------:|------------:|------------:|-----------------:|-----------------:|---:|---:|---------------:|
| 0 (Initialization) | 64.6603 | 71.2827 | 10.9639 | 10.9639 | 0.5000 | 0.5000 | -5117.4118 |
| 1 | 65.8889 | 69.9436 | 8.1515 | 5.6309 | 0.4864 | 0.5136 | -4866.5010 |
| 2 | 65.8721 | 69.9197 | 8.5488 | 5.3198 | 0.4813 | 0.5187 | -4864.5545 |
| 3 | 65.8624 | 69.9094 | 8.6838 | 5.2163 | 0.4788 | 0.5212 | -4864.2930 |
| 4 | 65.8558 | 69.9037 | 8.7332 | 5.1798 | 0.4773 | 0.5227 | -4864.2437 |
| ... | | | | | | | |
| 7 (converged) | 65.8408 | 69.8944 | 8.7478 | 5.1692 | 0.4744 | 0.5256 | -4864.2075 |

---

# Distribution Before and After EM

![Distribution before and after EM](em_before_after.png)

The grey bars are the same unlabeled heights in both plots. Only the Gaussians change.

- **Before (iteration 0):** the two Gaussians come from the initial guesses (μ = 64.66 and 71.28, σ² = 10.96 for both, π = 0.50 each). They are wide, identical in shape and fit the data poorly (log-likelihood -5117.41).
- **After (iteration 7):** EM has moved the means (65.84 and 69.89), reduced the variances (8.75 and 5.17) and adjusted the mixing coefficients (0.47 and 0.53). The mixture (dashed line) now follows the shape of the histogram much better (log-likelihood -4864.21).
- **Global mean (red dotted line, 67.97 in):** heights from both groups fall on each side of it, which is why a hard split at the mean misclassifies them.

---

# Classification: Posterior Probability for a Test Height

For a new height, Bayes' rule with the learned parameters gives the posterior probability of each group (computed in log-space so extreme heights never cause a zero-division error).

```
Enter test height: 67

Prediction for: 67.0
Probability it is a Child : 0.590952 (59.10 %)
Probability it is a Father: 0.409048 (40.90 %)
Sum of probabilities      : 1.0
Decision -> Child
```

---

# Running the Project

## Requirements

Python 3.x with the standard libraries `csv`, `math`, and `pandas` (only used to preview the dataset with `df.head()`).

No machine learning libraries are used.

## Execute

Put `GaltonFamilies.csv` in the same folder (or in `/content/` on Google Colab), then run:

```bash
jupyter notebook Probabilty_Distribution_Presentation_Improved.ipynb
```

Run all cells (this also saves the histogram as `em_before_after.png`) and enter any height when prompted, for example:

```
Enter test height:

67
```

---
