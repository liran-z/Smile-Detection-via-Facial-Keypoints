# Smile Detection via Facial Keypoints

A machine learning project that detects smiles using only facial landmark coordinates —
no face images required. Built as part of a recruitment coding assessment.

## Project Overview

Live streaming platforms and social media advertisers need real-time insight into viewer
reactions beyond likes and comments. This project implements a lightweight smile classifier
(`smile_predict()`) using 15 facial keypoints (x, y coordinates), enabling millisecond-speed
inference with minimal privacy risk.

## Files

| File | Description |
|------|-------------|
| `coding_test_smile_detection.ipynb` | Full implementation: EDA, feature engineering, model training & evaluation |
| `data/facial_keypoints.json` | Dataset — 657 train samples, 282 test samples with smile labels |
| `smile_detection_report.pdf` | Verification report: analysis methodology and findings |

## Approach

1. **Exploratory Analysis** — Visualized keypoint positions to identify which points
   correspond to eyes, nose, and mouth without labels
2. **Feature Design** — Compared two plans: mouth-region only (Plan A) vs. all 15 points (Plan B)
3. **Model Comparison** — Evaluated Decision Tree, Random Forest, SVM, and Logistic Regression
4. **Final Model** — Plan B + Logistic Regression: **94.3% test accuracy**, no overfitting

## Results

| Plan | Model | Train Acc | Test Acc | Notes |
|------|-------|-----------|----------|-------|
| Plan A | Logistic Regression | 93.3% | 93.3% | Stable |
| Plan B | Logistic Regression ⭐ | 94.1% | **94.3%** | Best |
| Plan B | Random Forest | 100% | 91.5% | Overfit |

## Tech Stack

- Python, scikit-learn, NumPy, Matplotlib
- Models: Logistic Regression, SVM, Decision Tree, Random Forest
