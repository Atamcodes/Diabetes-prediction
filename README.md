# Diabetes Prediction Using Ensemble Machine Learning Methods
### A Comparative Study on PIMA Indians & CDC BRFSS 2015 Datasets

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Datasets](#3-datasets)
4. [System Architecture](#4-system-architecture)
5. [Methodology](#5-methodology)
6. [Results](#6-results)
7. [Cross-Dataset Comparison](#7-cross-dataset-comparison)
8. [Key Findings](#8-key-findings)
9. [How to Run](#9-how-to-run)
10. [Project Structure](#10-project-structure)
11. [Future Work](#11-future-work)
12. [References](#12-references)

---

## 1. Project Overview

This project implements a **complete supervised machine learning pipeline** for diabetes prediction using ensemble methods. The pipeline is executed independently on two distinct diabetes datasets, and results are compared across both to identify the most effective approach for production deployment.

**Key Highlights:**
- 10 base classifiers evaluated on each dataset
- 4 ensemble methods: Voting, Bagging, Stacking, Weighted Voting
- Hyperparameter tuning via RandomizedSearchCV
- SHAP explainability analysis
- Cross-dataset comparative report
- Unified prediction API with saved models

---

## 2. Problem Statement

Diabetes mellitus is a chronic metabolic disorder affecting **537 million adults worldwide** (IDF Diabetes Atlas, 2021). Early prediction through machine learning can enable timely clinical intervention and reduce complications.

**Objectives:**
- Build an end-to-end ML pipeline with proper preprocessing, training, and evaluation
- Compare 10 classification algorithms on two distinct diabetes datasets
- Apply ensemble methods to improve prediction performance
- Explain model predictions using SHAP and permutation importance
- Recommend the best model and dataset combination for production

---

## 3. Datasets

### 3.1 PIMA Indians Diabetes Dataset
| Attribute | Value |
|-----------|-------|
| Source | UCI Machine Learning Repository / Kaggle |
| Records | 768 |
| Features | 8 numeric + 1 binary target |
| Target | `Outcome` (0 = No diabetes, 1 = Diabetes) |
| Class Distribution | 65.1% negative / 34.9% positive |
| Known Issues | Zeros used as missing values in 5 columns |
| Population | Pima Indian women ≥ 21 years |

**Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

### 3.2 CDC BRFSS 2015 Dataset
| Attribute | Value |
|-----------|-------|
| Source | CDC Behavioral Risk Factor Surveillance System / Kaggle |
| Records | 253,680 |
| Features | 21 numeric + 1 binary target |
| Target | `Diabetes_binary` (0 = No diabetes, 1 = Diabetes) |
| Class Distribution | 86.07% negative / 13.93% positive |
| Known Issues | Severe class imbalance; self-reported survey data |
| Population | US adults (telephone survey) |

**Features:** HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income

---

## 4. System Architecture

### 4.1 ML Pipeline Architecture

```
┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────────┐
│   Data   │──▶│   EDA &  │──▶│    Model     │──▶│  Ensemble    │
│ Ingestion│   │  Preproc  │   │  Training    │   │  Methods     │
└──────────┘   └──────────┘   └──────────────┘   └──────────────┘
                    │                │                    │
              ┌─────┘          ┌─────┘              ┌────┘
              ▼                ▼                     ▼
        ┌──────────┐   ┌──────────────┐   ┌──────────────┐
        │  SMOTE   │   │  Hyper-Param │   │  Evaluation  │
        │  Scaling │   │   Tuning     │   │  & SHAP      │
        └──────────┘   └──────────────┘   └──────────────┘
```

### 4.2 Proposed Deployment Architecture

```
┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────┐
│  Client  │──▶│  Flask /     │──▶│ Feature  │──▶│   Ensemble   │──▶│  Risk    │
│  (Web/   │   │  FastAPI     │   │ Validate │   │   Model      │   │  Score   │
│   API)   │   │  Endpoint    │   │ & Scale  │   │   Predict    │   │  Output  │
└──────────┘   └──────────────┘   └──────────┘   └──────────────┘   └──────────┘
                                       │                │
                                  ┌────┘           ┌────┘
                                  ▼                ▼
                            ┌──────────┐    ┌──────────────┐
                            │ Scaler   │    │ Best Model   │
                            │  .pkl    │    │    .pkl      │
                            └──────────┘    └──────────────┘
```

### 4.3 Technology Stack
| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| ML Framework | scikit-learn, XGBoost, LightGBM, CatBoost |
| Imbalanced Learning | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Visualization | matplotlib, seaborn |
| Serialization | joblib |

---

## 5. Methodology

### 5.1 Preprocessing
| Step | PIMA | BRFSS |
|------|------|-------|
| Missing Values | Replaced zeros with column median in 5 features | None needed |
| Train/Test Split | 80/20 stratified (random_state=42) | 80/20 stratified |
| Class Balancing | SMOTE on training set (400→400) | SMOTE (174K→174K per class) |
| Feature Scaling | StandardScaler | StandardScaler |

### 5.2 Base Models (10 Classifiers)
1. Logistic Regression
2. Decision Tree (max_depth=5)
3. Random Forest (100 estimators)
4. Gradient Boosting
5. XGBoost
6. LightGBM
7. CatBoost
8. K-Nearest Neighbors
9. Support Vector Machine (RBF kernel)
10. Extra Trees Classifier

### 5.3 Hyperparameter Tuning
- **Method:** RandomizedSearchCV
- **Top-3 models** selected by ROC-AUC from base evaluation
- **PIMA:** cv=5, n_iter=30
- **BRFSS:** cv=3, n_iter=20 (adjusted for dataset size)
- **Scoring:** ROC-AUC

### 5.4 Ensemble Methods
| Method | Description |
|--------|-------------|
| **Soft Voting** | Average predict_proba from all 3 tuned models |
| **Bagging** | Bootstrap aggregation with best model as base estimator |
| **Stacking** | 3 tuned models as base + Logistic Regression meta-learner |
| **Weighted Voting** | Manual weighted average using individual ROC-AUC as weights |

---

## 6. Results

### 6.1 PIMA — Base Model Results (Sorted by ROC-AUC)
| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|-------|----------|-----------|--------|----------|---------|
| 1 | Extra Trees | 0.7273 | 0.6034 | 0.6481 | 0.6250 | **0.8241** |
| 2 | Gradient Boosting | 0.7338 | 0.6032 | 0.7037 | 0.6496 | 0.8220 |
| 3 | CatBoost | 0.7403 | 0.6167 | 0.6852 | 0.6491 | 0.8215 |
| 4 | Random Forest | 0.7532 | 0.6429 | 0.6667 | 0.6545 | 0.8213 |
| 5 | LightGBM | 0.7338 | 0.6102 | 0.6667 | 0.6372 | 0.8141 |

### 6.2 BRFSS — Base Model Results (Sorted by ROC-AUC)
| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|-------|----------|-----------|--------|----------|---------|
| 1 | XGBoost | 0.8638 | 0.5323 | 0.1853 | 0.2749 | **0.8235** |
| 2 | LightGBM | 0.8629 | 0.5206 | 0.2054 | 0.2946 | 0.8234 |
| 3 | CatBoost | 0.8645 | 0.5389 | 0.1890 | 0.2798 | 0.8234 |
| 4 | Logistic Regression | 0.7322 | 0.3110 | 0.7581 | 0.4410 | 0.8177 |
| 5 | Gradient Boosting | 0.8449 | 0.4351 | 0.3784 | 0.4048 | 0.8174 |

### 6.3 Ensemble Results
| Ensemble | PIMA ROC-AUC | BRFSS ROC-AUC |
|----------|-------------|---------------|
| Voting (Soft) | 0.8180 | 0.8227 |
| **Bagging** | **0.8287** ★ | 0.8214 |
| Stacking | 0.8165 | 0.8197 |
| Weighted Voting | 0.8180 | 0.8227 |

### 6.4 Final Winners
| Dataset | Best Model | ROC-AUC | Top-3 Features |
|---------|-----------|---------|----------------|
| **PIMA** | Bagging (Extra Trees) | **0.8287** | Glucose, BMI, Age |
| **BRFSS** | XGBoost | **0.8235** | GenHlth, HighBP, BMI |

---

## 7. Cross-Dataset Comparison

| Metric | PIMA (Bagging) | BRFSS (XGBoost) |
|--------|---------------|-----------------|
| Accuracy | 0.7338 | 0.8638 |
| Precision | 0.6066 | 0.5323 |
| Recall | **0.6852** | 0.1853 |
| F1-Score | **0.6435** | 0.2749 |
| ROC-AUC | **0.8287** | 0.8235 |

**Key Observations:**
- BRFSS achieves higher **accuracy** (86%) but at the cost of very low **recall** (19%)
- PIMA achieves much better **recall** (69%) — better at detecting actual diabetics
- ROC-AUC is comparable (~0.82–0.83), suggesting similar discriminative power
- BMI is a critical feature across **both** datasets

---

## 8. Key Findings

1. **Ensemble methods improve robustness** — Bagging achieved best overall ROC-AUC on PIMA
2. **Tree-based models dominate** — XGBoost, LightGBM, CatBoost consistently in top-3
3. **Feature importance aligns with medical literature** — Glucose (PIMA) and GenHlth/HighBP (BRFSS)
4. **Class imbalance requires careful handling** — SMOTE + threshold tuning critical for BRFSS
5. **Production recommendation:** BRFSS model for general-purpose screening; PIMA model for clinical settings with basic measurements

### Dataset Limitations
| PIMA | BRFSS |
|------|-------|
| Small dataset (768 records) | Survey-based, self-reported |
| Women-only (Pima Indian) | US-specific population |
| Zeros as missing values | Binary features lose granularity |
| Limited generalizability | Response bias possible |

---

## 9. How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn \
            xgboost lightgbm catboost shap joblib kaggle
brew install libomp  # macOS only, required for XGBoost
```

### Download Datasets
```bash
kaggle datasets download -d uciml/pima-indians-diabetes-database -p data --unzip
kaggle datasets download -d alexteboul/diabetes-health-indicators-dataset -p data --unzip
```

### Run Pipeline
```bash
python3 diabetes_pipeline.py
```

### Use Prediction API
```python
from diabetes_pipeline import predict

result = predict({
    "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72,
    "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627, "Age": 50
}, "pima")
# → {'prediction': 1, 'probability': 0.6576}
```

---

## 10. Project Structure

```
fds project/
├── diabetes_pipeline.py          # Main ML pipeline (all 7 steps)
├── README.md                     # This documentation
├── data/
│   ├── diabetes.csv              # PIMA dataset
│   └── diabetes_binary_health_indicators_BRFSS2015.csv
├── models/
│   ├── pima_best_model.pkl       # Saved Bagging model
│   ├── pima_scaler.pkl           # PIMA StandardScaler
│   ├── brfss_best_model.pkl      # Saved XGBoost model
│   └── brfss_scaler.pkl          # BRFSS StandardScaler
└── plots/
    ├── pima_class_distribution.png
    ├── pima_correlation_heatmap.png
    ├── pima_pairplot.png
    ├── pima_boxplots.png
    ├── pima_confusion_matrix.png
    ├── pima_roc_curve.png
    ├── pima_precision_recall.png
    ├── pima_permutation_importance.png
    ├── brfss_class_distribution.png
    ├── brfss_correlation_heatmap.png
    ├── brfss_chi2_top10.png
    ├── brfss_confusion_matrix.png
    ├── brfss_roc_curve.png
    ├── brfss_precision_recall.png
    ├── brfss_shap_beeswarm.png
    ├── brfss_shap_bar.png
    ├── cross_dataset_roc_overlay.png
    └── cross_dataset_ensemble_comparison.png
```

---

## 11. Future Work

- **Threshold optimization** for BRFSS to improve recall on minority class
- **Deep learning models** (Neural Networks, TabNet) for comparison
- **Feature engineering** — interaction terms, polynomial features
- **Real-time API** deployment with Flask/FastAPI + Docker
- **Federated learning** approach for privacy-preserving diabetes prediction
- **Multi-class classification** using the 3-class BRFSS dataset (pre-diabetes)

---

## 12. References

1. UCI Machine Learning Repository — Pima Indians Diabetes Dataset
2. CDC BRFSS 2015 — Behavioral Risk Factor Surveillance System
3. Chawla et al. (2002) — SMOTE: Synthetic Minority Over-sampling Technique
4. Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions (SHAP)
5. IDF Diabetes Atlas, 10th Edition (2021)
6. Chen & Guestrin (2016) — XGBoost: A Scalable Tree Boosting System
7. Ke et al. (2017) — LightGBM: A Highly Efficient Gradient Boosting Decision Tree

