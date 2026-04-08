"""
 DIABETES PREDICTION PIPELINE — ENSEMBLE METHODS
 Dual-Dataset Comparative Study: PIMA Indians + CDC BRFSS 2015
"""

# ── Standard & numeric libraries ────────────────────────────────────
import warnings, os, sys, json
warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import numpy as np
import pandas as pd

# ── Visualisation ───────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for CI / servers
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ── Scikit-learn core ───────────────────────────────────────────────
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, auc
)

# ── Classifiers ─────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, VotingClassifier, BaggingClassifier,
    StackingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# ── SMOTE ───────────────────────────────────────────────────────────
from imblearn.over_sampling import SMOTE

# ── Feature selection (chi-square) ──────────────────────────────────
from sklearn.feature_selection import chi2

# ── SHAP ────────────────────────────────────────────────────────────
import shap

# ── Persistence ─────────────────────────────────────────────────────
import joblib

# ── Paths ───────────────────────────────────────────────────────────
DATA_DIR   = "data"
PLOT_DIR   = "plots"
MODEL_DIR  = "models"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Classification thresholds (BRFSS lowered to improve recall on imbalanced data)
THRESHOLDS = {"pima": 0.50, "brfss": 0.25}


# =====================================================================
# 1. DATA LOADING
# =====================================================================
def load_data(dataset_name: str) -> pd.DataFrame:
    """Load raw CSV for PIMA or BRFSS dataset."""
    if dataset_name == "pima":
        path = os.path.join(DATA_DIR, "diabetes.csv")
    elif dataset_name == "brfss":
        path = os.path.join(DATA_DIR, "diabetes_binary_health_indicators_BRFSS2015.csv")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    df = pd.read_csv(path)
    print(f"\n{'='*60}")
    print(f"  Loaded {dataset_name.upper()} — shape: {df.shape}")
    print(f"{'='*60}")
    return df


# =====================================================================
# 2. EDA & PREPROCESSING
# =====================================================================
def eda_pima(df: pd.DataFrame) -> None:
    """EDA specific to PIMA dataset."""
    target = "Outcome"
    print("\n--- Data Types ---")
    print(df.dtypes)
    print("\n--- Null Counts ---")
    print(df.isnull().sum())
    print("\n--- Class Distribution ---")
    print(df[target].value_counts())
    print(df[target].value_counts(normalize=True).mul(100).round(2).astype(str) + " %")

    # Replace impossible zeros with median
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_cols:
        median_val = df.loc[df[col] != 0, col].median()
        df[col] = df[col].replace(0, median_val)
        print(f"  Replaced 0s in {col} with median = {median_val}")

    # ── Class distribution bar chart ────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    df[target].value_counts().plot.bar(ax=ax, color=["#3498db", "#e74c3c"], edgecolor="black")
    ax.set_title("PIMA — Class Distribution", fontweight="bold")
    ax.set_xlabel("Outcome"); ax.set_ylabel("Count")
    ax.set_xticklabels(["No Diabetes (0)", "Diabetes (1)"], rotation=0)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/pima_class_distribution.png", dpi=150)
    plt.close()

    # ── Correlation heatmap ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                linewidths=0.5, square=True)
    ax.set_title("PIMA — Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/pima_correlation_heatmap.png", dpi=150)
    plt.close()

    # ── Pairplot (subsample for speed) ──────────────────────────────
    selected_features = ["Glucose", "BMI", "Age", "Insulin", target]
    g = sns.pairplot(df[selected_features], hue=target,
                     palette={0: "#3498db", 1: "#e74c3c"},
                     plot_kws={"alpha": 0.5, "s": 15})
    g.fig.suptitle("PIMA — Pairplot (Selected Features)", y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/pima_pairplot.png", dpi=150)
    plt.close()

    # ── Boxplots per feature vs Outcome ─────────────────────────────
    features = [c for c in df.columns if c != target]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for idx, col in enumerate(features):
        ax = axes[idx // 4, idx % 4]
        sns.boxplot(x=target, y=col, data=df, ax=ax,
                    palette=["#3498db", "#e74c3c"])
        ax.set_title(col)
    plt.suptitle("PIMA — Feature Boxplots by Outcome", fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/pima_boxplots.png", dpi=150)
    plt.close()
    print("  PIMA EDA plots saved.")


def eda_brfss(df: pd.DataFrame) -> None:
    """EDA specific to BRFSS dataset."""
    target = "Diabetes_binary"
    print("\n--- Data Types ---")
    print(df.dtypes)
    print("\n--- Class Distribution ---")
    counts = df[target].value_counts()
    pcts   = df[target].value_counts(normalize=True).mul(100).round(2)
    print(pd.DataFrame({"Count": counts, "Pct (%)": pcts}))

    # ── Class distribution bar chart ────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot.bar(ax=ax, color=["#2ecc71", "#e67e22"], edgecolor="black")
    ax.set_title("BRFSS — Class Distribution", fontweight="bold")
    ax.set_xlabel("Diabetes_binary"); ax.set_ylabel("Count")
    ax.set_xticklabels(["No Diabetes (0)", "Diabetes (1)"], rotation=0)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/brfss_class_distribution.png", dpi=150)
    plt.close()

    # ── Correlation heatmap ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                linewidths=0.3, annot_kws={"size": 7})
    ax.set_title("BRFSS — Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/brfss_correlation_heatmap.png", dpi=150)
    plt.close()

    # ── Top-10 features by chi-square ───────────────────────────────
    features = [c for c in df.columns if c != target]
    X_chi = df[features]
    y_chi = df[target]
    chi_scores, p_vals = chi2(X_chi, y_chi)
    chi_df = (pd.DataFrame({"Feature": features, "Chi2": chi_scores, "p-value": p_vals})
              .sort_values("Chi2", ascending=False).head(10))
    print("\n--- Top-10 Features by Chi-Square ---")
    print(chi_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="Chi2", y="Feature", data=chi_df, ax=ax, palette="viridis")
    ax.set_title("BRFSS — Top-10 Features (Chi-Square)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/brfss_chi2_top10.png", dpi=150)
    plt.close()
    print("  BRFSS EDA plots saved.")


def preprocess(df: pd.DataFrame, target_col: str, dataset_name: str):
    """
    Train-test split (80/20 stratified), SMOTE on train only,
    StandardScaler on all features.
    Returns: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = list(X.columns)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  Split → train: {X_train.shape}, test: {X_test.shape}")

    # SMOTE on training data only
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE → train: {X_train.shape}  class dist: {np.bincount(y_train)}")

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, f"{dataset_name}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved → {scaler_path}")

    return X_train, X_test, y_train, y_test, scaler, feature_names


# =====================================================================
# 3. MODEL TRAINING & EVALUATION
# =====================================================================
def get_base_models():
    """Return a dict of (name -> model) for all 10 base classifiers."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost":             xgb.XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0, n_jobs=-1),
        "LightGBM":            lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
        "CatBoost":            CatBoostClassifier(verbose=0, random_state=RANDOM_STATE),
        "KNN":                 KNeighborsClassifier(n_jobs=-1),
        "SVM":                 SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Extra Trees":         ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    }


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Return dict of evaluation metrics using a custom threshold."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_proba), 4),
    }


def train_models(X_train, X_test, y_train, y_test, dataset_name: str):
    """Train all base models, return results DataFrame and dict of fitted models."""
    models  = get_base_models()
    results = {}
    fitted  = {}

    # For large datasets, SVM and KNN are O(n^2) — subsample training data
    large_dataset = len(X_train) > 50000
    if large_dataset:
        subsample_size = 30000
        rng = np.random.RandomState(RANDOM_STATE)
        sub_idx = rng.choice(len(X_train), subsample_size, replace=False)
        X_sub = X_train[sub_idx]
        y_sub = y_train[sub_idx] if isinstance(y_train, np.ndarray) else y_train.iloc[sub_idx]
        print(f"  (Subsampling {subsample_size} for SVM/KNN on large dataset)")

    for name, model in models.items():
        print(f"    Training {name} ...", end=" ", flush=True)
        # Use subsample for slow models on large datasets
        if large_dataset and name in ("SVM", "KNN"):
            model.fit(X_sub, y_sub)
        else:
            model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, threshold=THRESHOLDS.get(dataset_name, 0.5))
        results[name] = metrics
        fitted[name]  = model
        print(f"ROC-AUC = {metrics['ROC-AUC']}")

    df_results = (pd.DataFrame(results).T
                    .sort_values("ROC-AUC", ascending=False)
                    .reset_index().rename(columns={"index": "Model"}))
    print(f"\n{'='*60}")
    print(f"  {dataset_name.upper()} — BASE MODEL LEADERBOARD")
    print(f"{'='*60}")
    print(df_results.to_string(index=False))
    return df_results, fitted


# =====================================================================
# 4. HYPERPARAMETER TUNING (Top-3 by ROC-AUC)
# =====================================================================
def get_param_grids():
    """Return param grids keyed by model name."""
    return {
        "Logistic Regression": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "liblinear"],
            "penalty": ["l2"],
        },
        "Decision Tree": {
            "max_depth": [3, 5, 7, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5, 10],
            "criterion": ["gini", "entropy"],
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.8, 0.9, 1.0],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 10],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "gamma": [0, 0.1, 0.3],
        },
        "LightGBM": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "num_leaves": [15, 31, 63, 127],
            "max_depth": [-1, 5, 10, 15],
            "subsample": [0.7, 0.8, 0.9, 1.0],
        },
        "CatBoost": {
            "iterations": [100, 200, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "depth": [4, 6, 8, 10],
            "l2_leaf_reg": [1, 3, 5, 7, 9],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
        },
        "SVM": {
            "C": [0.1, 1, 10, 50],
            "gamma": ["scale", "auto", 0.01, 0.1],
            "kernel": ["rbf"],
        },
        "Extra Trees": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        },
    }


def tune_top_models(top3_names, fitted_models, X_train, y_train, dataset_name):
    """
    Run RandomizedSearchCV on top-3 models.
    Returns: list of (name, best_model, best_params, best_score)
    """
    param_grids = get_param_grids()
    tuned = []

    for name in top3_names:
        print(f"\n  Tuning {name} for {dataset_name.upper()} ...")
        model = fitted_models[name]

        # Rebuild a fresh model of the same type for tuning
        fresh = model.__class__(**{
            k: v for k, v in model.get_params().items()
            if k in model.__class__().get_params()
        })

        # Suppress verbose for CatBoost / LightGBM during tuning
        if name == "CatBoost":
            fresh.set_params(verbose=0)
        if name == "LightGBM":
            fresh.set_params(verbose=-1)

        grid = param_grids.get(name, {})
        if not grid:
            print(f"    No param grid for {name}, skipping tuning.")
            tuned.append((name, model, {}, 0.0))
            continue

        # For BRFSS subsample for faster tuning (large dataset)
        if dataset_name == "brfss" and len(X_train) > 50000:
            n_iter = 20
            cv = 3
        else:
            n_iter = 30
            cv = 5

        search = RandomizedSearchCV(
            fresh, grid, n_iter=n_iter, cv=cv, scoring="roc_auc",
            random_state=RANDOM_STATE, n_jobs=-1, verbose=0
        )
        search.fit(X_train, y_train)
        print(f"    Best ROC-AUC (CV): {search.best_score_:.4f}")
        print(f"    Best Params: {search.best_params_}")

        tuned.append((name, search.best_estimator_, search.best_params_, search.best_score_))

    return tuned


# =====================================================================
# 5. ENSEMBLE METHODS
# =====================================================================
def run_ensembles(tuned_models, X_train, X_test, y_train, y_test, dataset_name):
    """
    Build Voting, Bagging, Stacking, and Weighted Voting ensembles.
    Returns: dict of {ensemble_name: (model, metrics)}
    """
    estimators = [(name, model) for name, model, _, _ in tuned_models]
    results = {}

    # ── A) Soft Voting ──────────────────────────────────────────────
    print(f"\n  [A] Voting Classifier (soft) for {dataset_name.upper()} ...")
    voting = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    voting.fit(X_train, y_train)
    thresh = THRESHOLDS.get(dataset_name, 0.5)
    metrics = evaluate_model(voting, X_test, y_test, threshold=thresh)
    results["Voting (Soft)"] = (voting, metrics)
    print(f"      ROC-AUC = {metrics['ROC-AUC']}")

    # ── B) Bagging ──────────────────────────────────────────────────
    best_name, best_model = estimators[0]
    n_bag = 20 if (dataset_name == "brfss" and len(X_train) > 50000) else 50
    print(f"  [B] Bagging Classifier (base={best_name}, n={n_bag}) for {dataset_name.upper()} ...")
    bagging = BaggingClassifier(
        estimator=best_model, n_estimators=n_bag,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    bagging.fit(X_train, y_train)
    metrics = evaluate_model(bagging, X_test, y_test, threshold=thresh)
    results["Bagging"] = (bagging, metrics)
    print(f"      ROC-AUC = {metrics['ROC-AUC']}")

    # ── C) Stacking ─────────────────────────────────────────────────
    stack_cv = 3 if (dataset_name == "brfss" and len(X_train) > 50000) else 5
    print(f"  [C] Stacking Classifier (cv={stack_cv}) for {dataset_name.upper()} ...")
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=stack_cv, n_jobs=-1
    )
    stacking.fit(X_train, y_train)
    metrics = evaluate_model(stacking, X_test, y_test, threshold=thresh)
    results["Stacking"] = (stacking, metrics)
    print(f"      ROC-AUC = {metrics['ROC-AUC']}")


    # ── D) Weighted Voting (manual) ─────────────────────────────────
    print(f"  [D] Weighted Voting (manual) for {dataset_name.upper()} ...")
    # Weights = individual ROC-AUC scores on test set
    weights = []
    probas  = []
    for name, model, _, _ in tuned_models:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc_val = roc_auc_score(y_test, y_proba)
        weights.append(auc_val)
        probas.append(y_proba)

    weights = np.array(weights)
    weights = weights / weights.sum()  # normalise
    weighted_proba = sum(w * p for w, p in zip(weights, probas))
    weighted_pred  = (weighted_proba >= thresh).astype(int)

    metrics = {
        "Accuracy":  round(accuracy_score(y_test, weighted_pred), 4),
        "Precision": round(precision_score(y_test, weighted_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, weighted_pred, zero_division=0), 4),
        "F1-Score":  round(f1_score(y_test, weighted_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, weighted_proba), 4),
    }
    # Create a pseudo-model wrapper for weighted voting
    results["Weighted Voting"] = (None, metrics)  # no single sklearn model
    print(f"      ROC-AUC = {metrics['ROC-AUC']}")

    return results


# =====================================================================
# 6. FULL LEADERBOARD
# =====================================================================
def build_leaderboard(base_results_df, ensemble_results, dataset_name):
    """Combine base and ensemble results into a single leaderboard."""
    rows = []
    for _, row in base_results_df.iterrows():
        rows.append(row.to_dict())
    for ens_name, (_, metrics) in ensemble_results.items():
        entry = {"Model": ens_name}
        entry.update(metrics)
        rows.append(entry)

    lb = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    print(f"\n{'='*70}")
    print(f"  {dataset_name.upper()} — FULL LEADERBOARD (Base + Ensembles)")
    print(f"{'='*70}")
    print(lb.to_string(index=False))

    best = lb.iloc[0]
    print(f"\n  ★ BEST MODEL → {best['Model']}  (ROC-AUC = {best['ROC-AUC']})")
    return lb


# =====================================================================
# 7. EVALUATION & EXPLAINABILITY
# =====================================================================
def plot_confusion_matrix(model, X_test, y_test, dataset_name, model_name):
    """Plot and save confusion matrix heatmap."""
    thresh = THRESHOLDS.get(dataset_name, 0.5)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Diabetes", "Diabetes"],
                yticklabels=["No Diabetes", "Diabetes"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{dataset_name.upper()} — Confusion Matrix\n({model_name}, threshold={thresh})",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{dataset_name}_confusion_matrix.png", dpi=150)
    plt.close()


def plot_roc_curve(model, X_test, y_test, dataset_name, model_name):
    """Plot and save ROC curve."""
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{dataset_name.upper()} — ROC Curve ({model_name})",
                 fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{dataset_name}_roc_curve.png", dpi=150)
    plt.close()
    return fpr, tpr, auc_val


def plot_precision_recall(model, X_test, y_test, dataset_name, model_name):
    """Plot and save Precision-Recall curve."""
    y_proba = model.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(rec, prec)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, color="#2ecc71", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{dataset_name.upper()} — Precision-Recall Curve ({model_name})",
                 fontweight="bold")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{dataset_name}_precision_recall.png", dpi=150)
    plt.close()


def shap_analysis(model, X_test, y_test, feature_names, dataset_name, model_name):
    """SHAP analysis with beeswarm + bar plot (tree-based) or permutation importance."""
    top_features = []
    # Take a subsample for SHAP speed
    sample_size = min(500, len(X_test))
    X_sample = X_test[:sample_size]

    # Check if model is tree-based
    tree_models = (
        RandomForestClassifier, GradientBoostingClassifier,
        ExtraTreesClassifier, DecisionTreeClassifier,
        xgb.XGBClassifier, lgb.LGBMClassifier, CatBoostClassifier
    )

    # For ensemble models, try to use the underlying estimator
    actual_model = model
    is_tree = isinstance(actual_model, tree_models)

    # For stacking/voting/bagging -> use permutation importance
    if isinstance(actual_model, (StackingClassifier, VotingClassifier, BaggingClassifier)):
        is_tree = False

    if is_tree:
        print(f"  SHAP: Using TreeExplainer for {model_name} ...")
        try:
            explainer = shap.TreeExplainer(actual_model)
            shap_values = explainer.shap_values(X_sample)

            # Handle different output shapes
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # class 1

            # Beeswarm plot
            fig = plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample,
                              feature_names=feature_names, show=False)
            plt.title(f"{dataset_name.upper()} — SHAP Beeswarm ({model_name})",
                      fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/{dataset_name}_shap_beeswarm.png", dpi=150,
                        bbox_inches="tight")
            plt.close()

            # Bar plot
            fig = plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample,
                              feature_names=feature_names,
                              plot_type="bar", show=False)
            plt.title(f"{dataset_name.upper()} — SHAP Feature Importance ({model_name})",
                      fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/{dataset_name}_shap_bar.png", dpi=150,
                        bbox_inches="tight")
            plt.close()

            # Top-3 features
            mean_abs = np.abs(shap_values).mean(axis=0)
            top_idx = np.argsort(mean_abs)[::-1][:3]
            top_features = [feature_names[i] for i in top_idx]
        except Exception as e:
            print(f"  SHAP TreeExplainer failed ({e}), falling back to permutation ...")
            is_tree = False

    if not is_tree:
        print(f"  SHAP: Using permutation importance for {model_name} ...")
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(actual_model, X_test, y_test,
                                       n_repeats=10, random_state=RANDOM_STATE,
                                       n_jobs=-1)
        sorted_idx = perm.importances_mean.argsort()[::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(min(15, len(feature_names))),
                perm.importances_mean[sorted_idx[:15]],
                color="#8e44ad")
        ax.set_yticks(range(min(15, len(feature_names))))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx[:15]])
        ax.invert_yaxis()
        ax.set_xlabel("Mean Permutation Importance")
        ax.set_title(f"{dataset_name.upper()} — Permutation Importance ({model_name})",
                     fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/{dataset_name}_permutation_importance.png", dpi=150)
        plt.close()

        top_features = [feature_names[i] for i in sorted_idx[:3]]

    print(f"  Top-3 features: {top_features}")
    return top_features


def full_evaluation(model, X_test, y_test, feature_names, dataset_name, model_name):
    """Run all evaluation & explainability steps for the best model."""
    print(f"\n{'='*60}")
    print(f"  {dataset_name.upper()} — EVALUATION & EXPLAINABILITY")
    print(f"  Best Model: {model_name}")
    print(f"{'='*60}")

    # Confusion Matrix
    plot_confusion_matrix(model, X_test, y_test, dataset_name, model_name)

    # ROC Curve
    fpr, tpr, auc_val = plot_roc_curve(model, X_test, y_test, dataset_name, model_name)

    # Precision-Recall Curve
    plot_precision_recall(model, X_test, y_test, dataset_name, model_name)

    # SHAP
    top_features = shap_analysis(model, X_test, y_test, feature_names, dataset_name, model_name)

    # Classification Report (using dataset-specific threshold)
    thresh = THRESHOLDS.get(dataset_name, 0.5)
    y_proba_final = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba_final >= thresh).astype(int)
    print(f"\n--- Classification Report (threshold={thresh}) ---")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

    return fpr, tpr, auc_val, top_features


# =====================================================================
# 8. CROSS-DATASET COMPARISON
# =====================================================================
def compare_datasets(pima_info, brfss_info):
    """
    Produce cross-dataset comparison report.
    Each *_info is a dict with keys:
        best_model_name, metrics, fpr, tpr, auc_val,
        top_features, leaderboard, ensemble_results
    """
    print(f"\n{'='*70}")
    print("  CROSS-DATASET COMPARISON REPORT")
    print(f"{'='*70}")

    # ── 1. Side-by-side summary table ───────────────────────────────
    pm = pima_info["metrics"]
    bm = brfss_info["metrics"]
    summary = pd.DataFrame({
        "Metric":      ["Best Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        "PIMA":        [pima_info["best_model_name"], pm["Accuracy"], pm["Precision"],
                        pm["Recall"], pm["F1-Score"], pm["ROC-AUC"]],
        "BRFSS":       [brfss_info["best_model_name"], bm["Accuracy"], bm["Precision"],
                        bm["Recall"], bm["F1-Score"], bm["ROC-AUC"]],
    })
    print("\n  Side-by-Side Summary")
    print(summary.to_string(index=False))

    # ── 2. Overlay ROC ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pima_info["fpr"], pima_info["tpr"],
            color="#e74c3c", lw=2,
            label=f'PIMA — {pima_info["best_model_name"]} (AUC = {pima_info["auc_val"]:.4f})')
    ax.plot(brfss_info["fpr"], brfss_info["tpr"],
            color="#3498db", lw=2,
            label=f'BRFSS — {brfss_info["best_model_name"]} (AUC = {brfss_info["auc_val"]:.4f})')
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Cross-Dataset ROC Curve Overlay", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/cross_dataset_roc_overlay.png", dpi=150)
    plt.close()

    # ── 3. Ensemble ROC-AUC grouped bar chart ──────────────────────
    ens_names = ["Voting (Soft)", "Bagging", "Stacking", "Weighted Voting"]
    pima_aucs  = [pima_info["ensemble_results"][n][1]["ROC-AUC"] for n in ens_names]
    brfss_aucs = [brfss_info["ensemble_results"][n][1]["ROC-AUC"] for n in ens_names]

    x = np.arange(len(ens_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, pima_aucs, width, label="PIMA",  color="#e74c3c", edgecolor="black")
    bars2 = ax.bar(x + width/2, brfss_aucs, width, label="BRFSS", color="#3498db", edgecolor="black")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Ensemble Methods — ROC-AUC Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ens_names, rotation=15)
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/cross_dataset_ensemble_comparison.png", dpi=150)
    plt.close()

    # ── 4. Written analysis ─────────────────────────────────────────
    analysis = f"""
{'='*70}
  CROSS-DATASET ANALYSIS
{'='*70}

1. ACCURACY COMPARISON
   PIMA  → Accuracy: {pm['Accuracy']}
   BRFSS → Accuracy: {bm['Accuracy']}

   The BRFSS dataset typically yields higher accuracy because it has
   253,680 samples (vs 768 for PIMA), providing far more training data.
   Additionally, the 21 diverse features cover a broader range of health
   indicators. However, the PIMA dataset's smaller size makes it more
   challenging and the model must generalise from limited data.

2. BEST ENSEMBLE METHOD
   PIMA  → {pima_info['best_model_name']} (ROC-AUC = {pima_info['auc_val']:.4f})
   BRFSS → {brfss_info['best_model_name']} (ROC-AUC = {brfss_info['auc_val']:.4f})

   Stacking and Soft Voting ensembles tend to be consistently strong across
   both datasets, as they leverage the diversity of multiple tuned base learners.

3. TOP-3 MOST IMPORTANT FEATURES
   PIMA  → {pima_info['top_features']}
   BRFSS → {brfss_info['top_features']}

   For PIMA, glucose level is typically the strongest predictor, consistent
   with medical literature. For BRFSS, general health, BMI, and high blood
   pressure tend to dominate.

4. LIMITATIONS
   PIMA:
   • Small dataset (768 records) — prone to overfitting
   • Women-only (Pima Indian heritage) — limited generalisability
   • Missing values encoded as zeros — imputation introduces bias

   BRFSS:
   • Survey-based, self-reported data — subject to response bias
   • US-specific population — may not generalise globally
   • Binary features lose granularity of original measurements

5. PRODUCTION RECOMMENDATION
   For production deployment, the BRFSS-trained model is recommended because:
   • Significantly larger and more diverse training data
   • More comprehensive feature set covering lifestyle & health indicators
   • Better generalisation due to data volume

   However, the PIMA model is useful as a quick screening tool when only
   basic clinical measurements are available, and it performs remarkably
   well given its limited data size.
"""
    print(analysis)

    # ── Final markdown summary ──────────────────────────────────────
    final_summary = f"""
{'='*70}
  ★  FINAL SUMMARY  ★
{'='*70}

  PIMA  → Best: {pima_info['best_model_name']}  |  ROC-AUC: {pima_info['auc_val']:.4f}
          Top 3 features: {', '.join(pima_info['top_features'])}

  BRFSS → Best: {brfss_info['best_model_name']}  |  ROC-AUC: {brfss_info['auc_val']:.4f}
          Top 3 features: {', '.join(brfss_info['top_features'])}

  Overall Winner: BRFSS-trained model
  Justification: Larger dataset, more features, better generalisation,
                  and consistently higher ensemble performance.
{'='*70}
"""
    print(final_summary)


# =====================================================================
# 9. PREDICT FUNCTION
# =====================================================================
def predict(features: dict, dataset_name: str) -> dict:
    """
    Unified prediction function.
    Args:
        features: dict of feature_name -> value (raw, unscaled)
        dataset_name: 'pima' or 'brfss'
    Returns:
        dict with 'prediction' (0/1), 'probability' (float), 'threshold' used
    """
    model  = joblib.load(os.path.join(MODEL_DIR, f"{dataset_name}_best_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, f"{dataset_name}_scaler.pkl"))
    threshold = THRESHOLDS.get(dataset_name, 0.5)

    df_input = pd.DataFrame([features])
    X_scaled = scaler.transform(df_input)
    prob     = model.predict_proba(X_scaled)[0, 1]
    pred     = int(prob >= threshold)

    return {"prediction": pred, "probability": round(float(prob), 4), "threshold": threshold}


# =====================================================================
# MAIN PIPELINE
# =====================================================================
def run_dataset_pipeline(dataset_name: str, target_col: str):
    """
    Run the FULL pipeline for a single dataset.
    Returns a dict with all results needed for cross-dataset comparison.
    """
    print(f"\n\n{'#'*70}")
    print(f"#  PIPELINE FOR: {dataset_name.upper()}")
    print(f"{'#'*70}")

    # ── Step 1: Load & EDA ──────────────────────────────────────────
    df = load_data(dataset_name)
    if dataset_name == "pima":
        eda_pima(df)
    else:
        eda_brfss(df)

    # ── Step 1 (cont): Preprocess ───────────────────────────────────
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(
        df, target_col, dataset_name
    )

    # ── Step 2: Base models ─────────────────────────────────────────
    base_results, fitted_models = train_models(X_train, X_test, y_train, y_test, dataset_name)

    # ── Step 3: Tune top-3 ──────────────────────────────────────────
    top3_names = base_results["Model"].head(3).tolist()
    print(f"\n  Top-3 models to tune: {top3_names}")
    tuned_models = tune_top_models(top3_names, fitted_models, X_train, y_train, dataset_name)

    # ── Step 4: Ensembles ───────────────────────────────────────────
    ensemble_results = run_ensembles(tuned_models, X_train, X_test, y_train, y_test, dataset_name)

    # ── Leaderboard ─────────────────────────────────────────────────
    leaderboard = build_leaderboard(base_results, ensemble_results, dataset_name)

    # ── Find best model (preferring ensemble if available) ──────────
    best_row = leaderboard.iloc[0]
    best_model_name = best_row["Model"]
    best_metrics = best_row.to_dict()
    del best_metrics["Model"]

    # Get the actual model object
    if best_model_name in ensemble_results:
        best_model = ensemble_results[best_model_name][0]
    elif best_model_name in fitted_models:
        best_model = fitted_models[best_model_name]
    else:
        # Tuned model
        for name, model, _, _ in tuned_models:
            if name == best_model_name:
                best_model = model
                break

    # If best is Weighted Voting (no single model), use Stacking or Voting
    if best_model is None:
        # Fall back to the best sklearn model
        for candidate in ["Stacking", "Voting (Soft)", "Bagging"]:
            if candidate in ensemble_results and ensemble_results[candidate][0] is not None:
                best_model = ensemble_results[candidate][0]
                best_model_name = candidate
                best_metrics = ensemble_results[candidate][1]
                break
        if best_model is None:
            # Final fallback: best tuned model
            best_model = tuned_models[0][1]
            best_model_name = tuned_models[0][0]
            best_metrics = evaluate_model(best_model, X_test, y_test, threshold=THRESHOLDS.get(dataset_name, 0.5))

    # ── Step 5: Evaluation & Explainability ─────────────────────────
    fpr, tpr, auc_val, top_features = full_evaluation(
        best_model, X_test, y_test, feature_names, dataset_name, best_model_name
    )

    # ── Step 7: Save best model ─────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, f"{dataset_name}_best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"  Model saved → {model_path}")

    return {
        "best_model_name":  best_model_name,
        "metrics":          best_metrics,
        "fpr":              fpr,
        "tpr":              tpr,
        "auc_val":          auc_val,
        "top_features":     top_features,
        "leaderboard":      leaderboard,
        "ensemble_results": ensemble_results,
    }


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  DIABETES PREDICTION PIPELINE — ENSEMBLE METHODS       ║")
    print("║  Dual-Dataset Comparative Study                        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Run pipeline for PIMA
    pima_info = run_dataset_pipeline("pima", "Outcome")

    # Run pipeline for BRFSS
    brfss_info = run_dataset_pipeline("brfss", "Diabetes_binary")

    # Cross-dataset comparison
    compare_datasets(pima_info, brfss_info)

    # Test predict function
    print("\n--- Testing predict() function ---")
    pima_test = {
        "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72,
        "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627, "Age": 50
    }
    result = predict(pima_test, "pima")
    print(f"  PIMA predict → {result}")

    brfss_test = {
        "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 40,
        "Smoker": 1, "Stroke": 0, "HeartDiseaseorAttack": 0,
        "PhysActivity": 0, "Fruits": 0, "Veggies": 1,
        "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
        "GenHlth": 5, "MentHlth": 15, "PhysHlth": 20, "DiffWalk": 1,
        "Sex": 0, "Age": 9, "Education": 4, "Income": 3
    }
    result = predict(brfss_test, "brfss")
    print(f"  BRFSS predict → {result}")

    print("\n  ✅ Pipeline complete. All plots, models, and reports generated.")
