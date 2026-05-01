"""
train_manual.py — Upgraded ML Pipeline for Genomic Mutation Classification
============================================================================
Improvements over the baseline:
  1. Rich 12-feature input (from upgraded dataset_gen.py)
  2. SMOTE oversampling to handle class imbalance
  3. Two strong models: XGBoost + GradientBoostingClassifier
  4. GridSearchCV / RandomizedSearchCV hyperparameter tuning
  5. 5-fold Stratified Cross-Validation for reliable evaluation
  6. Voting ensemble of the two best models
  7. Full metrics: accuracy, F1, AUC-ROC, confusion matrix
  8. Calibrated probabilities (CalibratedClassifierCV)
  9. Model + feature list saved together for safe inference
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings

warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    GradientBoostingClassifier,
    VotingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Try importing XGBoost — fall back to extra GradientBoosting if not available
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠  XGBoost not installed — using extra GradientBoostingClassifier instead.")

# Try importing imbalanced-learn for SMOTE
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("⚠  imbalanced-learn not installed — SMOTE will be skipped.")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/manual_dataset.csv"
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "manual_model.pkl")
META_PATH   = os.path.join(MODEL_DIR, "manual_model_meta.json")

# Feature columns (must match dataset_gen.py)
FEATURE_COLS = (
    ['ref_enc', 'alt_enc'] +
    [f'ref_{b}' for b in ['A', 'C', 'G', 'T']] +
    [f'alt_{b}' for b in ['A', 'C', 'G', 'T']] +
    ['is_transition', 'ref_is_gc', 'alt_is_gc',
     'purine_to_pyrimidine', 'pyrimidine_to_purine',
     'ref_alt_product', 'allele_delta']
)


# ─── Model definitions ────────────────────────────────────────────────────────
def build_models():
    models = {}

    # 1. Random Forest (strong baseline)
    models['rf'] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    # 2. Gradient Boosting
    models['gb'] = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    )

    # 3. XGBoost (if available)
    if HAS_XGB:
        models['xgb'] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
        )

    return models


# ─── Hyperparameter search (RandomizedSearchCV) ───────────────────────────────
def tune_rf(X_train, y_train):
    """Quick RandomizedSearch for Random Forest."""
    param_dist = {
        'n_estimators':     [100, 200, 300, 500],
        'max_depth':        [None, 10, 20, 30],
        'min_samples_split':[2, 5, 10],
        'max_features':     ['sqrt', 'log2', None],
    }
    base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=20,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='roc_auc',
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"  Best RF params: {search.best_params_}  |  CV AUC: {search.best_score_:.4f}")
    return search.best_estimator_


# ─── Main training function ────────────────────────────────────────────────────
def train_manual_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data not found at {DATA_PATH}. Run dataset_gen.py first.")
        return

    df = pd.read_csv(DATA_PATH)

    # Check feature columns exist (backwards-compat: old 2-feature dataset)
    available = [c for c in FEATURE_COLS if c in df.columns]
    if len(available) < 2:
        print("❌ No valid feature columns found in dataset.")
        return

    if len(available) < len(FEATURE_COLS):
        print(f"⚠  Only {len(available)}/{len(FEATURE_COLS)} features found — "
              "run dataset_gen.py again for full features.")

    X = df[available].values
    y = df['Target'].values

    print(f"\n📊 Dataset: {len(df)} samples | {len(available)} features")
    print(f"   Class balance → Benign: {(y==0).sum()} | Pathogenic: {(y==1).sum()}")

    # ── 2. Train/test split (stratified) ─────────────────────────────────────
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 3. SMOTE oversampling ────────────────────────────────────────────────
    if HAS_SMOTE:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"   After SMOTE → {len(X_train)} training samples (balanced)")

    # ── 4. Hyperparameter tuning for RF ──────────────────────────────────────
    print("\n🔍 Tuning Random Forest...")
    best_rf = tune_rf(X_train, y_train)

    # ── 5. Train all models ───────────────────────────────────────────────────
    models = build_models()
    models['rf'] = best_rf  # replace with tuned version

    trained_models = {}
    print("\n🏋  Training all models with 5-fold cross-validation...\n")
    for name, model in models.items():
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_aucs = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        print(f"  [{name.upper():3s}] CV AUC = {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")
        model.fit(X_train, y_train)
        trained_models[name] = model

    # ── 6. Voting Ensemble ────────────────────────────────────────────────────
    estimators = [(name, m) for name, m in trained_models.items()]
    ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    ensemble.fit(X_train, y_train)

    # ── 7. Calibrate probabilities ────────────────────────────────────────────
    calibrated = CalibratedClassifierCV(ensemble, cv=3, method='isotonic')
    calibrated.fit(X_train, y_train)

    # ── 8. Test-set evaluation ────────────────────────────────────────────────
    preds     = calibrated.predict(X_test)
    probs     = calibrated.predict_proba(X_test)[:, 1]
    acc       = accuracy_score(y_test, preds)
    f1        = f1_score(y_test, preds)
    auc       = roc_auc_score(y_test, probs)
    cm        = confusion_matrix(y_test, preds)

    print("\n" + "="*60)
    print("✅  FINAL TEST METRICS")
    print("="*60)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print("\n" + classification_report(y_test, preds, target_names=['Benign', 'Pathogenic']))

    # ── 9. Save model + metadata ──────────────────────────────────────────────
    joblib.dump(calibrated, MODEL_PATH)

    meta = {
        "feature_cols": available,
        "n_features":   len(available),
        "accuracy":     round(acc, 4),
        "f1":           round(f1, 4),
        "auc_roc":      round(auc, 4),
        "models_used":  list(trained_models.keys()),
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n💾  Model saved → {MODEL_PATH}")
    print(f"📋  Metadata  → {META_PATH}")


if __name__ == "__main__":
    train_manual_model()
