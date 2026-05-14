"""
src/train_model.py
-------------------
Trains three ML classification models on the HR Attrition dataset:
  1. Logistic Regression  — Baseline linear model
  2. Decision Tree        — Simple tree-based model, highly interpretable
  3. Random Forest        — Ensemble of trees, typically best accuracy

Also performs:
  - GridSearchCV hyperparameter tuning on Random Forest
  - Cross-validation
  - Feature importance analysis
  - Model comparison chart

Run this script directly to train all models:
  python src/train_model.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing import preprocess_for_training
from src.utils import save_model, evaluate_model, plot_confusion_matrix, plot_roc_curve, plot_model_comparison

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model.

    Logistic Regression is a simple algorithm that predicts the probability
    of an outcome (leave / stay) using a sigmoid function.
    It works well as a baseline and is very explainable.

    Args:
        X_train: Scaled training features
        y_train: Training labels

    Returns:
        Trained LogisticRegression model
    """
    print("\n[TRAINING] Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,   # Allow enough iterations for convergence
        random_state=42,
        class_weight='balanced'  # Handle class imbalance (fewer attrition samples)
    )
    model.fit(X_train, y_train)
    print("[INFO] Logistic Regression training complete.")
    return model


def train_decision_tree(X_train, y_train):
    """
    Train a Decision Tree classifier.

    A Decision Tree splits data on the most informative features
    (based on Gini impurity) until it reaches a prediction.
    Very easy to visualise and explain.

    Args:
        X_train: Scaled training features
        y_train: Training labels

    Returns:
        Trained DecisionTreeClassifier model
    """
    print("\n[TRAINING] Decision Tree...")
    model = DecisionTreeClassifier(
        max_depth=6,         # Limit depth to prevent overfitting
        min_samples_split=20,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    print("[INFO] Decision Tree training complete.")
    return model


def train_random_forest(X_train, y_train, tune_hyperparams=False):
    """
    Train a Random Forest classifier (with optional GridSearchCV tuning).

    A Random Forest builds many decision trees on random subsets of the data
    and averages their predictions. This reduces overfitting and usually
    gives the best accuracy.

    Args:
        X_train: Scaled training features
        y_train: Training labels
        tune_hyperparams (bool): If True, run GridSearchCV (slower but better)

    Returns:
        Trained RandomForestClassifier model
    """
    print("\n[TRAINING] Random Forest...")

    if tune_hyperparams:
        print("[INFO] Running GridSearchCV (this may take a few minutes)...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 10, None],
            'min_samples_split': [10, 20],
        }
        base_rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(
            base_rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"[INFO] Best params: {grid_search.best_params_}")
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)

    print("[INFO] Random Forest training complete.")
    return model


def cross_validate_model(model, X_train, y_train, model_name="Model"):
    """
    Run 5-fold cross-validation and report average F1 score.

    Cross-validation trains the model on different splits of the training data
    to give a more reliable estimate of real-world performance.

    Args:
        model: Trained classifier
        X_train: Training features
        y_train: Training labels
        model_name (str): Label for display
    """
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"\n[Cross-Validation] {model_name}")
    print(f"  F1 scores across 5 folds: {[f'{s:.3f}' for s in scores]}")
    print(f"  Mean F1: {scores.mean():.4f} ± {scores.std():.4f}")


def plot_feature_importance(model, feature_names, model_name="Random Forest"):
    """
    Plot and save a bar chart of the most important features.

    Feature importance tells us which employee attributes have the greatest
    influence on whether the model predicts attrition.

    Args:
        model: Fitted tree-based model (Random Forest or Decision Tree)
        feature_names (list): Names of all features
        model_name (str): Label for plot title
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if not hasattr(model, 'feature_importances_'):
        print(f"[WARN] {model_name} does not support feature_importances_")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features

    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, palette='viridis')
    plt.title(f'Top 15 Feature Importances — {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()

    save_path = os.path.join(OUTPUTS_DIR, 'feature_importance.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Feature importance chart saved → {save_path}")


def run_training_pipeline(tune_hyperparams=False):
    """
    Main training pipeline.
    Preprocesses data, trains all three models, evaluates, and saves.

    Args:
        tune_hyperparams (bool): If True, run GridSearchCV for Random Forest

    Returns:
        dict: All trained models keyed by name
    """
    print("\n" + "="*60)
    print("  HR ATTRITION — MODEL TRAINING PIPELINE")
    print("="*60)

    # ── Step 1: Preprocess data ──────────────────────────────────
    X_train, X_test, y_train, y_test, feature_names, scaler, encoders = preprocess_for_training()

    # ── Step 2: Train all models ─────────────────────────────────
    lr_model  = train_logistic_regression(X_train, y_train)
    dt_model  = train_decision_tree(X_train, y_train)
    rf_model  = train_random_forest(X_train, y_train, tune_hyperparams=tune_hyperparams)

    models = {
        'Logistic Regression': lr_model,
        'Decision Tree':       dt_model,
        'Random Forest':       rf_model,
    }

    # ── Step 3: Evaluate all models ──────────────────────────────
    results = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        results[name] = metrics
        plot_confusion_matrix(model, X_test, y_test, model_name=name)
        plot_roc_curve(model, X_test, y_test, model_name=name)

    # ── Step 4: Cross-validation ─────────────────────────────────
    for name, model in models.items():
        cross_validate_model(model, X_train, y_train, model_name=name)

    # ── Step 5: Feature importance (Random Forest) ───────────────
    plot_feature_importance(rf_model, feature_names, model_name="Random Forest")

    # ── Step 6: Model comparison chart ──────────────────────────
    plot_model_comparison(results)

    # ── Step 7: Save all models ──────────────────────────────────
    for name, model in models.items():
        save_model(model, name.lower().replace(' ', '_'))

    # Print final winner
    best_model = max(results, key=lambda m: results[m]['f1_score'])
    print(f"\n{'='*60}")
    print(f"  BEST MODEL (by F1 Score): {best_model}")
    print(f"  F1 Score: {results[best_model]['f1_score']:.4f}")
    print(f"{'='*60}\n")

    return models


# ── Entry point ──────────────────────────────────────────────────
if __name__ == '__main__':
    # Set tune_hyperparams=True to run GridSearchCV (takes longer)
    run_training_pipeline(tune_hyperparams=False)
