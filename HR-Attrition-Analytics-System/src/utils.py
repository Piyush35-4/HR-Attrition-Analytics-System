"""
src/utils.py
-------------
Utility functions shared across the project.

Includes:
  - Saving and loading models
  - Printing evaluation metrics
  - Generating confusion matrix and ROC curve plots
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (required for Flask)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc, classification_report
)

# ─────────────────────────────────────────────
# Directory Paths
# ─────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


# ─────────────────────────────────────────────
# Model Save / Load
# ─────────────────────────────────────────────

def save_model(model, model_name):
    """
    Save a trained scikit-learn model to disk using joblib.

    Args:
        model: Trained model object
        model_name (str): Filename without extension (e.g. 'random_forest')
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    joblib.dump(model, path)
    print(f"[INFO] Model saved → {path}")


def load_model(model_name):
    """
    Load a trained model from disk.

    Args:
        model_name (str): Filename without extension

    Returns:
        Trained scikit-learn model
    """
    path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    model = joblib.load(path)
    print(f"[INFO] Model loaded ← {path}")
    return model


# ─────────────────────────────────────────────
# Evaluation Metrics
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Compute and print all key classification metrics.

    Metric explanations:
      Accuracy  → % of all predictions that were correct
      Precision → Of all predicted "Yes" (attrition), how many were actually "Yes"?
      Recall    → Of all actual "Yes" (attrition), how many did we catch?
      F1 Score  → Harmonic mean of Precision and Recall (best when classes are imbalanced)

    Args:
        model: Trained classifier
        X_test: Scaled test features
        y_test: True labels
        model_name (str): Label for printing

    Returns:
        dict: All metric values
    """
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'f1_score':  f1_score(y_test, y_pred, zero_division=0),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Metrics")
    print(f"{'='*50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1_score']:.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['No Attrition', 'Attrition']))

    return metrics


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test, model_name="Model"):
    """
    Generate and save a confusion matrix heatmap.

    A confusion matrix shows:
      - True Positives  (TP): Correctly predicted "will leave"
      - True Negatives  (TN): Correctly predicted "will stay"
      - False Positives (FP): Wrongly predicted "will leave" (false alarm)
      - False Negatives (FN): Wrongly predicted "will stay" (missed attrition)

    Args:
        model: Trained classifier
        X_test: Scaled test features
        y_test: True labels
        model_name (str): Used in plot title and filename
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Stay', 'Leave'],
        yticklabels=['Stay', 'Leave']
    )
    plt.title(f'Confusion Matrix — {model_name}', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    save_path = os.path.join(OUTPUTS_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved → {save_path}")


def plot_roc_curve(model, X_test, y_test, model_name="Model"):
    """
    Generate and save a ROC (Receiver Operating Characteristic) curve.

    The ROC curve shows the trade-off between:
      - True Positive Rate  (Sensitivity / Recall)
      - False Positive Rate (1 - Specificity)
    AUC (Area Under Curve) closer to 1.0 = better model.

    Args:
        model: Trained classifier (must support predict_proba)
        X_test: Scaled test features
        y_test: True labels
        model_name (str): Used in plot title and filename
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if not hasattr(model, 'predict_proba'):
        print(f"[WARN] {model_name} does not support predict_proba. Skipping ROC curve.")
        return

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {model_name}', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()

    filename = f"roc_curve_{model_name.lower().replace(' ', '_')}.png"
    save_path = os.path.join(OUTPUTS_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] ROC curve saved → {save_path}")


def plot_model_comparison(results_dict):
    """
    Bar chart comparing all trained models on four metrics.

    Args:
        results_dict (dict): {model_name: {metric: value, ...}, ...}
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [results_dict[m][metric] for m in models]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(), color=color)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.2f}',
                ha='center', va='bottom', fontsize=8
            )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison — All Metrics', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()

    save_path = os.path.join(OUTPUTS_DIR, 'model_comparison.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Model comparison chart saved → {save_path}")
