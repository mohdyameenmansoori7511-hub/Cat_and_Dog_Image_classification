"""Accuracy, Precision, Recall and Confusion Matrix — """
from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_binary_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray) -> Dict[str, Any]:
    """Computes Accuracy, Precision, Recall, F1-Score, and Confusion Matrix at threshold 0.5."""
    y_pred = (y_pred_probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
