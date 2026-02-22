import numpy as np
from sklearn.metrics import f1_score


def compute_metrics(eval_pred):
    """Compute multi-label metrics for the HuggingFace Trainer.

    Applies sigmoid + threshold of 0.5 to logits, then computes
    F1 macro, F1 micro, and per-emotion F1 scores.
    """
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs >= 0.5).astype(int)
    labels = labels.astype(int)

    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    f1_per_label = f1_score(labels, preds, average=None, zero_division=0)

    metrics = {
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
    }
    for i, score in enumerate(f1_per_label):
        metrics[f"f1_label_{i}"] = score

    return metrics
