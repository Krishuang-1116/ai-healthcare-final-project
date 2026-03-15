# src/metrics_utils.py

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def compute_metrics(y_true, y_pred_proba) -> dict:
    """
    Compute evaluation metrics.

    Parameters:
    y_true (array-like): True binary labels.
    y_pred_proba (array-like): Predicted probabilities for the positive class.

    Returns:
    dict: A dictionary containing the computed metrics.
    """
    metrics = {}
    # Compute AUC-ROC
    metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
    # Compute Average Precision Score
    metrics['average_precision'] = average_precision_score(
        y_true, y_pred_proba)
    # Compute Brier Score Loss
    metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)

    return metrics
