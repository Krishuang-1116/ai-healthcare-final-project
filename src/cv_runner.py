# src/cv_runner.py
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import N_SPLITS, RANDOM_STATE
from src.metrics_utils import compute_metrics


def run_cv_model(
        X: pd.DataFrame,
        y: pd.DataFrame,
        ids: pd.DataFrame,
        model_name: str,
        fit_predict_fn: callable
):
    '''
    Run cross-validation for a given model
    '''
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)

    metrics_rows = []
    pred_rows = []
    params_rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        ids_test = ids.iloc[test_idx]

        # Fit a model and get predictions

        result = fit_predict_fn(X_train, y_train, X_test)

        if isinstance(result, tuple):
            y_prob, best_params = result
        else:
            y_prob = result
            best_params = {}

        # Compute and store metrics
        fold_metrics = compute_metrics(y_test, y_prob)
        fold_metrics['model'] = model_name
        fold_metrics['fold'] = fold
        metrics_rows.append(fold_metrics)

        # Store predictions for test set
        fold_preds = ids_test.copy()
        fold_preds['model'] = model_name
        fold_preds['fold'] = fold
        fold_preds['y_true'] = y_test
        fold_preds['y_prob'] = y_prob
        pred_rows.append(fold_preds)

        # Store best hyperparameters
        if best_params:
            row = {"model": model_name, "fold": fold}
            row.update(best_params)
            params_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    pred_df = pd.concat(pred_rows, axis=0, ignore_index=True)
    params_df = pd.DataFrame(params_rows) if params_rows else None

    return metrics_df, pred_df, params_df
