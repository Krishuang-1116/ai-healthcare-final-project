# src/results_utils.py

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


def summarize_metrics_results(metrics_df):
    metrics_summary_df = (metrics_df.groupby('model')
                          [['auc_roc', 'average_precision', 'brier_score']]
                          .agg(['mean', 'std'])
                          .round(4))

    return metrics_summary_df


def plot_calibration_curve(preds_df, model_name):
    prob_true, prob_pred = calibration_curve(
        preds_df["y_true"],
        preds_df["y_prob"],
        n_bins=10,
        strategy="quantile"
    )

    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"{model_name} Calibration")
    plt.show()
