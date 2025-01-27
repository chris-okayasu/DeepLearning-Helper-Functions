import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def fun_compare_histories(histories, labels=None, save=False, save_path=None):
    """
    Compare multiple Keras/TensorFlow training histories.

    Args:
    - histories: List of Keras/TensorFlow history objects.
    - labels: List of model names (for legend). If None, defaults to "Model 1", "Model 2", etc.
    - save: Boolean, if True, saves plots and results as text and CSV files.
    - save_path: Folder to save the files (default: 'model_comparison_results' in the root directory).
    """

    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(histories))]

    if save and save_path is None:
        save_path = os.path.join(os.getcwd(), "model_comparison_results")

    colors = ['deepskyblue', 'lime', 'crimson', 'gold', 'magenta', 'darkorange',
              'orchid', 'turquoise', 'chartreuse', 'blueviolet', 'coral', 'mediumspringgreen',
              'dodgerblue', 'hotpink', 'greenyellow', 'firebrick', 'steelblue', 'darkviolet']

    metrics = ["accuracy", "loss", "val_accuracy", "val_loss"]
    available_metrics = set(histories[0].history.keys())
    selected_metrics = [m for m in metrics if m in available_metrics]

    num_metrics = len(selected_metrics)
    fig, axes = plt.subplots((num_metrics + 1) // 2, 2, figsize=(12, 4 * ((num_metrics + 1) // 2)))

    if num_metrics == 1:
        axes = np.array([axes])

    metric_results = {metric: {} for metric in selected_metrics}

    for i, (ax, metric) in enumerate(zip(axes.flat, selected_metrics)):
        values_table = []

        for j, history in enumerate(histories):
            values = history.history[metric]
            final_value = values[-1]
            metric_results[metric][labels[j]] = final_value
            ax.plot(values, label=f"{labels[j]}: {final_value:.4f}", color=colors[j % len(colors)], linewidth=2)
            values_table.append([labels[j], f"{final_value:.4f}"])

        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric.capitalize())
        ax.legend(loc="upper left")
        ax.set_title(f"Comparison of {metric.capitalize()}")

    plt.tight_layout()

    if save:
        os.makedirs(save_path, exist_ok=True)
        plot_filename = os.path.join(save_path, "model_comparison.png")
        plt.savefig(plot_filename, dpi=300)
        print(f"📷 Plot saved at: {plot_filename}")

    plt.show()

    summary_text = "\n📊 **Model Performance Summary:**\n"

    def highlight_best(metric, is_higher_better=True):
        """Highlight the best model for a given metric."""
        if metric not in available_metrics:
            return None
        best_model = max(metric_results[metric], key=metric_results[metric].get) if is_higher_better else \
                     min(metric_results[metric], key=metric_results[metric].get)
        best_value = metric_results[metric][best_model]
        arrow = "🟢✅⬆️" if is_higher_better else "🔴⚠️⬇️"
        return f"{arrow} Best {metric.capitalize()}: {best_model} → {best_value:.4f}"

    model_scores = {label: 0 for label in labels}

    for metric in selected_metrics:
        result = highlight_best(metric, is_higher_better=("loss" not in metric))
        if result:
            summary_text += result + "\n"
            best_model_name = result.split(": ")[1].split(" → ")[0]
            model_scores[best_model_name] += 1

    best_model = max(model_scores, key=model_scores.get)
    summary_text += "\n🏆 **Overall Best Model:**\n"
    summary_text += f"🎯 {best_model} with {model_scores[best_model]} best metrics!\n"

    summary_text += "\n📜 **Emoji Legend:**\n"
    summary_text += "✅⬆️ Best (higher is better) → Accuracy, Precision, Recall, F1-score\n"
    summary_text += "🔴⚠️⬇️ Best (lower is better) → Loss, Validation Loss\n"
    summary_text += "🎯 Overall Best Model → The model with the most best metrics\n"
    summary_text += "🏆 Winner → The best-performing model in general\n"

    print(summary_text)

    if save:
        summary_filename = os.path.join(save_path, "model_comparison.txt")
        with open(summary_filename, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"📝 Summary saved at: {summary_filename}")
