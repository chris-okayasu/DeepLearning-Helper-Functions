import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

def fun_visualize_training_history(history, save=False, save_path="training_metrics.png", ylim=None, font_size=12):
    """
    Visualizes the training history (plots and table).

    Args:
        history (tf.keras.callbacks.History): Model training history.
        save (bool, optional): If set to True, saves the plots. Default is False.
        save_path (str, optional): Path to save the plot. Used only if save=True.
        ylim (tuple, optional): Y-axis limits.
        font_size (int, optional): Font size for the plots.
    """
    # Font configuration
    mpl.rcParams.update({'font.size': font_size})

    # Convert history to DataFrame for tabular view
    history_df = pd.DataFrame(history.history)

    # Display table of metrics (last row with values)
    print("Table of Metrics:")
    display(history_df)

    # Show final metrics per epoch
    final_metrics = history_df.iloc[-1]
    print("\nFinal Metrics per Epoch:")
    print(final_metrics.to_string())

    # Create a list of training metrics (excluding validation metrics)
    metrics = [col for col in history_df.columns if not col.startswith("val_")]

    # Colors for the plots
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Plot training and validation metrics
    plt.figure(figsize=(12, len(metrics) * 4))

    for i, metric in enumerate(metrics, start=1):
        plt.subplot(len(metrics), 1, i)

        # Training metric plot
        plt.plot(history_df[metric], label=f"Training {metric}", color=colors[i % len(colors)])

        # Validation metric plot if it exists
        val_metric = f"val_{metric}"
        if val_metric in history_df.columns:
            plt.plot(history_df[val_metric], label=f"Validation {metric}", linestyle="--", color=colors[i % len(colors)])

        # Add title and labels
        plt.title(f"{metric.capitalize()}")
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())

        # Adjust Y-axis limits if provided
        if ylim:
            plt.ylim(ylim)

        plt.legend()

    # Layout adjustment
    plt.tight_layout()
    plt.show()

    # Plot all metrics together
    plt.figure(figsize=(12, 6))
    history_df.plot(ax=plt.gca())
    plt.title("Training and Validation Metrics Combined")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Values")
    plt.legend(loc="upper right")

    if ylim:
        plt.ylim(ylim)

    plt.show()

    # Save plots if requested
    if save:
        plt.savefig(save_path, dpi=300)
        print(f"Graphs saved to: {save_path}")
