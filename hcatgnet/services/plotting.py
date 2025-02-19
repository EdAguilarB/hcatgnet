import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def create_st_parity_plot(real, predicted, figure_name, save_path=None):
    """
    Create a parity plot and display R2, MAE, and RMSE metrics.

    Args:
        real (numpy.ndarray): An array of real (actual) values.
        predicted (numpy.ndarray): An array of predicted values.
        save_path (str, optional): The path where the plot should be saved. If None, the plot is not saved.

    Returns:
        matplotlib.figure.Figure: The Matplotlib figure object.
        matplotlib.axes._axes.Axes: The Matplotlib axes object.
    """
    # Calculate R2, MAE, and RMSE
    r2 = r2_score(real, predicted)
    mae = mean_absolute_error(real, predicted)
    rmse = np.sqrt(mean_squared_error(real, predicted))

    # Create the parity plot
    plt.figure(figsize=(8, 8))
    plt.scatter(real, predicted, alpha=0.7)
    plt.plot(
        [min(real), max(real)], [min(real), max(real)], color="red", linestyle="--"
    )
    plt.xlabel("Real Values")
    plt.ylabel("Predicted Values")

    # Display R2, MAE, and RMSE as text on the plot
    textstr = f"$R^2$ = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}"
    plt.gcf().text(0.15, 0.75, textstr, fontsize=12)

    # Save the plot if save_path is provided
    if save_path:
        # Ensure the directory exists
        save_path = os.path.join(save_path, figure_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.close()


def create_training_plot(df, save_path):

    df = pd.read_csv(df)

    epochs = df.iloc[:, 0]
    train_loss = df.iloc[:, 1]
    val_loss = df.iloc[:, 2]
    test_loss = df.iloc[:, 3]

    min_val_loss_epoch = epochs[val_loss.idxmin()]

    # Create a Matplotlib figure and axis
    plt.figure(figsize=(10, 6), dpi=300)  # Adjust the figure size as needed
    plt.plot(epochs, train_loss, label="Train Loss", marker="o", linestyle="-")
    plt.plot(epochs, val_loss, label="Validation Loss", marker="o", linestyle="-")
    plt.plot(epochs, test_loss, label="Test Loss", marker="o", linestyle="-")

    plt.axvline(
        x=min_val_loss_epoch,
        color="gray",
        linestyle="--",
        label=f"Min Validation Epoch ({min_val_loss_epoch})",
    )

    # Customize the plot
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(False)
    plt.legend()

    # Save the plot in high resolution (adjust file format as needed)
    plt.savefig("{}/loss_vs_epochs.png".format(save_path), bbox_inches="tight")

    plt.close()


def plot_tsne_with_subsets(
    data_df,
    feature_columns,
    color_column,
    set_column,
    fig_name=None,
    save_path=None,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    show=False,
):
    # Perform t-SNE
    features = data_df[feature_columns]
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=42,
    )
    tsne_results = tsne.fit_transform(features)

    # Add t-SNE results back to the DataFrame
    data_df["tSNE1"] = tsne_results[:, 0]
    data_df["tSNE2"] = tsne_results[:, 1]

    # Define subsets
    subsets = data_df[set_column].unique()

    # Create subplots
    fig, axes = plt.subplots(
        1, len(subsets), figsize=(20, 8), sharex=True, sharey=True, dpi=300
    )

    # Plot each subset
    for i, subset in enumerate(subsets):
        subset_df = data_df[data_df[set_column] == subset]
        scatter = axes[i].scatter(
            subset_df["tSNE1"],
            subset_df["tSNE2"],
            c=subset_df[color_column],
            cmap="plasma",
            s=100,
            alpha=0.7,
            edgecolors="w",
            linewidth=0.5,
        )
        axes[i].set_title(f"{subset.capitalize()} Set", fontsize=18, pad=15)
        if i == 0:
            axes[i].set_ylabel("tSNE2", fontsize=20, labelpad=15)
        if i == 1:
            axes[i].set_xlabel("tSNE1", fontsize=20, labelpad=15)
        axes[i].grid(True, linestyle="--", alpha=0.6)
        axes[i].tick_params(axis="both", which="major", labelsize=12)

    # Add color bar to the right
    cbar = fig.colorbar(
        scatter, ax=axes, orientation="vertical", fraction=0.02, pad=0.02
    )
    cbar.set_label("$\Delta \Delta G$", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Enhancing the overall look
    plt.suptitle("t-SNE 2D Visualization by Set", fontsize=22, y=1.05)
    sns.despine()
    # plt.tight_layout(rect=[0, 0, 0.95, 1])  # Adjust layout to make room for color bar
    if save_path and fig_name:
        # Ensure the directory exists
        save_path = os.path.join(save_path, fig_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()
