from tqdm import tqdm
import pandas as pd
import os
from torch.utils.data import DataLoader
import torch 
import matplotlib.pyplot as plt
import numpy as np


from dataset import MRI_dataset
from args import get_args
args = get_args()
def get_normalization_parameters(args):
    """_summary_

    Args:
        args : args to read csv

    Returns:
        mean, std: params for normalization

    """
    train_set = pd.read_csv(os.path.join(args.csv_dir, "data.csv"))
    train_dataset = MRI_dataset(dataset=train_set)

    # get loader
    loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    mean = 0.
    std = 0.
    n_samples = 0.

    for images in loader:
        images = images["img"]
        images = images.view(images.size(0), images.size(1), -1)  # Flatten H x W into a vector
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_samples += images.size(0)

    mean /= n_samples
    std /= n_samples

    return mean, std

# Mean: tensor([0.1543, 0.1543, 0.1543])
# Std: tensor([0.1668, 0.1668, 0.1668])

def plot_summary_metrics(df, fold, out_dir):
        """
        Plots the training and validation metrics over all epochs after training completes.
        Saves the summary plot in the specified output directory.

        Args:
        - df (pd.DataFrame): DataFrame containing the training and validation metrics over epochs.
        - fold (int): The fold number for which metrics are being plotted.
        """
        # Set up subplots for all metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Summary of Training and Validation Metrics for Fold {fold}')

        # Plot training and validation loss
        axes[0, 0].plot(df['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(df['val_loss'], label='Validation Loss', color='orange')
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()

        # Plot balanced accuracy
        axes[0, 1].plot(df['train_balanced_acc'], label='Train Balanced Accuracy', color='blue')
        axes[0, 1].plot(df['val_balanced_acc'], label='Validation Balanced Accuracy', color='orange')
        axes[0, 1].set_title("Balanced Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Balanced Accuracy")
        axes[0, 1].legend()

        # Plot ROC-AUC score
        axes[1, 0].plot(df['train_roc_auc'], label='Train ROC-AUC', color='blue')
        axes[1, 0].plot(df['val_roc_auc'], label='Validation ROC-AUC', color='orange')
        axes[1, 0].set_title("ROC-AUC")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("ROC-AUC")
        axes[1, 0].legend()

        # Plot average precision
        axes[1, 1].plot(df['train_avg_precision'], label='Train Average Precision', color='blue')
        axes[1, 1].plot(df['val_avg_precision'], label='Validation Average Precision', color='orange')
        axes[1, 1].set_title("Average Precision")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Average Precision")
        axes[1, 1].legend()

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save the summary plot
        plot_filename = os.path.join(out_dir, f"summary_metrics_fold_{fold}.png")
        plt.savefig(plot_filename)
        
        plt.show()
        plt.close(fig)


