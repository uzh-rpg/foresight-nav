"""
Occupancy map metrics for evaluating the performance of occupancy map predictions.
This module provides functions to compute accuracy, precision-recall curves, 
and F1 scores for occupancy map predictions.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from torcheval.metrics import BinaryAccuracy


def accuracy(
        output: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
    """
    Computes accuracy of occupancy map predictions given the ground truth.
    """
    metric = BinaryAccuracy(threshold=threshold)
    metric.update(output.flatten(), target.flatten())
    return metric.compute().item() * 100

def accuracy_masked(
        output: torch.Tensor,
        target: torch.Tensor,
        input: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
    """
    Computes accuracy of occupancy map predictions only for regions 
    where input is 0.5 (unobserved) i.e. computes tp/(tp+fn) for masked regions.
    """
    mask = (input == 0.5).float()
    output = output * mask
    target = target * mask

    pred = (output > threshold).float()
    tp = torch.sum(pred * target)
    fp = torch.sum(pred * (1 - target))
    fn = torch.sum((1 - pred) * target)

    return (tp / (tp + fn)).item() * 100
    

def update_pr_curve(
        output: torch.Tensor,
        target: torch.Tensor,
        thresholds: torch.Tensor,
    ) -> tuple:
    """
    Computes the precision-recall curve for occupancy map predictions.
    """
    output = output.detach().cpu()
    target = target.detach().cpu()

    tp = torch.zeros(len(thresholds))
    fp = torch.zeros(len(thresholds))
    fn = torch.zeros(len(thresholds))
    
    for i, thresh in enumerate(thresholds):
        pred = (output > thresh).float()
        tp[i] = torch.sum(pred * target)
        fp[i] = torch.sum(pred * (1 - target))
        fn[i] = torch.sum((1 - pred) * target)

    return tp, fp, fn


def get_pr_curve(
        tp: torch.Tensor,
        fp: torch.Tensor,
        fn: torch.Tensor,
    ) -> tuple:
    """
    Computes the precision-recall curve for occupancy map predictions.
    """
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    # handle division by zero
    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0

    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    f1[np.isnan(f1)] = 0

    return precision, recall, f1

def get_best_threshold(
        precision: torch.Tensor,
        recall: torch.Tensor,
        f1: torch.Tensor,
        thresholds: torch.Tensor,
    ) -> tuple:
    """
    Computes the best threshold for the precision-recall curve.
    """
    best_idx = np.argmax(f1)
    return (
        thresholds[best_idx].item(),
        precision[best_idx],
        recall[best_idx],
        f1[best_idx]
    )

def plot_pr_curve(
        precision: torch.Tensor,
        recall: torch.Tensor,
        f1: torch.Tensor,
        thresholds: torch.Tensor,
    ):
    """
    Plots the precision-recall curve for occupancy map predictions.
    """

    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.plot(recall, f1, label='F1 score')

    best_thresh, best_prec, best_rec, best_f1 = get_best_threshold(precision, recall, f1, thresholds)
    plt.plot(best_rec, best_prec, 'ro', label=f'Best threshold: {best_thresh:.2f}\nPrecision: {best_prec:.2f}\nRecall: {best_rec:.2f}\nF1: {best_f1:.2f}')

    # display thresholds
    for i, thresh in enumerate(thresholds):
        plt.text(recall[i], precision[i], f'{thresh:.2f}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend()
    
    return plt.gcf()
