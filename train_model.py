from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)

def evaluate_proba(y_true, y_proba, threshold=0.5, title='Model'):
    y_pred = (y_proba >= threshold).astype(int)

    print(f'===== {title} =====')
    print(f'Threshold: {threshold:.4f}')
    print(f'ROC-AUC: {roc_auc_score(y_true, y_proba):.4f}')
    print(f'PR-AUC:  {average_precision_score(y_true, y_proba):.4f}')
    print(f'Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}')
    print(f'Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}')
    print(f'F1-score:  {f1_score(y_true, y_pred, zero_division=0):.4f}')
    print('Classification report:')
    print(classification_report(y_true, y_pred, target_names=['Não fraude', 'Fraude'], zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Não fraude', 'Fraude'])
    disp.plot(values_format='d')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_roc_pr_curves(y_true, y_proba, title='Model'):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, y_proba):.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title(f'Curva ROC — {title}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"PR-AUC = {average_precision_score(y_true, y_proba):.3f}")
    plt.title(f'Curva Precision-Recall — {title}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.tight_layout()
    plt.show()


# def best_f1_threshold(y_true, y_proba):
#     precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
#     f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
#     best_idx = np.nanargmax(f1)
#     return thresholds[best_idx], f1[best_idx], precision[best_idx], recall[best_idx]