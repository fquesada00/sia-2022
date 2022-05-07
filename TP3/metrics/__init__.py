import numpy as np

from .Metrics import Metrics


def print_metrics(confusion_matrix):
    # accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    # precision = TP / (TP + FP)
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    # recall = TP / (TP + FN)
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    # f1 = 2 * precision * recall / (precision + recall)
    f1 = 2 * precision * recall / (precision + recall)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)


def generate_metrics(confusion_matrix):
    # accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    # precision = TP / (TP + FP)
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    # recall = TP / (TP + FN)
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    # f1 = 2 * precision * recall / (precision + recall)
    f1 = 2 * precision * recall / (precision + recall)

    return Metrics(accuracy, precision, recall, f1)
