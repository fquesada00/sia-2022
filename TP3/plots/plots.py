import matplotlib.pyplot as plt
import numpy as np


def plot_epochs_vs_error(errors_by_epoch):
    """
    Plot the evolution of the error over the epochs.
    """
    plt.plot(errors_by_epoch)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()


def plot_epochs_vs_accuracy(accuracies_by_epoch):
    """
    Plot the evolution of the accuracy over the epochs.
    """
    plt.plot(accuracies_by_epoch)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()


def plot_epochs_vs_precision(precisions_by_epoch):
    """
    Plot the evolution of the precision over the epochs.
    """
    plt.plot(list(map(np.average, precisions_by_epoch)))
    plt.xlabel("Epochs")
    plt.ylabel("Precision")
    plt.show()


def plot_epochs_vs_precision_in_classes(precisions_by_epoch):
    """
    Plot the evolution of the precision over the epochs.
    """
    precision_matrix = np.array(precisions_by_epoch)
    for class_precision in precision_matrix.T:
        plt.plot(class_precision)
    plt.xlabel("Epochs")
    plt.ylabel("Precision")
    plt.legend([f'Class {i}' for i in range(len(precisions_by_epoch))])
    plt.show()


def plot_epochs_vs_recall(recalls_by_epoch):
    """
    Plot the evolution of the recall over the epochs.
    """
    plt.plot(list(map(np.average, recalls_by_epoch)))
    plt.xlabel("Epochs")
    plt.ylabel("Recall")
    plt.show()


def plot_epochs_vs_recall_in_classes(recalls_by_epoch):
    """
    Plot the evolution of the recall over the epochs.
    """
    recall_matrix = np.array(recalls_by_epoch)
    for class_recall in recall_matrix.T:
        plt.plot(class_recall)
    plt.xlabel("Epochs")
    plt.ylabel("Recall")
    plt.legend([f'Class {i}' for i in range(len(recalls_by_epoch))])
    plt.show()


def plot_epochs_vs_f1(f1s_by_epoch):
    """
    Plot the evolution of the f1 over the epochs.
    """
    plt.plot(list(map(np.average, f1s_by_epoch)))
    plt.xlabel("Epochs")
    plt.ylabel("F1")
    plt.show()


def plot_epochs_vs_f1_in_classes(f1s_by_epoch):
    """
    Plot the evolution of the f1 over the epochs.
    """
    f1_matrix = np.array(f1s_by_epoch)
    for class_f1 in f1_matrix.T:
        plt.plot(class_f1)
    plt.xlabel("Epochs")
    plt.ylabel("F1")
    plt.legend([f'Class {i}' for i in range(len(f1s_by_epoch))])
    plt.show()


def parse_metrics_file(filepath):
    """
    Parse the metrics file and return the errors by epoch.
    """
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    scaled_predictions_errors = []
    scaled_expected_output_errors = []
    number_of_metrics = 7
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            line_data = line.split()
            metric_index = i % number_of_metrics
            if metric_index == 0:
                epoch = int(line_data[0])
            elif metric_index == 1:
                accuracy = float(line_data[0])
                accuracies.append(accuracy)
            elif metric_index == 2:
                precision = list(map(float, line_data))
                precisions.append(precision)
            elif metric_index == 3:
                recall = list(map(float, line_data))
                recalls.append(recall)
            elif metric_index == 4:
                f1 = list(map(float, line_data))
                f1s.append(f1)
            elif metric_index == 5:
                error = float(line_data[0])
                scaled_predictions_errors.append(error)
            elif metric_index == 6:
                error = float(line_data[0])
                scaled_expected_output_errors.append(error)
    return accuracies, precisions, recalls, f1s, scaled_predictions_errors, scaled_expected_output_errors


if __name__ == "__main__":
    filepath = "TP3/metrics.txt"
    accuracies, precisions, recalls, f1s, scaled_predictions_errors, scaled_expected_output_errors = parse_metrics_file(filepath)
    plot_epochs_vs_error(scaled_predictions_errors)
    plot_epochs_vs_accuracy(accuracies)
    plot_epochs_vs_precision(precisions)
    plot_epochs_vs_precision_in_classes(precisions)
    plot_epochs_vs_recall(recalls)
    plot_epochs_vs_recall_in_classes(recalls)
    plot_epochs_vs_f1(f1s)
    plot_epochs_vs_f1_in_classes(f1s)
