from matplotlib import markers
import matplotlib.pyplot as plt
import numpy as np
from ..logs.files.constants import TRAIN_ERROR_BY_EPOCH_FILE_PATH,TEST_ERROR_BY_EPOCH_FILE_PATH


def get_log_stats(file_path):
    """
    Plot the train error by epoch for the linear model.
    """

    epochs = {}
    with open(file_path, "r") as f:
        for line_index, line in enumerate(f):
            line_data = line.split()
            epoch_number = int(line_data[0])
            if epoch_number not in epochs:
                epochs[epoch_number] = []
            
            epochs[epoch_number].append(float(line_data[1]))

    errors_stdev = np.std(list(epochs.values()), axis=1)
    errors_mean = np.mean(list(epochs.values()), axis=1)

    return  epochs, errors_mean, errors_stdev
    
def get_network_log_stats(file_path, from_line_index=0, to_line_index=None):
    """
    Plot the train error by epoch for the linear model.
    """
    print(f"from line index: {from_line_index}")
    print(f"to line index: {to_line_index}")
    epochs = {}
    next_line_index = 0
    with open(file_path, "r") as f:
        for line_index, line in enumerate(f):
            if line_index >= from_line_index and (to_line_index is None or line_index <= to_line_index):
                line_data = line.split()
                epoch_number = int(line_data[0])
                if epoch_number not in epochs:
                    epochs[epoch_number] = []
            
                epochs[epoch_number].append(float(line_data[1]))
                next_line_index = line_index + 1

    errors_stdev = np.std(list(epochs.values()), axis=1)
    errors_mean = np.mean(list(epochs.values()), axis=1)

    return  epochs, errors_mean, errors_stdev, next_line_index

def plot_ej_2_linear():
    epochs, errors_mean_train, errors_stdev_train = get_log_stats(TRAIN_ERROR_BY_EPOCH_FILE_PATH)
    plot_epoch_vs_error_with_stdev(epochs, [errors_mean_train], [errors_stdev_train],['Training'])

def plot_ej_2_linear_with_batches(number_of_epochs=1, iterations_per_network=1, batches=[10, 20, 50, 100, 200]):
    # File structure:
    # epoch_number error
    # so every iterations per network the batch size is increased

    mean_errors = []
    stdev_errors = []
    legends = [f"Batch size = {batch}" for batch in batches]

    items = iterations_per_network * number_of_epochs - 1
    next_line_index = 0
    from_line_index = 0

    for _ in batches:
        next_line_index += items
        epochs, errors_mean, errors_stdev, next_line_index = get_network_log_stats(TRAIN_ERROR_BY_EPOCH_FILE_PATH, from_line_index=from_line_index, to_line_index=next_line_index)
        from_line_index = next_line_index
        mean_errors.append(errors_mean)
        stdev_errors.append(errors_stdev)

    plot_epoch_vs_error(epochs, mean_errors, stdev_errors, legends)

def plot_ej_2_non_linear_with_batches(number_of_epochs=1, iterations_per_network=1, batches=[10, 20, 50, 100, 200], test_batches=[]):
    # File structure:
    # epoch_number error
    # so every iterations per network the batch size is increased

    train_mean_errors = []
    train_stdev_errors = []

    test_mean_errors = []
    test_stdev_errors = []

    legends = [f"Batch size = {batch}" for batch in [*batches, *test_batches]]

    items = iterations_per_network * number_of_epochs - 1
    next_line_index = 0
    from_line_index = 0

    for _ in batches:
        next_line_index += items
        epochs, errors_mean, errors_stdev, ignore = get_network_log_stats(TRAIN_ERROR_BY_EPOCH_FILE_PATH, from_line_index=from_line_index, to_line_index=next_line_index)
        train_mean_errors.append(errors_mean)
        train_stdev_errors.append(errors_stdev)

        epochs, test_errors_mean, test_errors_stdev, next_line_index = get_network_log_stats(TEST_ERROR_BY_EPOCH_FILE_PATH, from_line_index=from_line_index, to_line_index=next_line_index)
        test_mean_errors.append(test_errors_mean)
        test_stdev_errors.append(test_errors_stdev)

        from_line_index = next_line_index


    print(f"train_mean_errors: {train_mean_errors}")
    print(f"train_stdev_errors: {train_stdev_errors}")
    print(f"test_mean_errors: {test_mean_errors}")
    print(f"test_stdev_errors: {test_stdev_errors}") # [[], []]

    mean_errors = np.append(train_mean_errors, test_mean_errors, axis=0)
    stdev_errors = np.append(train_stdev_errors, test_stdev_errors, axis=0)

    print(f"mean_errors: {mean_errors}")
    print(f"stdev_errors: {stdev_errors}")

    plot_epoch_vs_error(epochs, mean_errors, stdev_errors, legends)


def plot_ej_2_non_linear():
    epochs, errors_mean_train, errors_stdev_train = get_log_stats(TRAIN_ERROR_BY_EPOCH_FILE_PATH)
    epochs, errors_mean_test, errors_stdev_test = get_log_stats(TEST_ERROR_BY_EPOCH_FILE_PATH)
    plot_epoch_vs_error_with_stdev(epochs, [errors_mean_train,errors_mean_test], [errors_stdev_train,errors_stdev_test],['Training','Test'])

def plot_epoch_vs_error(epochs:dict, mean_errors_by_set, error_stdevs_by_set,legends):
    """
    Plot the evolution of the error over the epochs.
    """

    colors = ["red", "black", "magenta", "yellow", "cyan", "lightcoral", "darkgrey", "violet", "khaki", "paleturquoise"]

    for  i in range(len(mean_errors_by_set)):

        epochs_list = list(epochs.keys())

        # for index, value in enumerate(mean_errors_by_set[i]):
        #     print(index)
        #     # ax.annotate(str(round(value, 2)), xy=(index,value), xytext=(-7,7), textcoords='offset points')
        #     ax.text(epochs_list[index], value + 150, f"{round(value, 3)}", ha="center", va="bottom")

        plt.plot(epochs_list, mean_errors_by_set[i], marker='o', color=colors[i])
    
    plt.legend(legends,prop={'size': 10})
    plt.xlabel("Epochs",fontdict={"size":20})
    plt.ylabel("Error",fontdict={"size":20})
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yscale("log")
    plt.show()


def plot_epoch_vs_error_with_stdev(epochs:dict, mean_errors_by_set, error_stdevs_by_set,legends):
    """
    Plot the evolution of the error over the epochs.
    """

    colors = ["red", "black", "magenta", "yellow", "cyan", "lightcoral", "darkgrey", "violet", "khaki", "paleturquoise"]
    print(epochs)
    for  i in range(len(mean_errors_by_set)):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        epochs_list = list(epochs.keys())

        for index, value in enumerate(mean_errors_by_set[i]):
            print(index)
            # ax.annotate(str(round(value, 2)), xy=(index,value), xytext=(-7,7), textcoords='offset points')
            ax.text(epochs_list[index], value + 150, f"{round(value, 3)}", ha="center", va="bottom")

        plt.errorbar(epochs_list, mean_errors_by_set[i], yerr=error_stdevs_by_set[i],
                 ecolor='blue', marker='o', color=colors[i], elinewidth=0.5, capsize=5)
    
    plt.legend(legends)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    # plt.yscale("log")
    plt.show()

def plot_epochs_vs_error(epochs, errors_by_epoch):
    """
    Plot the evolution of the error over the epochs.
    """
    plt.plot(epochs, errors_by_epoch)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.yscale("log")
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
    plot_ej_2_non_linear_with_batches(number_of_epochs=50, iterations_per_network=5, batches=[10, 20, 50, 100, 200], test_batches=[10,20,50,100,200])
    # accuracies, precisions, recalls, f1s, scaled_predictions_errors, scaled_expected_output_errors = parse_metrics_file(filepath)
    # plot_epochs_vs_error(scaled_predictions_errors)
    # plot_epochs_vs_accuracy(accuracies)
    # plot_epochs_vs_precision(precisions)
    # plot_epochs_vs_precision_in_classes(precisions)
    # plot_epochs_vs_recall(recalls)
    # plot_epochs_vs_recall_in_classes(recalls)
    # plot_epochs_vs_f1(f1s)
    # plot_epochs_vs_f1_in_classes(f1s)
