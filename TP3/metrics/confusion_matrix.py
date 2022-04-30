import numpy as np


def get_class(value):
    return int(value / 10)

    
# classes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
def generate_confusion_matrix(expected_output, predicted_output, classes):
    confusion_matrix = np.zeros((len(classes), len(classes)))

    for i in range(len(expected_output)):
        confusion_matrix[get_class(expected_output[i])][get_class(predicted_output[i])] += 1

    return confusion_matrix