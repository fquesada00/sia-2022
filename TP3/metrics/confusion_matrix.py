import numpy as np


def get_class(value):
    return np.where(value == np.max(value))[0][0]

    
# classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
def generate_confusion_matrix(expected_output, predicted_output, classes):
    confusion_matrix = np.zeros(shape=(len(classes), len(classes)))

    for i in range(len(expected_output)):
        confusion_matrix[get_class(expected_output[i])][get_class(predicted_output[i])] += 1

    print(confusion_matrix)

    return confusion_matrix