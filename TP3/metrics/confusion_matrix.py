import numpy as np


def get_parity_class(value):
    return 1 if (1 - value[0]) < 0.5 else 0


def get_numbers_class(value):
    return np.where(value == np.max(value))[0][0]


def get_xor_class(value):
    return 1 if value[0] != value[1] else 0


def get_ej2_class(value):
    index = int(value) // 10
    if abs(index) > 9:
        index = 9
    return index


def generate_confusion_matrix(expected_output, predicted_output, dataset):
    if dataset in ['parity', 'parity_noise']:
        classes = [0, 1]
        get_class = get_parity_class
    elif dataset in ['number', 'number_noise']:
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        get_class = get_numbers_class
    elif dataset in ['xor_simple, xor_multilayer', 'and']:
        classes = [-1, 1]
        get_class = get_xor_class
    elif dataset in ['ej_2_linear', 'ej_2_non_linear']:
        classes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        get_class = get_ej2_class

    else:
        raise Exception(f'Dataset {dataset} not found')

    confusion_matrix = np.zeros(shape=(len(classes), len(classes)))

    for i in range(len(expected_output)):
        print(predicted_output[i])
        confusion_matrix[get_class(expected_output[i])
                         ][get_class(predicted_output[i])] += 1

    print(confusion_matrix)

    return confusion_matrix
