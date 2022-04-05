import math

from ..constants import DATASET_INPUT, DATASET_INPUT, DATASET_OUTPUT


# W 3 element vector
# w 2x3 matrix as 6 element vector
# w_0 2 element vector
# reactive_values 3 element vector
def function_to_approximate(W, w, w_0, reactive_values):
    inner_g_input = 0
    outer_g_input = 0

    for j in range(2):
        for k in range(3):
            inner_g_input += w[k + (j * 3)] * reactive_values[k] - w_0[j]
        outer_g_input += W[j] * g(inner_g_input)
    return g(outer_g_input - W[0])


# dataset_input 3 element vector
# dataset_output 3 element vector
def error(W, w, w_0, dataset_input, dataset_output):
    error = 0

    for i in range(len(dataset_input)):
        error += math.pow(dataset_output[i] -
                          function_to_approximate(W, w, w_0, dataset_input[i]), 2)

    return error


def g(x):
    # If over the limit, return the limit value
    try:
        exp_value = math.exp(x)
    except OverflowError:
        return 1

    return exp_value / (1 + exp_value)


def fitness_function(chromosome):
    return len(DATASET_INPUT) - error(chromosome.W, chromosome.w, chromosome.w_0, DATASET_INPUT, DATASET_OUTPUT)
