import math

# W 3 element vector
# w 2x3 matrix as 6 element vector
# w0 2 element vector
# reactive_values 3 element vector
def formula(W, w, w0, reactive_values):
    inner_g_input = 0
    outer_g_input = 0

    for j in range(2):
        for k in range(3):
            inner_g_input += w[j + (k * 3)] * reactive_values[k] - w0[j]
        outer_g_input += W[j] * g(inner_g_input)
    return g(outer_g_input - W[0])

# dataset 3 element vector
# dataset_expected_value 3 element vector
def error(W, w, w0, dataset, dataset_expected_value):
    error = 0

    for i in range(3):
        error += math.pow(dataset_expected_value[i] - formula(W, w, w0, dataset[i]), 2)

    return error

def g(x):
    # If over the limit, return the limit value
    try:
        exp_value = math.exp(x)
    except OverflowError:
        return 1
        
    return exp_value / (1 + exp_value)