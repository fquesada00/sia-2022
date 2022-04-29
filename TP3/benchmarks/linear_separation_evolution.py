import matplotlib.pyplot as plt
import argparse

import numpy as np
import matplotlib


def plot_linear_separation(input_dataset, expected_output, weights):
    plt.figure(figsize=[10, 5])
    # Load dataset values

    legends = []

    # Having x*w1 + y*w2 - w0 = 0
    # y = (x*w1 + w0) / w2
    for index, weight in enumerate(weights):
        x = np.linspace(-2, 2, 100)
        y = (-x * weight[1] + weight[0]) / weight[2]
        legends.append(f"Iteration {index}")
        plt.plot(x, y)

    for data_number, data in enumerate(input_dataset):
        plt.plot(data[0], data[1], marker='o',
                 color=f'{ "blue" if expected_output[data_number][0] == 1 else "red" }')
    plt.xlim(left=-3, right=3)
    plt.ylim(bottom=-3, top=3)
    plt.legend(legends)
    plt.grid()
    plt.show()


def parse_weights(weights_file_name):
    output_weights_iterations = []
    with open(weights_file_name) as weights_file:

        for line_number, line in enumerate(weights_file):
            if line_number == 0:
                weights_rows, weights_cols = [int(dim) for dim in line.split()]
                continue

            output_weights_iterations.append(list(map(float, line.split())))

    # return [np.ndarray(weights, shape=(weights_rows, weights_cols)) for weights in output_weights_iterations]
    return output_weights_iterations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default=None, help="File path of output weights evolution.", dest="input_file_name", required=True)

    args = parser.parse_args()

    weights = parse_weights(args.input_file_name)
    # Inputs: [ [x0, y0], [x1, y1] ]
    dataset = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
    expected_output = [[1], [1], [1], [-1]]
    plot_linear_separation(dataset, expected_output, weights)
