import argparse
import numpy as np

from ..metrics.confusion_matrix import generate_confusion_matrix
from ..metrics import print_metrics
from ..neural_network import NeuralNetwork
from ..metrics.holdout_validation import holdout_eval
from ..metrics.k_fold_cross_validation import k_fold_cross_validation_eval

def get_activation_function(activation_function):
    if activation_function == "non-linear":
        return "tanh"
    if activation_function == "linear":
        return "identity"

def parse_dataset(dataset_file_name):
    dataset = np.array([[]])
    
    with open(dataset_file_name, 'r') as dataset_file:
        for line in dataset_file:
            data = np.array([float(i) for i in line.split()]) 
            dataset = np.append(dataset, data).reshape(-1,len(data))

    return dataset


if __name__ == "__main__":
        # numbers from 1 to 100
    
    for i in range(1,101):
        print(i / 100)

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--function", type=str,
                        default="linear", help="Activation function to use with perceptron.\n Value may be 'linear' or 'non-linear'", dest="activation_function", required=False)
    parser.add_argument("--split_method", help="Method to split input data into train and test set.\n Use 'all' to use the entire dataset for training, 'holdout' for holdout validation, 'k-fold' for k-fold cross validation.", 
        dest='training_method', required=False, default='all')
    parser.add_argument("--k", help="Number of folds to use for k-fold cross validation.", dest='k', required=False, default=5)
    parser.add_argument("--ratio", help="Ratio of training data to use for holdout validation.", dest='ratio', required=False, default=0.8)
   
    args = parser.parse_args()

    input_dataset = parse_dataset('./TP3/ej2/dataset/input.txt')
    expected_output = parse_dataset('./TP3/ej2/dataset/expected_output.txt')
    
    activation_function = get_activation_function(args.activation_function)
    training_method = args.training_method
    k = int(args.k)
    training_ratio = float(args.ratio)

    neural_network = NeuralNetwork(hidden_sizes=[], input_size=3, output_size=1, learning_rate=0.5,
                                   bias=4.5, activation_function_str=activation_function, batch_size=1, beta=0.05)
    if training_method == 'all':
        test_set_input = input_dataset
        test_set_expected_output = expected_output
        
        neural_network.train(input_dataset, expected_output, epochs=50)
        error, predictions = neural_network.test(test_set_input,test_set_expected_output)

    elif training_method == 'holdout':
        error, holdout_sets = holdout_eval(input_dataset, expected_output, neural_network, activation_function=activation_function, training_ratio=training_ratio)
       
        test_set_input = holdout_sets['test_set'][0]
        test_set_expected_output = holdout_sets['test_set'][1]
        
        error, predictions = neural_network.test(test_set_input, test_set_expected_output)

    elif training_method == 'k-fold':
        average_error, min_error_sets = k_fold_cross_validation_eval(input_dataset, expected_output, neural_network, k=k, activation_function=activation_function)
        
        test_set_input = min_error_sets['test_set'][0]
        test_set_expected_output = min_error_sets['test_set'][1]
        
        error, predictions = neural_network.test(test_set_input, test_set_expected_output)

    else:
        print("Invalid training method")
        exit(1)

    print(f"Error: {error}")
    print(f"Predictions: {predictions}")
    print(f"Expected output: {test_set_expected_output}")
    # [0 0.3 0.5 1]
    classes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    confusion_matrix = generate_confusion_matrix(test_set_expected_output, predictions, classes)

    print_metrics(confusion_matrix)
    




