import numpy as np
from ..neural_network import NeuralNetwork
from TP3 import neural_network

def k_fold_cross_validation_eval(input_dataset, expected_output, model_supplier: lambda: NeuralNetwork, k=10, get_epoch_metrics_fn=None, training_parameters: dict={}, verbose=False):
    shuffled_input = np.array(input_dataset)
    np.random.shuffle(shuffled_input)

    # Split the data into k folds
    fold_size = int(len(shuffled_input) / k)

    # Initialize the error
    error = 0
    
    # Iterate through the folds
    min_error = float('inf')
    min_error_sets = {'test_set': [[], []], 'training_set': [[], []]}
    min_neural_network = None

    for i in range(k):
        # Get the test data
        print(f"Fold {i + 1}")

        model = model_supplier()
        test_set = shuffled_input[i * fold_size: (i + 1) * fold_size]
        print(f"Test set size: {len(test_set)}")

        expected_output_test_set = expected_output[i * fold_size: (i + 1) * fold_size]

        # Get the training data
        training_set = np.concatenate((shuffled_input[:i * fold_size], shuffled_input[(i + 1) * fold_size:]))
        print(f"Training set size: {len(training_set)}")
        
        expected_output_training_set = np.concatenate((expected_output[:i * fold_size], expected_output[(i + 1) * fold_size:]))
        model.train(
            training_set, expected_output_training_set, get_epoch_metrics_fn=get_epoch_metrics_fn, **training_parameters, verbose=False)
        
        current_error, predictions =  model.test(test_set, expected_output_test_set)

        print(f"current error ====> {current_error}")

        if current_error < min_error:
            print(f"New min error: {current_error}")
            min_error = current_error
            min_error_sets = {'test_set': [test_set, expected_output_test_set], 'training_set': [training_set, expected_output_training_set]}
            min_neural_network = model

        error += current_error

    return (error / k), min_error_sets, min_neural_network