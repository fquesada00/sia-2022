import numpy as np
from ..neural_network import NeuralNetwork

def k_fold_cross_validation_eval(input_dataset, expected_output, model: NeuralNetwork, k=10, activation_function='identity'):
    shuffled_input = np.array(input_dataset)
    np.random.shuffle(shuffled_input)

    # Split the data into k folds
    fold_size = int(len(shuffled_input) / k)

    # Initialize the error
    error = 0
    
    # Iterate through the folds
    min_error = 1000
    min_error_sets = {}

    for i in range(k):
        # Get the test data
        test_set = shuffled_input[i * fold_size: (i + 1) * fold_size]

        # Get the training data
        training_set = np.concatenate((shuffled_input[:i * fold_size], shuffled_input[(i + 1) * fold_size:]))



        model.train(training_set, expected_output, epochs=100, tol=1e-8)
        current_error =  model.test(test_set, expected_output)

        if current_error < min_error:
            print(f"New min error: {current_error}")
            min_error = current_error
            min_error_sets = {'test_set':test_set,'training_set':training_set}

        error += current_error

    return (error / k), min_error_sets