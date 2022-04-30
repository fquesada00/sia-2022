import numpy as np
from ..neural_network import NeuralNetwork

def holdout_eval(input_dataset, expected_output, model: NeuralNetwork, activation_function='identity', training_ratio=0.8):
    shuffled_input = np.array(input_dataset)
    np.random.shuffle(shuffled_input)

    training_set = shuffled_input[:int(len(shuffled_input) * training_ratio)]
    test_set = shuffled_input[int(len(shuffled_input) * training_ratio):]

    model.train(training_set, expected_output, epochs=100, tol=1e-8)
    return model.test(test_set, expected_output), {'test_set': test_set, 'training_set': training_set}