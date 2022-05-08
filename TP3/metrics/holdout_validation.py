import numpy as np
from ..neural_network import NeuralNetwork


def holdout_eval(input_dataset, expected_output, model: NeuralNetwork, training_ratio=0.8, get_epoch_metrics_fn=None, training_parameters=None, verbose=False):
    shuffled_input = np.array(input_dataset)
    np.random.shuffle(shuffled_input)

    training_set = shuffled_input[:int(len(shuffled_input) * training_ratio)]
    expected_output_training_set = expected_output[:int(
        len(shuffled_input) * training_ratio)]
    test_set = shuffled_input[int(len(shuffled_input) * training_ratio):]
    expected_output_test_set = expected_output[int(
        len(shuffled_input) * training_ratio):]

    model.train(training_set, expected_output, get_epoch_metrics_fn=get_epoch_metrics_fn,
                **training_parameters, verbose=verbose)
    return {'test_set': [test_set, expected_output_test_set], 'training_set': [training_set, expected_output_training_set]}
