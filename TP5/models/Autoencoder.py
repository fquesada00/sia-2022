from pprint import pprint
import numpy as np
from torch import nn

class Autoencoder():

    def __init__(self, input_size:int, hidden_layers_size:list[int], latent_space_size:int,
                 activation_function:str='sigmoid', optimizer:str='adam',
                 learning_rate:float=0.001):
        self.input_size = input_size
        self.hidden_layers_size = hidden_layers_size
        self.latent_space_size = latent_space_size
        self.activation_function = activation_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        weights = []
        # input -> hidden -> latent -> hidden -> output
        # build the encoder weights
        below_layer_size = input_size
        for i in range(len(hidden_layers_size)):
            weights.append(np.random.uniform(low=-1, high=1, size=(hidden_layers_size[i], below_layer_size)))
            below_layer_size = hidden_layers_size[i]
        weights.append(np.random.uniform(low=-1, high=1, size=(latent_space_size, below_layer_size)))
        below_layer_size = latent_space_size
        # build the decoder weights
        for i in range(len(hidden_layers_size)):
            weights.append(np.random.uniform(low=-1, high=1, size=(below_layer_size, hidden_layers_size[len(hidden_layers_size)-i-1])))
            below_layer_size = hidden_layers_size[len(hidden_layers_size)-i-1]
        weights.append(np.random.uniform(low=-1, high=1, size=(input_size, below_layer_size)))
        
        # self.weights = nn.ParameterList([nn.Parameter(torch.from_numpy(w).float()) for w in weights])
        # initialize optimizer
        # if optimizer == 'adam':
            # self.optimizer = torch.optim.Adam(self.weights, lr=learning_rate, objective=None)
        # elif optimizer == 'sgd':
            # self.optimizer = torch.optim.SGD(self.weights, lr=learning_rate)
        # else:
            # raise ValueError('Invalid optimizer')
        

    def predict(self,input_data:np.ndarray[float]):
        # weights is a list of matrixes, each matrix represents a layer's weights
        output = input_data
        for layer_weights in self.weights:
            output = self.activation_function(np.dot(layer_weights.numpy(), output.T))
        return output

    def train(self, dataset: np.ndarray, epochs:int=10):
        for epoch in range(epochs):
           self.weights(dataset)

