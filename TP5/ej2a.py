from utils import plot_decoded_latent_space_v1
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from keras.datasets import fashion_mnist
from keras import metrics
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#from tensorflow import keras
# hack tf-keras to appear as top level keras
#import sys
#sys.modules['keras'] = keras
# end of hack


disable_eager_execution()

batch_size = 100

original_dim = 28*28

latent_dim = 2
latent_activation = "sigmoid"
intermediate_dims = [256, 180, 128, 64]
activations=["relu", "relu", "relu", "relu"]
epochs = 5


# input to our encoder
x = Input(shape=(original_dim,), name="input")
# intermediate layer
def create_hidden_encoding_layers(input_layer, intermediate_dims, activations):
    prev_layer = input_layer
    layers = []
    for i in range(len(intermediate_dims)):
        intermediate_layer = Dense(intermediate_dims[i], activation=activations[i], name=f"encoding-{i + 1}")(prev_layer)
        prev_layer = intermediate_layer
        layers.append(intermediate_layer)
    return layers

last_hidden_encoding_layer = create_hidden_encoding_layers(x, intermediate_dims, activations)[-1]

latent_space_layer = Dense(latent_dim, activation=latent_activation, name="latent-space")(last_hidden_encoding_layer)
# defining the encoder as a keras model
encoder = Model(x, latent_space_layer, name="encoder")
# print out summary of what we just did
encoder.summary()

# Input to the decoder
input_decoder = Input(shape=(latent_dim,), name="decoder_input")
def create_hidden_decoding_layers(input_layer, intermediate_dims, activations):
    print(intermediate_dims)
    prev_layer = input_layer
    layers = []
    for i in range(len(intermediate_dims)):
        intermediate_layer = Dense(intermediate_dims[i], activation=activations[i], name=f"decoding-{i + 1}")(prev_layer)
        prev_layer = intermediate_layer
        layers.append(intermediate_layer)
    return layers

last_hidden_decoding_layer = create_hidden_decoding_layers(input_decoder, intermediate_dims[::-1], activations[::-1])[-1]
decoder_output_layer = Dense(original_dim, activation="sigmoid", name="decoder_output")(last_hidden_decoding_layer)
# defining the decoder as a keras model
decoder = Model(input_decoder, decoder_output_layer, name="decoder")
decoder.summary()

# grab the output. Recall, that we need to grab the 3rd element our sampling z
output_combined = decoder(encoder(x))
# link the input and the overall output
simple_autoencoder = Model(x, output_combined)
# print out what the overall model looks like
simple_autoencoder.summary()


simple_autoencoder.compile(loss='mse',
    metrics=[tf.keras.metrics.MeanAbsoluteError()])
simple_autoencoder.summary()

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

simple_autoencoder.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size)

x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
for x_text in x_test_encoded:
    if x_text[0] < 0 or x_text[1] < 0:
        print(x_text)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0],
            x_test_encoded[:, 1], c=y_test, cmap='viridis')
plt.colorbar()
plt.show()

def plot_decoded_latent_space():
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = np.linspace(0.05, 0.95, n)
    grid_y = np.linspace(0.05, 0.95, n)

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


# plot_decoded_latent_space((28, 28), decoder, norm.ppf, 15)
plot_decoded_latent_space()
