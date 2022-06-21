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


original_dim = 28*28
batch_size = original_dim

latent_dim = 2
intermediate_dim = 450
epochs = 200
epsilon_std = 1.0


def sampling(args: tuple):
    # we grab the variables from the tuple
    z_mean, z_log_var = args
    print(z_mean)
    print(z_log_var)
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon  # h(z)


# input to our encoder
x = Input(shape=(original_dim,), name="input")
# intermediate layer
h = Dense(intermediate_dim, activation='relu', name="encoding")(x)
h = Dense(275, activation='relu', name="encoding-3")(h)
h = Dense(128, activation='relu', name="encoding-4")(h)
# defining the mean of the latent space
z_mean = Dense(latent_dim, name="mean")(h)
# defining the log variance of the latent space
z_log_var = Dense(latent_dim, name="log-variance")(h)
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
# defining the encoder as a keras model
encoder = Model(x, [z_mean, z_log_var, z], name="encoder")
# print out summary of what we just did
encoder.summary()

# Input to the decoder
input_decoder = Input(shape=(latent_dim,), name="decoder_input")
# taking the latent space to intermediate dimension
decoder_h = Dense(128, activation='relu',
                  name="decoder_h-3")(input_decoder)
decoder_h = Dense(275, activation='relu',
                  name="decoder_h-2")(decoder_h)
decoder_h = Dense(intermediate_dim, activation='relu',
                  name="decoder_h")(decoder_h)
# getting the mean from the original dimension
x_decoded = Dense(original_dim, activation='sigmoid',
                  name="flat_decoded")(decoder_h)
# defining the decoder as a keras model
decoder = Model(input_decoder, x_decoded, name="decoder")
decoder.summary()

# grab the output. Recall, that we need to grab the 3rd element our sampling z
output_combined = decoder(encoder(x)[2])
# link the input and the overall output
vae = Model(x, output_combined)
# print out what the overall model looks like
vae.summary()


def vae_loss(x: tf.Tensor, x_decoded_mean: tf.Tensor):
    # Aca se computa la cross entropy entre los "labels" x que son los valores 0/1 de los pixeles, y lo que salió al final del Decoder.
    xent_loss = original_dim * \
        metrics.binary_crossentropy(x, x_decoded_mean)  # x-^X
    kl_loss = - 0.5 * K.sum(1 + z_log_var -
                            K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)
    return vae_loss


vae.compile(loss=vae_loss)
vae.summary()

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

history = vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size)

x_test_encoded = encoder.predict(x_test, batch_size=batch_size)[0]
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0],
            x_test_encoded[:, 1], c=y_test, cmap='viridis')
plt.colorbar()
plt.show()

with open('vae_fashion_mnist_history.txt', 'a') as f:
    f.write(f"{original_dim}-{intermediate_dim}-{375}-{200}-{75}-{latent_dim}-{75}-{200}-{375}-{intermediate_dim}-{original_dim}\n")
    for loss in history.history["loss"]:
        f.write(f"{loss};")
    f.write("\n")

# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train'], loc='upper right')
# plt.show()

def plot_decoded_latent_space():
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

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

# n = 15  # figure with 15x15 digits
# digit_size = 28
# figure = np.zeros((digit_size * n, digit_size * n))
# # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# # to produce values of the latent variables z, since the prior of the latent space is Gaussian
# grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
# grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

# for i, yi in enumerate(grid_x):
#     for j, xi in enumerate(grid_y):
#         z_sample = np.array([[xi, yi]])
#         x_decoded = decoder.predict(z_sample)
#         digit = x_decoded[0].reshape(digit_size, digit_size)
#         figure[i * digit_size: (i + 1) * digit_size,
#                j * digit_size: (j + 1) * digit_size] = digit

# plt.figure(figsize=(10, 10))
# plt.imshow(figure, cmap='Greys_r')
# plt.show()