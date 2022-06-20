import time
from matplotlib import pyplot as plt
import numpy as np
from models.Autoencoder import Autoencoder, create_autoencoder
from models.Model import Model
from utils import to_bin_array, to_raw_dataset
from utils.dataset import font_1, font_2, font_3
import warnings
warnings.filterwarnings('ignore')

monocromatic_cmap = plt.get_cmap('binary')


def plot_denoiser(test_set: np.ndarray, denoised_output: np.ndarray, labelled_dataset: list[dict], title: str, filename: str):
    fig, axs = plt.subplots(
        len(test_set), 3, sharey=False, figsize=(2, 10), facecolor='white')

    original_dataset = np.array([data.flatten() for data in map(
        to_bin_array, to_raw_dataset(labelled_dataset))])

    for i in range(0, len(test_set)):
        axs[i][0].imshow(test_set[i].reshape(7, 5), cmap=monocromatic_cmap)
        axs[i][1].imshow(original_dataset[i].reshape(
            7, 5), cmap=monocromatic_cmap)
        axs[i][2].imshow(denoised_output[i].reshape(
            7, 5), cmap=monocromatic_cmap)
        axs[i][0].axis('off')
        axs[i][1].axis('off')
        axs[i][2].axis('off')

    fig.suptitle(title)
    fig.savefig(filename, dpi=300)


def add_noise(image: np.ndarray, mode: str, amount: float = 0.1, seed: float = 1) -> np.ndarray:
    """
     Add noise to an image.

     Parameters
     ----------
     image : ndarray
         Input image data. Will be converted to float.
     mode : str
         One of the following strings, selecting the type of noise to add:

         'gauss'     Gaussian-distributed additive noise.

         's&p'   Salt and pepper noise.
     amount : float
            Amount of noise to add. Default is 0.1. Must be a value between 0 and 1.

     """
    np.random.seed(seed)

    if mode == "gauss":
        mean = 0
        sigma = 1
        gauss = np.random.normal(mean, sigma, image.shape) * amount
        # gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss

    elif mode == "s&p":
        noisy = np.copy(image)
        s_vs_p = 0.5
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        noisy[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        noisy[coords] = 0

    # ensure values are between 0 and 1
    return np.clip(noisy, 0., 1.)


def denoising_autoencoder(dataset: list[list[int]], encoder_layers: list[int], decoder_layers: list[int],
                          encoder_activations: list[str], decoder_activations: list[str], optimizer: str,
                          epochs: int, noise_type: str, noise_amount: float, noise_samples: int = 1, error_filename: str = None, seed: float = 1) -> Model:
    """
    Create and train a denoising autoencoder.

    Parameters
    ----------
    dataset : list[list[int]]
        Dataset to train the autoencoder on.
    encoder_layers : list[int]
        List of hidden layers sizes for the encoder.
    decoder_layers : list[int]
        List of hidden layers sizes for the decoder.
    encoder_activations : list[str]
        List of activations for the encoder.
    decoder_activations : list[str]
        List of activations for the decoder.
    optimizer : str
        Optimizer to use for training.
    epochs : int
        Number of epochs to train the model for.
    noise_type : str
        Type of noise to add to the dataset.
    noise_amount : float
        Amount of noise to add to the dataset.
    """

    # Create autoencoder
    input_size = len(dataset[0])
    np.random.seed(seed)
    encoder, decoder = create_autoencoder(
        input_size, encoder_layers, decoder_layers, encoder_activations, decoder_activations)
    autoencoder = Autoencoder(encoder, decoder, optimizer)

    # Create noisy dataset
    noisy_dataset = np.array([add_noise(np.copy(bitmap), noise_type, noise_amount, seed=i)
                              for i, bitmap in enumerate(np.repeat(dataset, noise_samples, axis=0))])
    target_dataset = np.repeat(dataset, noise_samples, axis=0)

    # Train autoencoder
    autoencoder.fit(noisy_dataset, target_dataset,
                    epochs, error_filename=error_filename)

    return autoencoder


def get_architecture_label(architecture) -> str:
    return ' '.join(
        [str(size) for size in [35, *architecture['encoder_layers'], *architecture['decoder_layers']]])


def run_architectures(dataset: np.ndarray, architectures: list[dict[list[int], list[int], list[str], list[str]]], optimizer: str, epochs: int, noise_type: str, noise_amount: float, noise_samples: int):

    for i, architecture in enumerate(architectures):
        encoder_layers = architecture["encoder_layers"]
        decoder_layers = architecture["decoder_layers"]
        encoder_activations = architecture["encoder_activations"]
        decoder_activations = architecture["decoder_activations"]

        start = time.time()
        dae = denoising_autoencoder(dataset, encoder_layers, decoder_layers, encoder_activations, decoder_activations,
                                    optimizer, epochs, noise_type, noise_amount, noise_samples, error_filename=f"denoiser_error_{i+1}.txt", seed=1)
        end = time.time()
        print("Training time: ", end - start)

        test_set = [add_noise(char.copy(), "gauss", 0.5)
                    for char in raw_dataset[:10]]

        predictions = [dae(noisy_char)
                       for noisy_char in test_set]
        # Plot denoised output
        plot_denoiser(test_set, predictions, font_2,
                      get_architecture_label(architecture),
                      f"denoiser_output_{i+1}.png")


def plot_architecture_errors(error_filename_prefix: str, architectures: list[dict[list[int], list[int], list[str], list[str]]]):
    error_filenames = [
        f"{error_filename_prefix}{i+1}.txt" for i in range(len(architectures))]

    errors = [np.loadtxt(filename) for filename in error_filenames]
    epochs = np.arange(len(errors[0]))

    for i, error in enumerate(errors):
        plt.plot(epochs, errors[i],
                 label=get_architecture_label(architectures[i]))

    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Error", fontsize=15)
    plt.title("Error vs Epochs for different architectures", fontsize=20)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    labelled_dataset = font_2

    # Set parameters
    raw_dataset = np.array([data.flatten() for data in map(
        to_bin_array, to_raw_dataset(labelled_dataset))])

    input_size = len(raw_dataset[0])
    latent_space_size = 45
    encoder_layers = [40, latent_space_size]
    decoder_layers = [40, input_size]
    encoder_activations = ["relu", "relu", "logistic"]
    decoder_activations = ["relu", "relu", "logistic"]
    architectures = [
        {
            "encoder_layers": [40, 45],
            "decoder_layers": [40, 35],
            "encoder_activations": ["relu", "logistic"],
            "decoder_activations": ["relu", "logistic"],
        },
        {
            "encoder_layers": [40],
            "decoder_layers": [35],
            "encoder_activations": ["logistic"],
            "decoder_activations": ["logistic"],
        },
        {
            "encoder_layers": [20],
            "decoder_layers": [35],
            "encoder_activations": ["logistic"],
            "decoder_activations": ["logistic"],
        },
        {
            "encoder_layers": [20, 10],
            "decoder_layers": [20, 35],
            "encoder_activations": ["relu", "logistic"],
            "decoder_activations": ["relu", "logistic"],
        },
        {
            "encoder_layers": [15, 2],
            "decoder_layers": [15, 35],
            "encoder_activations": ["relu", "logistic"],
            "decoder_activations": ["relu", "logistic"],
        },
    ]

    run_architectures(raw_dataset, architectures=architectures, optimizer="powell", epochs=10,
                      noise_type="gauss", noise_amount=0.5, noise_samples=5)

    plot_architecture_errors("denoiser_error_", architectures)

    # # Create and train autoencoder
    # start = time.time()
    # dae = denoising_autoencoder(raw_dataset, encoder_layers, decoder_layers, encoder_activations,
    #                             decoder_activations, optimizer='powell', epochs=10, noise_type="gauss", noise_amount=0.5, noise_samples=5, error_filename="denoiser_error.txt")
    # end = time.time()
    # print("Training time: ", end - start)

    # # Test autoencoder
    # test_set = [add_noise(char.copy(), "gauss", 0.5)
    #             for char in raw_dataset]

    # predictions = [dae(noisy_char)
    #                for noisy_char in test_set]

    # plot_denoiser(test_set, predictions, labelled_dataset)
