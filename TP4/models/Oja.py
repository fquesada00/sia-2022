from cProfile import label
from matplotlib import animation, pyplot as plt
import numpy as np


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def plot(input_dataset: np.ndarray, weights_evolution: np.ndarray, iterations: np.ndarray, real_pc1: np.ndarray, headers: list[str], sample_labels: list[str], correct_direction: bool = True, animation_step: int = 100):
    # correct approximation sign in case it was computed in opposite direction
    if not np.array_equiv(np.sign(weights_evolution[-1]), np.sign(real_pc1)) and correct_direction:
        weights_evolution = [-weights_evolution[i]
                             for i in range(len(weights_evolution))]

    # generate data for plots

    plot_loadings_animation(weights_evolution, real_pc1,
                            headers, sample_labels, animation_step)

    plot_pc1_projection_animation(weights_evolution, input_dataset, real_pc1,
                                  headers, sample_labels, animation_step)

    plot_angle_error_evolution(
        weights_evolution, iterations, real_pc1, animation_step)

    # plots

    # angle and error between real PCA and approx PCA vs iteration

    biplot_evolution_fig = plt.figure(5)

    # anim.save('loadings_bar_evolution.gif', writer='imagemagick', fps=10

    plt.show()


def plot_pc1_projection_animation(weights_evolution: np.ndarray, input_dataset: np.ndarray, real_pc1: np.ndarray, headers: list[str], sample_labels: list[str], animation_step: int = 100):
    proj_Y_axis = np.arange(len(input_dataset))

    proj_bar_evolution = [input_dataset.dot(loadings)
                          for loadings in weights_evolution]

    proj_bar_evolution = proj_bar_evolution[0::animation_step]

    proj_barh_evolution_fig, proj_barh_evolution_ax = plt.subplots(
        1, 1)

    proj_barh_evolution_ax.legend()

    def animate(i):
        # i = i * 100
        proj_barh_evolution_ax.clear()

        proj_barh_evolution_ax.barh(
            proj_Y_axis - 0.2, proj_bar_evolution[i], 0.4, color="blue", label="Aproximación")

        proj_barh_evolution_ax.barh(proj_Y_axis + 0.2, input_dataset.dot(
            real_pc1), 0.4, color="red", label="Librería")

        proj_barh_evolution_ax.legend()
        proj_barh_evolution_ax.set_xlim([-7, 7])

        proj_barh_evolution_ax.set_title(
            "PC1 aproximada vs librería\n Iteración: {}".format(i * animation_step))
        proj_barh_evolution_ax.set_xlabel("PC1")
        proj_barh_evolution_ax.set_ylabel("País")

        proj_barh_evolution_ax.set_yticks(range(len(sample_labels)))
        proj_barh_evolution_ax.set_yticklabels(sample_labels)

    anim = animation.FuncAnimation(
        proj_barh_evolution_fig, animate, frames=len(proj_bar_evolution), interval=1)

    plt.show()


def plot_loadings_animation(weights_evolution: np.ndarray, real_pc1: np.ndarray, headers: list[str], sample_labels: list[str], animation_step: int = 100):
    loadings_X_axis = np.arange(len(weights_evolution[0]))

    loadings_bar_evolution_fig, loadings_bar_evolution_ax = plt.subplots(
        1, 1)

    weights_evolution = weights_evolution[0::animation_step]

    def animate(i):
        # i = i * 100
        loadings_bar_evolution_ax.clear()

        loadings_bar_evolution_ax.bar(
            loadings_X_axis - 0.2, weights_evolution[i], 0.4, color="blue", label="Aproximación")

        loadings_bar_evolution_ax.bar(
            loadings_X_axis + 0.2, real_pc1, 0.4, color="red", label="Librería")

        loadings_bar_evolution_ax.legend()
        loadings_bar_evolution_ax.set_ylim([-1, 1])

        loadings_bar_evolution_ax.set_title(
            "Cargas de la PC1 aproximada\n Iteración: {}".format(i * animation_step))
        loadings_bar_evolution_ax.set_ylabel("Carga")
        # set x ticks
        loadings_bar_evolution_ax.set_xticks(range(len(headers)))
        loadings_bar_evolution_ax.set_xticklabels(headers)

    anim = animation.FuncAnimation(
        loadings_bar_evolution_fig, animate, frames=len(weights_evolution), interval=1)

    plt.show()


def plot_angle_error_evolution(weights_evolution: np.ndarray, iterations: np.ndarray, real_pc1: np.ndarray, animation_step: int = 100):
    errors = [np.linalg.norm(loadings - real_pc1)
              for loadings in weights_evolution]
    angles = [angle_between(loadings, real_pc1)
              for loadings in weights_evolution]

    angle_error_evolution_fig, angle_error_evolution_axs = plt.subplots(
        2, 1)

    angle_error_evolution_axs[0].set_title(
        "Ángulo entre autovector real y aproximado")
    angle_error_evolution_axs[0].set_xlabel("Iteración")
    angle_error_evolution_axs[0].set_ylabel("Ángulo (radianes)")
    angle_error_evolution_axs[1].set_title(
        "Distancia euclidiana entre autovector real y aproximado")
    angle_error_evolution_axs[1].set_xlabel("Iteración")
    angle_error_evolution_axs[1].set_ylabel("Distancia euclidiana")
    angle_error_evolution_axs[0].set_yscale('log')
    angle_error_evolution_axs[1].set_yscale('log')

    angle_error_evolution_axs[0].plot(iterations, angles)
    angle_error_evolution_axs[1].plot(iterations, errors)

    plt.show()


class Oja:

    # angle_and_error_vs_learning_rate_fig = plt.figure(2)

    @classmethod
    def compute_pc1(self, input_dataset: np.ndarray, epochs: int, learning_rate: float, generate_plots: bool = True, real_pc1: np.ndarray = None, headers: list[str] = [], sample_labels: list[str] = []) -> np.ndarray:
        """
        Approximate the first principal component of the input dataset.

        :param input_dataset: The input dataset.
        :param epochs: The number of epochs.
        :param learning_rate: The learning rate.
        :param generate_plots: Whether to generate plots.
        :param real_pc1: The real first principal component used for plots.
        :param headers: The headers of the dataset.
        :param sample_labels: The labels of the samples.
        :return: The first principal component approximation.
        """

        # Initialize the weights with random values and norm 1.
        weights = np.random.uniform(
            size=input_dataset.shape[1], low=-1, high=1)
        weights /= np.linalg.norm(weights)

        # for plots
        if generate_plots:
            weights_evolution = [weights]
            iterations = [0]

        # For each epoch.
        for epoch in range(epochs):
            # For each sample.
            for i, sample in enumerate(input_dataset):
                # Compute the projection of the sample on the weights.
                proj = np.dot(sample, weights)
                # Update the weights.
                weights += learning_rate * proj * (sample - proj * weights)

                # for plots
                if generate_plots:
                    iterations.append(epoch * input_dataset.shape[0] + i)
                    weights_evolution.append(weights.copy())

        # for plots
        if generate_plots:
            plot(input_dataset, weights_evolution,
                 iterations, real_pc1, headers, sample_labels)

        return weights
