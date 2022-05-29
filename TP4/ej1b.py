from matplotlib import animation, pyplot as plt
import numpy as np
from models.PCA_SVD import PCA_SVD
from models.Oja import Oja
from utils.read_dataset import read_dataset


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


def generate_plots(input_dataset: np.ndarray, weights_evolution: list[np.ndarray], iterations: list[int], real_pc1: np.ndarray, real_pc2: np.ndarray, headers: list[str], sample_labels: list[str], correct_direction: bool = True, animation_step: int = 100):
    # correct approximation sign in case it was computed in opposite direction
    if not np.array_equiv(np.sign(weights_evolution[-1]), np.sign(real_pc1)) and correct_direction:
        weights_evolution = [-weights_evolution[i]
                             for i in range(len(weights_evolution))]

    plot_loadings_animation(weights_evolution, real_pc1,
                            headers, sample_labels, animation_step)

    plot_pc1_projection_animation(weights_evolution, input_dataset, real_pc1,
                                  headers, sample_labels, animation_step)

    plot_angle_error_evolution(
        weights_evolution, iterations, real_pc1, animation_step)

    plot_biplot_evolution(
        input_dataset, weights_evolution, real_pc2, headers, sample_labels, animation_step)

    # anim.save('loadings_bar_evolution.gif', writer='imagemagick', fps=10

    plt.show()


def plot_pc1_projection_animation(weights_evolution: list[np.ndarray], input_dataset: np.ndarray, real_pc1: np.ndarray, headers: list[str], sample_labels: list[str], animation_step: int = 100):
    proj_Y_axis = np.arange(len(input_dataset))

    proj_bar_evolution = [input_dataset.dot(loadings)
                          for loadings in weights_evolution]

    proj_bar_evolution = proj_bar_evolution[0::animation_step]

    fig, ax = plt.subplots(
        1, 1)

    ax.legend()

    def animate(i):
        # i = i * 100
        ax.clear()

        ax.barh(
            proj_Y_axis - 0.2, proj_bar_evolution[i], 0.4, color="blue", label="Aproximación")

        ax.barh(proj_Y_axis + 0.2, input_dataset.dot(
            real_pc1), 0.4, color="red", label="Librería")

        ax.legend()
        ax.set_xlim([-7, 7])

        ax.set_title(
            "PC1 aproximada vs librería\n Iteración: {}".format(i * animation_step))
        ax.set_xlabel("PC1")
        ax.set_ylabel("País")

        ax.set_yticks(range(len(sample_labels)))
        ax.set_yticklabels(sample_labels)

    anim = animation.FuncAnimation(
        fig, animate, frames=len(proj_bar_evolution), interval=1)

    plt.show()


def plot_loadings_animation(weights_evolution: list[np.ndarray], real_pc1: np.ndarray, headers: list[str], sample_labels: list[str], animation_step: int = 100):
    loadings_X_axis = np.arange(len(weights_evolution[0]))

    fig, ax = plt.subplots(
        1, 1)

    weights_evolution = weights_evolution[0::animation_step]

    def animate(i):
        # i = i * 100
        ax.clear()

        ax.bar(
            loadings_X_axis - 0.2, weights_evolution[i], 0.4, color="blue", label="Aproximación")

        ax.bar(
            loadings_X_axis + 0.2, real_pc1, 0.4, color="red", label="Librería")

        ax.legend()
        ax.set_ylim([-1, 1])

        ax.set_title(
            "Cargas de la PC1 aproximada\n Iteración: {}".format(i * animation_step))
        ax.set_ylabel("Carga")
        # set x ticks
        ax.set_xticks(range(len(headers)))
        ax.set_xticklabels(headers)

    anim = animation.FuncAnimation(
        fig, animate, frames=len(weights_evolution), interval=1)

    plt.show()


def plot_angle_error_evolution(weights_evolution: list[np.ndarray], iterations: list[int], real_pc1: np.ndarray, real_pc2: np.ndarray, animation_step: int = 100):
    errors = [np.linalg.norm(loadings - real_pc1)
              for loadings in weights_evolution]
    angles = [angle_between(loadings, real_pc1)
              for loadings in weights_evolution]

    fig, axs = plt.subplots(
        2, 1)

    axs[0].set_title(
        "Ángulo entre autovector real y aproximado")
    axs[0].set_xlabel("Iteración")
    axs[0].set_ylabel("Ángulo (radianes)")
    axs[1].set_title(
        "Distancia euclidiana entre autovector real y aproximado")
    axs[1].set_xlabel("Iteración")
    axs[1].set_ylabel("Distancia euclidiana")
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')

    axs[0].plot(iterations, angles)
    axs[1].plot(iterations, errors)

    plt.show()


def plot_biplot_evolution(input_dataset: np.ndarray, weights_evolution: list[np.ndarray], real_pc2: np.ndarray, headers: list[str], sample_labels: list[str], animation_step: int = 100):
    fig, ax = plt.subplots(1, 1)

    weights_evolution = weights_evolution[0::animation_step]

    def animate(i):
        # i = i * 100
        ax.clear()
        pc1_approx = weights_evolution[i]

        pc1_approx_proj = input_dataset.dot(pc1_approx)
        # pc1_proj = input_dataset.dot(pc1_real)
        pc2_proj = input_dataset.dot(real_pc2)

        ax.scatter(
            pc1_approx_proj, pc2_proj, color="blue", alpha=0.5)

        ax.legend()
        # ax.set_ylim([-1, 1])

        ax.set_title(
            "PCA Biplot\n Iteración: {}".format(i * animation_step))
        ax.set_ylabel("PC2 (Librería)")
        ax.set_xlabel("PC1 (Aproximación)")

        # annotate the samples
        for i, txt in enumerate(sample_labels):
            plt.annotate(txt, (pc1_approx_proj[
                i], pc2_proj[i]), alpha=0.5)

        # annotate the attributes
        for x, y, col_name in zip(pc1_approx, real_pc2, headers):
            print(x, y)
            plt.arrow(0, 0, x, y, head_width=0.05, color='red')
            plt.annotate(col_name, (x, y))

    anim = animation.FuncAnimation(
        fig, animate, frames=len(weights_evolution), interval=1)

    # pc2_proj.index = data['Country'].to_numpy()

    plt.show()


if __name__ == '__main__':
    data = read_dataset('./datasets/europe.csv')

    # remove Country column from dataset
    data_no_countries = data.drop(['Country'], axis=1)

    # scale dataset
    data_scaled = (data_no_countries - data_no_countries.mean()
                   ) / data_no_countries.std()

    # compute the first principal component with SVD
    components = PCA_SVD.compute_pca(data_scaled.to_numpy())

    # compute the first principal component with Oja's algorithm
    pc1_oja, pc1_evolution, iterations = Oja.compute_pc1(data_scaled.to_numpy(),
                                                         epochs=1200, learning_rate=0.0001, generate_plot_data=True)

    generate_plots(data_scaled.to_numpy(), pc1_evolution, iterations,
                   components[0], components[1], data_no_countries.columns, data['Country'].to_list(), True, 100)

    # compare
    print("With Oja's algorithm:")
    print(pc1_oja)

    print("With SVD:")
    print(components[0])

    # distance
    print("Distance:")
    print(np.linalg.norm(pc1_oja - components[0]))
