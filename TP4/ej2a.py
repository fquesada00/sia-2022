import argparse
from datetime import datetime
import itertools
import random
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.ticker import MaxNLocator

import numpy as np
from models.Hopfield import Hopfield
from datasets.font_1 import alphabet

import matplotlib.pyplot as plt

def add_noise(pattern: list[int], noise_level: float):
    pattern_with_noise = []
    for i in range(len(pattern)):
        if random.random() < noise_level:
            pattern_with_noise.append(pattern[i] * -1)
        else:
            pattern_with_noise.append(pattern[i])

    return pattern_with_noise

def add_noise_to_dataset(letters: list[str], dataset: list[list[list[int]]], noise_level: float, total_noisy_patterns: int, randomize: bool, received_letters: list[str] = None):
    dataset_with_noise = []
    dataset_without_noise = []
    selected_letters = []

    if total_noisy_patterns > len(dataset):
        total_noisy_patterns = len(dataset)

    if randomize:
        indexes = random.sample(range(len(dataset)), total_noisy_patterns)
    else:
        indexes = range(total_noisy_patterns)

    for index in indexes:
        pattern = np.array(dataset[index]).flatten()
        dataset_without_noise.append(pattern)
        noisy_pattern = add_noise(pattern, noise_level)
        dataset_with_noise.append(noisy_pattern)

        if received_letters is not None:
            selected_letters.append(received_letters[index])
        else:
            selected_letters.append(letters[index])

    return selected_letters, dataset_with_noise, dataset_without_noise


def calculate_most_orthogonal_patterns(dataset: dict[str, list[list[int]]], top_patterns: int = 10): 
    # flatten the matrixes
    flat_letters = {
        k: np.array(m).flatten() for (k, m) in dataset.items()
    }

    # generate all possible pattern combinations with 4 letters 
    all_groups = itertools.combinations(flat_letters.keys(), r=4)

    avg_dot_product = []
    max_dot_product = []

    for g in all_groups:
        # get the pattern vectors
        group = np.array([v for k,v in flat_letters.items() if k in g])

        # calculate all dot products
        ortho_matrix = group.dot(group.T)

        # fill diagonal with zeros as we don't want to compare a pattern with itself
        np.fill_diagonal(ortho_matrix, 0)

        row, _ = ortho_matrix.shape
        avg_dot_product.append((np.abs(ortho_matrix).sum()/(ortho_matrix.size-row), g))
        max_dot_product.append((np.abs(ortho_matrix).max(), g))

    best_avg_dot_product = sorted(avg_dot_product, key=lambda x: x[0], reverse=False)

    # for i, p in enumerate(best_avg_dot_product):
    #     p = p[1]
    #     if "O" in p and "C" in p and "T" in p and "A" in p:
    #         print(f"Letters: {best_avg_dot_product[i][1]} - Average dot product: {best_avg_dot_product[i][0]}")
    #         break

    print("=============================================")
    print(f"Top {top_patterns} most orthogonal patterns:")
    with open("ortho.csv", "w") as f:
        f.write("best, average dot product\n")
        for i in range(top_patterns):
            csv_letters = []
            for l in best_avg_dot_product[i][1]:
                csv_letters.append(l)
            csv_letters = ' - '.join(csv_letters)
            f.write(f"{csv_letters}, {best_avg_dot_product[i][0]}\n")
            print(f"Letters: {best_avg_dot_product[i][1]} - Average dot product: {best_avg_dot_product[i][0]}")
        print("=============================================")

        worst_max_dot_product = sorted(max_dot_product, key=lambda x: x[0], reverse=True)
        f.write("least, average dot product\n")
        print(f"Last {top_patterns} least orthogonal patterns:")
        for i in range(top_patterns):
            csv_letters = []
            for l in worst_max_dot_product[i][1]:
                csv_letters.append(l)
            csv_letters = ' - '.join(csv_letters)
            f.write(f"{csv_letters}, {worst_max_dot_product[i][0]}\n")
            print(f"Letters: {worst_max_dot_product[i][1]} - Average dot product: {worst_max_dot_product[i][0]}")
        print("=============================================")
    
def matches(pattern1: list[int], pattern2: list[int]):
    return np.array_equal(pattern1, pattern2)

if __name__ == "__main__":
    dataset = list(alphabet.values())
    letters = list(alphabet.keys())

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-level", "-p", help="Probability to add noise to a given pattern. Defaults to 0.", type=float, default=0.0, required=False)
    parser.add_argument("--iterations", "-N", help="Number of maximum iterations of Hopfield neural network. Defaults to 100.", type=int, default=100, required=False)
    parser.add_argument("--total-noisy-patterns", "-T", help="Number of patterns to add noise to. Defaults to 0.", type=int, default=0, required=False)
    parser.add_argument("--random", "-r", help="Randomize the order of the noise patterns. Defaults to False.", action="store_true", default=False, required=False)
    parser.add_argument("--show-patterns", "-show", help="Show the patterns before and after adding noise. Defaults to False.", action="store_true", default=False, required=False)
    parser.add_argument("--show-animation", "-anim", help="Show the animation of the test patterns. Defaults to False.", action="store_true", default=False, required=False)
    parser.add_argument("--calculate-orthogonal-patterns", "-calc", help="Calculate the most and least orthogonal patterns. Defaults to False.", action="store_true", default=False, required=False)
    parser.add_argument("--predict-patterns", "-pred", help="Predict the patterns. Defaults to ''.", type=str, default="", required=False)
    parser.add_argument("--train-patterns", "-train", help="Train with the patterns. Defaults to ''.", type=str, default="", required=False)

    args = parser.parse_args()

    if args.calculate_orthogonal_patterns:
        calculate_most_orthogonal_patterns(alphabet,10)
        exit(0)


    iterations = args.iterations

    if len(args.train_patterns) > 0:
        inputs = []
        args.train_patterns = args.train_patterns.upper()
        train_patterns = args.train_patterns.split(",")
        for pattern in train_patterns:
            inputs.append(alphabet[pattern])
    else:
        inputs = [alphabet["A"], alphabet["F"], alphabet["I"], alphabet["P"]]

    hopfield = Hopfield(inputs)

    energies_per_pattern = []
    iterations_per_pattern = []
    states_per_pattern = []
    predictions = []

    received_patterns = None
    if len(args.predict_patterns) > 0:
        dataset = []
        args.predict_patterns = args.predict_patterns.upper()
        received_patterns = args.predict_patterns.split(",")
        args.total_noisy_patterns = len(received_patterns)
        for pattern in received_patterns:
            dataset.append(alphabet[pattern])

    selected_letters, dataset_with_noise, dataset_without_noise = add_noise_to_dataset(letters, dataset, args.noise_level, args.total_noisy_patterns, args.random, received_patterns)
    for pattern in dataset_with_noise:
        print(f"Pattern: {pattern}")
        predicted_state, energies, states = hopfield.predict(pattern, iterations)
        energies_per_pattern.append(energies)
        iterations_per_pattern.append(len(energies))
        predictions.append(predicted_state)
        states_per_pattern.append(states)
    
        if args.show_animation or args.show_patterns:
            # plot pixel img
            # show modified dataset
            for pattern, letter, predicted_pattern, states, pattern_without_noise in zip(dataset_with_noise, selected_letters, predictions, states_per_pattern, dataset_without_noise):
                if args.show_animation:
                    anim_fig, ax = plt.subplots()

                    def animate(i):
                        ax.clear()
                        ax.imshow(np.reshape(states[i], (5, 5)), cmap="gray_r")
                        ax.set_title(f"Iteration: {i}")

                    ani = FuncAnimation(anim_fig, animate, frames=len(states), interval=1000, repeat=False)
                    my_date = datetime.now()
                    # FFwriter = FFMpegWriter()
                    # ani.save('plot.mp4', writer=FFwriter)
                    ani.save(f"{letter}_pattern_{my_date.strftime('%Y_%m_%dT%H_%M_%SZ')}.mp4", writer='ffmpeg', fps=1, dpi=600)
                    plt.show()
                    

                if args.show_patterns:
                    fig, (ax_left, ax_right) = plt.subplots(1, 2)
                    ax_left.imshow(np.reshape(pattern, (5, 5)), cmap="gray_r")
                    ax_left.text(0.5, -0.8, f"Input pattern", fontsize=15)
                    ax_right.imshow(np.reshape(predicted_pattern, (5, 5)), cmap="gray_r")
                    ax_right.text(0.3, -0.8, f"Output pattern", fontsize=15)
                    fig.suptitle(f"Matches: {'Yes' if matches(pattern_without_noise, predicted_pattern) else 'No'}", fontsize=16)

            if args.show_animation:
                plt.close("all")
            # else:
            #     plt.close(1)
            

    # plot energies
    labels = []
    fig, ax = plt.subplots()
    with open("energy_vs_iterations.csv", "w") as f:
        max_iterations = max(iterations_per_pattern)
        header = ",".join(["Pattern"] + ["Iteration " + str(i) for i in range(max_iterations)])
        f.write(header + "\n")
        for energy, total_iterations, letter in zip(energies_per_pattern, iterations_per_pattern, selected_letters):
            ax.plot(np.arange(total_iterations), energy, linestyle="--", marker="o")
            labels.append(f"{letter} pattern - {total_iterations - 1} iterations")

            for _ in range(max_iterations - total_iterations):
                energy.append("-")
                
            energies = ','.join(str(round(float(e), 4) if e != "-" else e) for e in energy)
            f.write(f"{letter},{energies}\n")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    ax.legend(labels)
    # ax = plt.figure(args.total_noisy_patterns, figsize=(10, 10)).gca()
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()
    plt.close()