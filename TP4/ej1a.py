from pprint import pprint
import numpy as np

from models.Kohonen import Kohonen
from utils.read_dataset import read_dataset

import matplotlib.pyplot as plt
from matplotlib import animation


def generate_shifted_dots(winners):
    visited = {}
    x = np.array([])
    y = np.array([])
    for index in range(len(winners)):
        winner = winners[index]
        shift = visited.get(f'{winner[0]},{winner[1]}', 0)

        x = np.append(x, winner[0]-shift)
        y = np.append(y, winner[1] + shift)

        visited[f'{winner[0]},{winner[1]}'] = shift - 0.05

    return np.hstack((x[:, np.newaxis], y[:, np.newaxis]))


def plot_map(winners_sequence, winners, countries, k):

    # fig, ax = plt.subplots(figsize=(10, 10))
    fig = plt.figure(1, figsize=(10, 7))
    plt.tight_layout()
    data = generate_shifted_dots(winners)
    ax = plt.axes(xlim=(-1, k), ylim=(-1, k))

    scatter = ax.scatter([], [], s=500,  color='red', edgecolors='black')
    annotations = []
    for i in range(len(data)):
        annotation = ax.annotate(
            '', (data[i][0], data[i][1]), fontsize=10, color='black')
        annotations.append(annotation)

    # Plot kohonen map
    ax.set_title('Kohonen Map')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Plot neurons
    im = ax.imshow(np.zeros((k, k)), cmap='jet',
                   clim=(0, k), interpolation='none')
    # Plot colorbar
    cbar = ax.figure.colorbar(ax.images[0], ax=ax)
    cbar.ax.set_ylabel('Winners', rotation=-90, va="bottom")
    # Limit color bar values
    ax.grid(False)

    def init():
        for annotation in annotations:
            annotation.set_text('')
        im.set_data(np.zeros((k, k)))
        return im, scatter, *annotations,

    def animate(i):
        # Add country name with font size of 20
        im_data = im.get_array()
        im_data[winners_sequence[i][1]][winners_sequence[i][0]] += 1
        im.set_data(im_data)
        annotations[i].set_text(countries[i])
        scatter.set_offsets(data[:(i+1), :])
        return im, scatter, *annotations

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=28, interval=1000, blit=True, repeat=False)

    # anim.save('kohonen_map.mp4', fps=2, dpi=300)
    plt.show()


def main():
    data = read_dataset('./datasets/europe.csv')

    # remove Country column from dataset
    data_no_countries = data.drop(['Country'], axis=1)

    # scale dataset
    data_scaled = (data_no_countries - data_no_countries.mean()
                   ) / data_no_countries.std()
    data_scaled_numpy = data_scaled.to_numpy()

    k = 7

    kohonen = Kohonen(k, data_scaled_numpy, k, 0.01)
    winner_idx_arr_row, winner_idx_arr_col, radius_arr, learning_rate_arr, dist_arr = kohonen.train(
        data_scaled_numpy, 100)

    winners_sequence, winners = kohonen.test(data_scaled_numpy)

    plot_map(winners_sequence, winners, data['Country'], k)


if __name__ == '__main__':
    main()
