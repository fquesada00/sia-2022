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

    fig = plt.figure(10, figsize=(10, 7))
    plt.tight_layout()
    ax = plt.axes(xlim=(-1, k), ylim=(-1, k))
    texts = [['' for i in range(k)] for j in range(k)]
    count_matrix = np.zeros((k, k))
    for i in range(len(winners_sequence)):
        x = winners_sequence[i][0]
        y = winners_sequence[i][1]
        count_matrix[x][y] += 1
        texts[x][y] += countries[i] + '\n'

    # # Trim end of lines at end of text
    for i in range(k):
        for j in range(k):
            texts[i][j] = texts[i][j][:-1]

    # Plot kohonen map
    # ax.set_title('Kohonen Map')

    # Plot neurons
    # ax.imshow(count_matrix, cmap='jet', interpolation='nearest')

    # Plot colorbar
    # cbar = ax.figure.colorbar(ax.images[0], ax=ax)
    # cbar.ax.set_ylabel('Winners', rotation=-90, va="bottom")
    # Limit color bar values
    # ax.grid(False)
    ax.axis('off')

    for i in range(k):
        for j in range(k):
            plt.annotate(texts[i][j], (j, i),
                         ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5))

    plt.savefig('demo.png', transparent=True)
    plt.show()


def plot_heatmap(heatmap, i, title, k):
    plt.subplot(2, 2, i)
    texts = [['' for i in range(k)] for j in range(k)]
    plt.axis('off')
    plt.title(title, fontsize=15)
    plt.tight_layout()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.imshow(np.flip(heatmap, axis=0), cmap='jet', interpolation='nearest')
    # Plot colorbar
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Variable Weights', rotation=-90, va="bottom")
    # Limit color bar values
    for i in range(k):
        for j in range(k):
            plt.annotate(texts[i][j], (j, i),
                         ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5))


def plot_u_matrix_anim(u_matrix_train):
    u_matrix_train = np.array(u_matrix_train)
    plt.figure(100)
    flatten_matrixes = []
    for i in range(len(u_matrix_train)):
        flatten_matrixes.append(u_matrix_train[i].flatten())
    plt.plot(np.mean(flatten_matrixes, axis=1))

    fig = plt.figure(3)
    txt = plt.title('U matrix ({0})'.format(0))
    plt.tight_layout()

    ax = fig.add_subplot(111)
    # Plot neurons
    im = ax.imshow(u_matrix_train[0], cmap='gray', interpolation='nearest')
    # Plot colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Mean distance with neighbors',
                       rotation=-90, va="bottom")

    def animate(i):
        # Add country name with font size of 20
        print(np.mean(u_matrix_train[i]))
        cbar_min, cbar_max = 0, u_matrix_train[i].max()
        im.set_data(u_matrix_train[i])
        im.set_clim(cbar_min, cbar_max)
        txt.set_text('U matrix ({0})'.format(i))
        # return im,

    anim = animation.FuncAnimation(fig, animate,
                                   frames=len(u_matrix_train), repeat=False, interval=25)
    plt.show()


def plot_u_matrix(u_matrix):

    plt.figure(2)
    plt.title('U matrix')
    plt.tight_layout()

    plt.imshow(u_matrix, cmap='gray', interpolation='nearest',
               clim=(0, u_matrix.max()))

    for i in range(len(u_matrix)):
        for j in range(len(u_matrix)):
            plt.text(j, i, '{0:.2f}'.format(u_matrix[i][j]),
                     ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5))
    # Plot colorbar
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Mean distance with neighbors',
                       rotation=-90, va="bottom")
    # Limit color bar values


def main():
    np.random.seed(13)
    data = read_dataset('./datasets/europe.csv')

    # remove Country column from dataset
    data_no_countries = data.drop(['Country'], axis=1)

    # scale dataset
    data_scaled = (data_no_countries - data_no_countries.mean()
                   ) / data_no_countries.std()

    data_scaled_numpy = data_scaled.to_numpy()

    k = 5

    kohonen = Kohonen(k, data_scaled_numpy, k, 0.90)

    winner_idx_arr_row, winner_idx_arr_col, radius_arr, learning_rate_arr, dist_arr, u_matrix_train = kohonen.train(
        data_scaled_numpy, 700)

    winners_sequence, winners = kohonen.test(data_scaled_numpy)

    u_matrix = kohonen.get_u_matrix()
    plot_u_matrix(u_matrix)
    plt.show()
    # plot_u_matrix_anim(u_matrix_train)

    for i in range(len(data_scaled_numpy[0])):
        if i > 3:
            break
        heatmap = kohonen.get_mean_column_weight(i)
        plot_heatmap(heatmap, (i+1) % 5, data.columns[i+1], k)

    plot_map(winners_sequence, winners, data['Country'], k)
    # plt.show()


if __name__ == '__main__':
    main()
