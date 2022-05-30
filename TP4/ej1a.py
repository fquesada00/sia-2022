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
    plt.show()
    # anim.save('kohonen_map.mp4', fps=2, dpi=300)


def plot_heatmap(heatmap, i, title):
    plt.figure(i)
    plt.title(title)
    plt.tight_layout()

    plt.imshow(heatmap, cmap='jet', interpolation='bicubic')
    # Plot colorbar
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Weights', rotation=-90, va="bottom")
    # Limit color bar values

def plot_u_matrix_anim(u_matrix_train):
    fig = plt.figure(3)
    txt = plt.title('U matrix ({0})'.format(0))
    plt.tight_layout()
    
    ax = fig.add_subplot(111)
# Plot neurons
    im = ax.imshow(u_matrix_train[0], cmap='gray', interpolation='nearest')
    # Plot colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Mean distance with neighbors', rotation=-90, va="bottom")
    # def init():
    #     cbar_min,cbar_max = u_matrix_train[0].min(), u_matrix_train[0].max()
    #     im.set_clim(cbar_min,cbar_max)
    #     im.set_data(np.zeros((k, k)))
    #     return im,

    def animate(i):
        # Add country name with font size of 20
        cbar_min,cbar_max = 0, u_matrix_train[i].max()
        im.set_data(u_matrix_train[i])
        im.set_clim(cbar_min,cbar_max)
        txt.set_text('U matrix ({0})'.format(i))
        # return im,

    anim = animation.FuncAnimation(fig, animate,
                                   frames=len(u_matrix_train), repeat=False)
    plt.show()
def plot_u_matrix(u_matrix):
    plt.figure(2)
    plt.title('U matrix')
    plt.tight_layout()

    plt.imshow(u_matrix, cmap='gray', interpolation='nearest',clim=(0,u_matrix.max()))
    # Plot colorbar
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Mean distance with neighbors', rotation=-90, va="bottom")
    # Limit color bar values

def main():
    data = read_dataset('./datasets/europe.csv')

    # remove Country column from dataset
    data_no_countries = data.drop(['Country'], axis=1)
    # scale dataset
    data_scaled = (data_no_countries - data_no_countries.mean()
                   ) / data_no_countries.std()
    
    data_scaled_numpy = data_scaled.to_numpy()
    print(len(data_scaled_numpy[0]))
    k = 5
    kohonen = Kohonen(k, data_scaled_numpy, k, 0.01)
    winner_idx_arr_row, winner_idx_arr_col, radius_arr, learning_rate_arr, dist_arr,u_matrix_train = kohonen.train(
        data_scaled_numpy, 300)

    winners_sequence, winners = kohonen.test(data_scaled_numpy)
    u_matrix = kohonen.get_u_matrix()
    plot_u_matrix(u_matrix)
    plot_u_matrix_anim(u_matrix_train)
    pprint(u_matrix)
    pprint(u_matrix_train[-1])
    for i in range (len(data_scaled_numpy[0])):
        print(i)
        heatmap = kohonen.get_mean_column_weight(i)
        plot_heatmap(heatmap, i+4, data.columns[i+1])

    plot_map(winners_sequence, winners, data['Country'], k)


if __name__ == '__main__':
    main()
