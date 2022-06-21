from matplotlib import pyplot as plt
import numpy as np


def plot_loss_from_txt(filename, epochs=100):
    with open(filename, "r") as f:
        legends = []
        x_values = np.linspace(0, epochs, epochs)
        for index, line in enumerate(f):
            line_split_by_semicolon = line.split(";")
            if len(line_split_by_semicolon) == 1:
                legends.append(line.split("\n")[0])
            else:
                line_split_by_semicolon.pop()
                y_values = np.array(line_split_by_semicolon).astype(np.float)
                plt.plot(x_values, y_values)
        plt.legend(legends)
        plt.ylabel("Error")
        plt.xlabel("Epoch")
        plt.show()

def main():
    plot_loss_from_txt("vae_fashion_mnist_history.txt")

if __name__ == "__main__":
    main()
