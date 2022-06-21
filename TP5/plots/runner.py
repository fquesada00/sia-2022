
import os


if __name__ == "__main__":
    latent_space = 2
    architectures = [
        [25, 10],
        [25, 10, 5],
        [20, 10, 2],
        [20, 10, 5, 2],
        [30, 20, 10, 2],
        [30, 25, 10, 5, 2]
    ]

    epochs = 200

    for architecture in architectures:
        cmd = f'python3 main.py --latent {latent_space} --architecture "{str(architecture)}" --epochs {epochs}&'
        print(cmd)
        os.system(cmd)
