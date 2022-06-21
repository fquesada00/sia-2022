
import os


if __name__ == "__main__":
    latent_spaces = [10, 7, 5, 3, 2]
    architecture = [25, 10]
    epochs = 150

    background = False

    read_weights = True

    for latent_space in latent_spaces:
        cmd = f'python3 main.py --latent {latent_space} --architecture "{str(architecture)}" --epochs {epochs} {"&" if background else ""} {"--read_weights" if read_weights else ""}'
        print(cmd)
        os.system(cmd)
