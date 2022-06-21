
import os


if __name__ == "__main__":
    latent_space = 2
    architectures = [
        [25, 10]
    ]

    epochs = 200

    background = False

    read_weights = True

    for architecture in architectures:
        cmd = f'python3 main.py --latent {latent_space} --architecture "{str(architecture)}" --epochs {epochs} {"&" if background else ""} {"--read_weights" if read_weights else ""}'
        print(cmd)
        os.system(cmd)
