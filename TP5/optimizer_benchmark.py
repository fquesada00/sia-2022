
import os


if __name__ == "__main__":
    latent_space = 2
    architecture = [25, 10]

    optimizers = [
        "bfgs"
    ]

    epochs = 200

    for optimizer in optimizers:
        cmd = f'python3 main.py --latent {latent_space} --architecture "{str(architecture)}" --epochs {epochs} --optimizer {optimizer} &'
        print(cmd)
        os.system(cmd)
