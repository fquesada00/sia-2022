
import os


if __name__ == "__main__":
    latent_space = 2
    architecture = [25, 10]
    lambda_terms = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    reg_term = "l1"

    epochs = 100

    background = True

    read_weights = False

    for lambda_term in lambda_terms:
        cmd = f'python3 main.py --latent {latent_space} --architecture "{str(architecture)}" --epochs {epochs} {"--read_weights" if read_weights else ""} --reg-term {reg_term} --lambda_var {lambda_term} {"&" if background else ""}'
        print(cmd)
        os.system(cmd)
