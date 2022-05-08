import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", help="Number of epochs to train the network.",
                        dest='epochs', required=False, default='10')

    args = parser.parse_args()

    epochs = int(args.epochs)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        # Execute python script
        cmd = f"python -m TP3.ej2.main --epochs {epoch + 1}"
        print(cmd)
        os.system(cmd)

        # TODO: Reset metrics and weights file
        open('TP3/metrics.txt', 'w').close()
        open('TP3/weights.txt', 'w').close()