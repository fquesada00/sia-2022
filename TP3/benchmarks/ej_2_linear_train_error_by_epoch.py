import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", help="Number of epochs to train the network.",
                        dest='epochs', required=False, default='10')
    parser.add_argument("--iterations", help="Number of iterations per epoch.",
                        dest='iterations', required=False, default='5')

    args = parser.parse_args()

    epochs = int(args.epochs)
    iterations = int(args.iterations)
    for iteration in range(iterations):
        print(f"Iteration {iteration + 1} of {iterations}")
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}")
            # Execute python script
            cmd = f"python -m TP3.ej2.main --epochs {epoch + 1} --log-test --log-train --split-method holdout"
            print(cmd)
            os.system(cmd)

            open('TP3/metrics.txt', 'w').close()
            open('TP3/weights.txt', 'w').close()
            open('TP3/test_metrics.txt', 'w').close()
