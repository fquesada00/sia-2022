import argparse
import json
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", help="Number of epochs to train the network.",
                        dest='epochs', required=False, default='3')
    parser.add_argument("--iterations", help="Number of iterations per epoch.",
                        dest='iterations', required=False, default='2')

    args = parser.parse_args()

    config_file = open("TP3/config.json", "r")
    json_object = json.load(config_file)
    config_file.close()

    json_object["problem"] = "ej_2_non_linear"
    config_file = open("TP3/config.json", "w")
    json.dump(json_object, config_file)
    config_file.close()

    epochs = int(args.epochs)
    iterations = int(args.iterations)

    k_folds = [5, 10, 15, 20, 25, 30]
    for k_fold in k_folds:
        print(f"K fold: {k_fold}")
        for iteration in range(iterations):
            print(f"Iteration {iteration + 1} of {iterations}")
            # Execute python script
            cmd = f"python -m TP3.ej2.main --epochs {epochs} --log-test --log-train --split-method k-fold --k {k_fold} --seed {iteration + 2}"
            print(cmd)
            os.system(cmd)

            open('TP3/metrics.txt', 'w').close()
            open('TP3/weights.txt', 'w').close()
            open('TP3/test_metrics.txt', 'w').close()
