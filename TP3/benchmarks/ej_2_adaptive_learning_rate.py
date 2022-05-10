import argparse
import json
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", help="Number of epochs to train the network.",
                        dest='epochs', required=False, default='15')
    parser.add_argument("--iterations", help="Number of iterations per epoch.",
                        dest='iterations', required=False, default='5')

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

    adaptives = [
        {
            "a": 0,
            "b": 0,
        }, {
            "a": 0.05,
            "b": 0.05,
        }, {
            "a": 0.1,
            "b": 0.1
        }, {
            "a": 0.2,
            "b": 0.2
        }, {
            "a": 0.3,
            "b": 0.3
        }]

    for adaptive in adaptives:
        print(f"Adaptive: {adaptive}")

        config_file = open("TP3/config.json", "r")
        config = json.load(config_file)
        config_file.close()

        config["optimal_parameters"]["ej_2_non_linear"]["training"]["use_adaptive_learning_rate"] = True
        config["optimal_parameters"]["ej_2_non_linear"]["training"]["alpha"] = adaptive["a"]
        config["optimal_parameters"]["ej_2_non_linear"]["training"]["beta"] = adaptive["b"]

        config_file = open("TP3/config.json", "w")

        json.dump(config, config_file)

        config_file.close()

        for iteration in range(iterations):
            print(f"Iteration {iteration + 1} of {iterations}")

            # Execute python script
            # cmd = f"python -m TP3.ej2.main --epochs {epoch + 1} --log-test --log-train --split-method holdout --ratio 0.9 --seed {iteration + 2}"
            cmd = f"sudo  -E PATH=$PATH nice -n -20 python -m TP3.ej2.main --epochs {epochs} --log-train --seed {iteration + 2}"
            print(cmd)
            os.system(cmd)

            open('TP3/metrics.txt', 'w').close()
            open('TP3/weights.txt', 'w').close()
            open('TP3/test_metrics.txt', 'w').close()
