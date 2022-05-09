import argparse
import json
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", help="Number of epochs to train the network.",
                        dest='epochs', required=False, default='50')
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

    batches = [10, 20, 50, 100, 200]
    for batch in batches:
        print(f"Batch size: {batch}")

            
        config_file = open("TP3/config.json", "r")
        json_object = json.load(config_file)
        config_file.close()

        json_object["optimal_parameters"]["ej_2_non_linear"]["training"]["batch_size"] = batch
        config_file = open("TP3/config.json", "w")
        json.dump(json_object, config_file)
        config_file.close()

        for iteration in range(iterations):
            print(f"Iteration {iteration + 1} of {iterations}")
            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}")
                # Execute python script
                cmd = f"python -m TP3.ej2.main --epochs {epoch + 1} --log-test --log-train --split-method holdout --ratio 0.9 --seed {iteration + 2}"
                # cmd = f"python -m TP3.ej2.main --epochs {epoch + 1} --log-train --seed {iteration + 2}"
                print(cmd)
                os.system(cmd)

                open('TP3/metrics.txt', 'w').close()
                open('TP3/weights.txt', 'w').close()
                open('TP3/test_metrics.txt', 'w').close()
