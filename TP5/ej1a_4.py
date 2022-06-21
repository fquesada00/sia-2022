import time
from matplotlib import pyplot as plt
import numpy as np
from utils import plot_decoded_latent_space
from models.Autoencoder import Autoencoder, create_autoencoder
from utils import to_bin_array, to_raw_dataset
from models.Model import Model
from utils.dataset import font_1, font_2, font_3
monocromatic_cmap = plt.get_cmap('binary')



def get_direction_vector(source, target):
    return (target[0] - source[0], target[1] - source[1])


def get_distance(source, target):
    return ((target[0] - source[0]) ** 2 + (target[1] - source[1]) ** 2) ** 0.5


def normalize_vector(vector, distance):
    print("vector unit & distance: ", vector, distance)
    if distance == 0:
        return (0, 0)
    return (vector[0] / distance, vector[1] / distance)


def move_point_in_direction(point, direction, distance):
    return (point[0] + direction[0] * distance, point[1] + direction[1] * distance)


def get_n_points_in_line(source, target, n):
    """
    Returns a list of n points in the line from source to target including source & target.
    """
    direction = get_direction_vector(source, target)
    distance = get_distance(source, target)
    normalized_direction = normalize_vector(direction, distance)
    step_size = distance / (n + 1)
    points = []
    for i in range(n+2):
        points.append(move_point_in_direction(
            source, normalized_direction, step_size * i))
    
    print("points: ", points)

    for index in range(1, len(points)):
        if points[index][0] == points[index-1][0]:
            slope = float("inf")
        elif points[index][1] == points[index-1][1]:
            slope = 0
        else:
            slope = (points[index][1] - points[index-1][1]) / (points[index][0] - points[index-1][0])
        # print("slope: ", slope)
    return points


def get_all_farthest_points_tuple(points: list[dict]):
    points_with_distance = []
    for i in range(len(points)):
        for j in range(i):
            distance = get_distance(points[i]["point"], points[j]["point"])
            if distance == 0:
                continue

            # if points[i]["char"] == "B" and points[j]["char"] == "A":
            points_with_distance.append({
                "distance": distance,
                "points": (points[i]["point"], points[j]["point"]),
                "chars": (points[i]["char"], points[j]["char"])
            })
    points_with_distance.sort(key=lambda x: x["distance"], reverse=True)
    return points_with_distance


def get_source_and_target_from_farthest_points(farthest_points):
    # Fetch 2 farthest points
    sources = []
    targets = []
    for i in range(2):
        sources.append({"point": farthest_points[i]["points"][0],
                        "char": farthest_points[i]["chars"][0],
                        "distance": farthest_points[i]["distance"]})
        targets.append({"point": farthest_points[i]["points"][1],
                        "char": farthest_points[i]["chars"][1],
                        "distance": farthest_points[i]["distance"]})
    
    # Fetch 2 intermediate farthest points
    if len(farthest_points) > 6:
        middle_index = int(len(farthest_points) / 2)
        for i in range(2):
            sources.append({"point": farthest_points[middle_index + i]["points"][0],
                            "char": farthest_points[middle_index + i]["chars"][0],
                            "distance": farthest_points[middle_index + i]["distance"]})
            targets.append({"point": farthest_points[middle_index + i]["points"][1],
                            "char": farthest_points[middle_index + i]["chars"][1],
                            "distance": farthest_points[middle_index + i]["distance"]})

            middle_index += 1
        
    # Fetch 2 nearest points
    for i in range(2):
        sources.append({"point": farthest_points[-i-1]["points"][0],
                        "char": farthest_points[-i-1]["chars"][0],
                        "distance": farthest_points[-i-1]["distance"]})
        targets.append({"point": farthest_points[-i-1]["points"][1],
                        "char": farthest_points[-i-1]["chars"][1],
                        "distance": farthest_points[-i-1]["distance"]})
    
    return sources, targets


def generate_n_new_letters(decoder: Model, source: dict, target: dict, n: int = 1) -> list[dict]:
    """
    Decode to generate n new letters between source & target.
    """
    points = get_n_points_in_line(source["point"], target["point"], n)
    decoded_letters = []
    for i in range(n+2):
        point = (points[i][0], points[i][1])
        decoded_letters.append({
            "letter": decoder(np.array([point[0], point[1]])).reshape((7, 5)),
            "point": point,
            "chars": (source["char"], target["char"]),
            "distance": source["distance"]
        })
    return decoded_letters


def generate_n_new_letters_per_source_and_target(decoder: Model, sources: list[dict], targets: list[dict], n: int) -> list[list[dict]]:
    generated_letters = []
    for source, target in zip(sources, targets):
        generated_letters.append(generate_n_new_letters(decoder, source, target, n=n))
    return generated_letters


def plot_generated_letters(generated_letters: list[list[dict]]):
    rows = len(generated_letters)
    columns = len(generated_letters[0])
    fig, axs = plt.subplots(
        rows, columns, sharey=False, tight_layout=True, figsize=(12, 6), facecolor='white')

    for i in range(rows):
        for j in range(columns):
            letter = generated_letters[i][j]["letter"]
            point = generated_letters[i][j]["point"]
            source_char = generated_letters[i][j]["chars"][0]
            target_char = generated_letters[i][j]["chars"][1]
            distance = generated_letters[i][j]["distance"]

            if rows == 1:
                ax = axs[j]
            else:
                ax = axs[i][j]

            ax.imshow(letter, cmap=monocromatic_cmap)
            ax.set_title(f"{source_char} -> {target_char}")
            # ax.title.set_size(12)

            # ax.set_title(f"{source_char} -> {target_char} ({point[0]}, {point[1]})")
            # ax.set_title(f"{source_char} -> {target_char} {distance}")

            axs[i][j].axis('off')
    plt.axis('off')
    plt.show()
    

def plot_generated_lines(generated_letters: list[list[dict]]):
    legends = []
    annotated_chars = []
    colors = ["blue", "orange", "green", "red", "purple", "brown", "black", "cyan", "yellow"]
    for index, letter_sequence in enumerate(generated_letters):
        x_axis = []
        y_axis = []
        for letter in letter_sequence:
            x_axis.append(letter["point"][0])
            y_axis.append(letter["point"][1])

        if letter_sequence[0]['chars'][0] not in annotated_chars:
            plt.annotate(letter_sequence[0]['chars'][0], xy=(letter_sequence[0]['point'][0], letter_sequence[0]['point'][1]))
            annotated_chars.append(letter_sequence[0]['chars'][0])
        
        if letter_sequence[-1]['chars'][1] not in annotated_chars:
            plt.annotate(letter_sequence[0]['chars'][1], xy=(letter_sequence[-1]['point'][0], letter_sequence[-1]['point'][1]))
            annotated_chars.append(letter_sequence[0]['chars'][1])
            
        legends.append(f"{letter_sequence[0]['chars'][0]} -> {letter_sequence[0]['chars'][1]} - {round(letter_sequence[0]['distance'], 2)}")
        plt.plot(x_axis, y_axis, "-o", color=colors[index])
        # plt.plot([letter_sequence[0]['point'][0], letter_sequence[-1]['point'][0]], [letter_sequence[0]['point'][1], letter_sequence[-1]['point'][1]], "-o", color=colors[index])
    plt.legend(legends)
    plt.tight_layout()
    plt.grid()
    plt.show()


def create_and_train_default_autoencoder(font, total_fonts: int = 5, epochs: int = 2):
    labelled_dataset = font

    raw_dataset = np.array([data.flatten() for data in map(
        to_bin_array, to_raw_dataset(labelled_dataset))])
    
    raw_dataset = raw_dataset[:total_fonts]

    # Create autoencoder
    input_size = len(raw_dataset[0])
    latent_space_size = 2
    encoder_layers = [20, 10, latent_space_size]
    decoder_layers = [10, 20, input_size]
    encoder_activations = ["logistic", "relu", "logistic"]
    decoder_activations = ["logistic", "relu", "logistic"]

    encoder, decoder = create_autoencoder(
        input_size, encoder_layers, decoder_layers, encoder_activations, decoder_activations)

    autoencoder = Autoencoder(encoder, decoder, optimizer='powell')

    start = time.time()
    autoencoder.fit(raw_dataset, raw_dataset, epochs=epochs)
    end = time.time()

    print("Training time: ", end - start)

    return encoder, decoder, autoencoder


def save_farthest_points_to_csv(farthest_points: list[dict], filename: str):
    with open(filename, "w") as f:
        f.write("source_char,target_char,source_point,target_point,distance\n")
        top_10_farthest_points = farthest_points[:10]
        middle_index = len(farthest_points) // 2
        top_10_intermediate_points = farthest_points[middle_index - 5:middle_index + 5]
        top_10_closer_points = farthest_points[-10:]
        save_points = top_10_farthest_points + top_10_intermediate_points + top_10_closer_points
        for point_data in save_points:
            f.write(f"{point_data['chars'][0]},{point_data['chars'][1]},({point_data['points'][0][0]}; {point_data['points'][0][1]}),({point_data['points'][1][0]}; {point_data['points'][1][1]}),{point_data['distance']}\n")

def main():
    np.random.seed(2)
    font = font_2
    total_fonts = 32
    epochs = 2
    encoder, decoder, autoencoder = create_and_train_default_autoencoder(font, total_fonts=total_fonts, epochs=epochs)
    plot_decoded_latent_space(decoder)
    latent_space_representation = [{"point": encoder(to_bin_array(font[i]["bitmap"]).flatten()), "char": font[i]["char"]} for i in range(total_fonts)]
    farthest_points = get_all_farthest_points_tuple(latent_space_representation)
    save_farthest_points_to_csv(farthest_points, "farthest_points.csv")
    sources, targets = get_source_and_target_from_farthest_points(farthest_points)
    number_of_points_in_line = 3
    generated_letters = generate_n_new_letters_per_source_and_target(decoder, sources, targets, n=number_of_points_in_line)
    plot_generated_letters(generated_letters)
    plot_generated_lines(generated_letters)


if __name__ == "__main__":
    main()