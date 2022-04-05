def get_results_data(results_filename):
    x_data = []
    y_data = []

    with open(results_filename) as results_file:
        for i, line in enumerate(results_file):
            if i < 18:
                continue
            line_data = line.split("\t")
            x_data.append(int(line_data[0]))
            y_data.append(float(line_data[1]))

        # print(x_data,y_data)
        return x_data, y_data