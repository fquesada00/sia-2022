import genetic_algorithms
class GenerationsPrinter():
    def __init__(self, output_filename, selection_parameters, crossover_parameters, mutation_parameters, cut_condition_parameters):
        self.output_filename = output_filename
        self.selection_parameters = selection_parameters
        self.crossover_parameters = crossover_parameters
        self.mutation_parameters = mutation_parameters
        self.cut_condition_parameters = cut_condition_parameters

    def open_file(self):
        self.file = open(self.output_filename, "w")

    def print_initial_parameters(self):
        self.file.write(str(self.selection_parameters.selection_method_name) + "\n" +
                        str(self.selection_parameters.initial_temperature) + "\n" + str(self.selection_parameters.final_temperature) +
                        "\n" +str(self.selection_parameters.exp_rate) + "\n" + str(self.selection_parameters.k) + "\n" +
                        str(self.selection_parameters.threshold) + "\n")

        self.file.write(str(self.crossover_parameters.crossover_method_name) +
                        "\n" + str(self.crossover_parameters.multiple_point_crossover_points)+ "\n")

        self.file.write(str(self.mutation_parameters.mutation_method_name) + "\n" + str(self.mutation_parameters.mutation_rate) + "\n" +
                        str(self.mutation_parameters.uniform_mutation_bound) + "\n" + str(self.mutation_parameters.normal_mutation_std)+ "\n")

        self.file.write(str(self.cut_condition_parameters.cut_condition_method_name) + "\n" + str(self.cut_condition_parameters.max_generations) + "\n" +
                        str(self.cut_condition_parameters.min_fitness_value) + "\n" + str(self.cut_condition_parameters.fitness_threshold) + "\n" +
                        str(self.cut_condition_parameters.fitness_required_generations_repeats) + "\n" + str(self.cut_condition_parameters.max_time)+ "\n")


    def print_generation(self, generation_fitness):
      self.file.write(str(genetic_algorithms.number_of_generations) +  "\t" + str(generation_fitness) + "\n")

    def close_file(self):
        self.file.close()