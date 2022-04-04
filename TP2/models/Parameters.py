class Parameters():

    def __init__(self, selection_parameters, crossover_parameters, mutation_parameters, cut_condition_parameters):
        self.selection_parameters = selection_parameters
        self.crossover_parameters = crossover_parameters
        self.mutation_parameters = mutation_parameters
        self.cut_condition_parameters = cut_condition_parameters

    def __str__(self):
        return str(self.selection_parameters) + "\n" +\
            str(self.crossover_parameters) + "\n" +\
            str(self.mutation_parameters) + "\n" +\
            str(self.cut_condition_parameters) + "\n"
