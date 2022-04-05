class Parameters():

    def __init__(self, selection_parameters, crossover_parameters, mutation_parameters, cut_condition_parameters):
        self._selection_parameters = selection_parameters
        self._crossover_parameters = crossover_parameters
        self._mutation_parameters = mutation_parameters
        self._cut_condition_parameters = cut_condition_parameters

    def __str__(self):
        return str(self.selection_parameters) + "\n" +\
            str(self.crossover_parameters) + "\n" +\
            str(self.mutation_parameters) + "\n" +\
            str(self.cut_condition_parameters) + "\n"

    @property
    def selection_parameters(self):
        return self._selection_parameters

    @property
    def crossover_parameters(self):
        return self._crossover_parameters

    @property
    def mutation_parameters(self):
        return self._mutation_parameters

    @property
    def cut_condition_parameters(self):
        return self._cut_condition_parameters
