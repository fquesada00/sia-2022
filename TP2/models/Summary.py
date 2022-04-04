

class Summary:

    def __init__(self, best_individual, fitness, execution_time):
        self.best_individual = best_individual
        self.fitness = fitness
        self.execution_time = execution_time

    def __str__(self):
        return f'- Execution time: {self.execution_time} s \n- Best individual: {self.best_individual} \n- Fitness: {self.fitness}'
