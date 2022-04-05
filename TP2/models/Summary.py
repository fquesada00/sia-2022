from ..optimization_problem import error,function_to_approximate
from ..constants import DATASET_INPUT, DATASET_OUTPUT
class Summary:

    def __init__(self, best_individual, fitness, execution_time):
        self.best_individual = best_individual
        self.W = best_individual.W
        self.w = best_individual.w
        self.w_0 = best_individual.w_0
        self.fitness = fitness
        self.execution_time = execution_time
        self.error = error(self.W,self.w,self.w_0,DATASET_INPUT,DATASET_OUTPUT)
        self.F_1 = function_to_approximate(self.W,self.w,self.w_0,DATASET_INPUT[0])
        self.F_2 = function_to_approximate(self.W,self.w,self.w_0,DATASET_INPUT[1])
        self.F_3 = function_to_approximate(self.W,self.w,self.w_0,DATASET_INPUT[2])  

    def __str__(self):
        return f'- Execution time: {self.execution_time} s \n- Best individual: {self.best_individual} \
            \n- Fitness: {self.fitness} \n- W: {self.W} \n- w: {self.w} \n- w_0: {self.w_0} \
            \n- Error: {self.error}\
            \n- F(W,w,w_p,xi_1): {self.F_1}\n\
            \n- F(W,w,w_p,xi_2): {self.F_2}\n\
            \n- F(W,w,w_p,xi_3): {self.F_3}\n'

    def to_csv(self):
        return f'{self.execution_time},{self.fitness},{self.W[0]},{self.W[1]},{self.W[2]},{self.w[0]},{self.w[1]},{self.w[2]},{self.w[3]},{self.w[4]},{self.w[5]},{self.w_0[0]},{self.w_0[1]},{self.error},{self.F_1},{self.F_2},{self.F_3}\n'