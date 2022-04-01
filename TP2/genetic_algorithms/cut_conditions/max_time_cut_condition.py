from constants import MAX_TIME


def max_time_cut_condition(population, elapsed_time, fitness_function):
    return elapsed_time > MAX_TIME
