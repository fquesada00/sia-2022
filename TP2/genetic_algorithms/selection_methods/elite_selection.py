

def elite_selection(population, fitness_function, selection_size, selection_parameter):
    population_sorted = sorted(
        population, key=lambda x: fitness_function(x), reverse=True)
    return population_sorted[:selection_size]
