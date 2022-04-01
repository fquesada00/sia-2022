import random

from constants import MAX_REAL, MIN_REAL
from models.Chromosome import Chromosome

number_of_generations = 0


def generate_random_individual(min, max):
    W = [random.uniform(min, max) for _ in range(3)]
    w = [random.uniform(min, max) for _ in range(6)]
    w_0 = [random.uniform(min, max) for _ in range(2)]

    return Chromosome([*W, *w, *w_0])


def generate_initial_population(size, min, max):
    population = []

    for _ in range(size):
        population.append(generate_random_individual(min, max))

    return population


def evolve(population, fitness_function, select, crossover, mutation_method, mutation_rate):
    new_population = []

    while len(new_population) < len(population):
        # Select parents
        parent1, parent2 = select(
            population, fitness_function, 2)

        # Reproduce
        child1, child2 = crossover(parent1, parent2)

        # Mutate
        child1.mutate(mutation_method, mutation_rate)
        child2.mutate(mutation_method, mutation_rate)

        # Add to new population
        new_population.append(child1, child2)

    return select(new_population + population, fitness_function, len(population))


def optimize(population_size, fitness_function, select, crossover, mutation_method, mutation_rate, cut_condition, min, max):
    population = generate_initial_population(population_size, min, max)

    while not cut_condition(population):
        population = evolve(population, fitness_function,
                            select, crossover, mutation_method, mutation_rate)
        number_of_generations += 1

    return population
