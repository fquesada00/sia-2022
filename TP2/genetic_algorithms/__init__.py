import random

from TP2.constants import MAX_REAL, MIN_REAL
from TP2.models.Chromosome import Chromosome


def generate_random_individual(min, max):
    W = [random.uniform(min, max) for _ in range(3)]
    w = [random.uniform(min, max) for _ in range(6)]
    w_0 = [random.uniform(min, max) for _ in range(2)]

    return Chromosome((*W, *w, *w_0))


def generate_initial_population(size, min, max):
    population = []

    for _ in range(size):
        population.append(generate_random_individual(min, max))

    return population


def evolve(population, fitness_function, select, crossover):
    new_population = []

    for _ in range(len(population)):
        parent1, parent2 = select(population, fitness_function, 2)
        child1, child2 = crossover(parent1, parent2)
        new_population.append(child1, child2)

    return new_population


def optimize(population_size, fitness_function, select, crossover, cut_condition, min, max):
    population = generate_initial_population(population_size, min, max)

    while not cut_condition(population):
        population = evolve(population, fitness_function, select, crossover)

    return population
