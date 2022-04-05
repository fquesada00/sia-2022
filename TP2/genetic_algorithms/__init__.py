import random
import time
from ..genetic_algorithms.selection_methods import uniform_selection
from ..genetic_algorithms.crossover_methods import CrossoverParameters
from ..genetic_algorithms.mutation_methods import MutationParameters
from ..models.Summary import Summary
from ..models.Chromosome import Chromosome
from ..models.GenerationsPrinter import GenerationsPrinter

from ..genetic_algorithms import cut_conditions

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


def evolve(population, fitness_function, selection_parameters, crossover_parameters: CrossoverParameters, mutation_parameters: MutationParameters, cut_condition_parameters):
    new_population = []

    while len(new_population) < len(population):
        # Select parents
        parent1, parent2 = uniform_selection(
            population, fitness_function, 2, selection_parameters)

        # Reproduce
        child1, child2 = crossover_parameters.crossover_method(
            parent1, parent2, len(parent1), crossover_parameters)

        # Mutate
        child1.mutate(mutation_parameters)
        child2.mutate(mutation_parameters)

        # Add to new population
        new_population.append(child1)
        new_population.append(child2)

    return selection_parameters.selection_method(new_population + population, fitness_function, len(population), selection_parameters)


def optimize(initial_population, fitness_function, selection_parameters, crossover_parameters: CrossoverParameters, mutation_parameters: MutationParameters, cut_condition_parameters, output_filename=None):

    global number_of_generations
    number_of_generations = 0

    cut_conditions.prev_best_fitness = None
    cut_conditions.repeated_generations = 0
    
    generations_printer = None
    
    if output_filename is not None:
        generations_printer = GenerationsPrinter(
            output_filename, selection_parameters, crossover_parameters, mutation_parameters, cut_condition_parameters)

        generations_printer.open_file()
        generations_printer.print_initial_parameters()

    population = initial_population

    start_time = time.time()
    while not cut_condition_parameters.cut_condition_method(population, fitness_function, time.time() - start_time, cut_condition_parameters):
        best_individual = sorted(
            population, key=fitness_function, reverse=True)[0]

        if output_filename is not None:
            generations_printer.print_generation(fitness_function(best_individual))

        print(
            f'Generation N°{number_of_generations}: {best_individual} - Fitness: {fitness_function(best_individual)}\n')

        # measure execution time for evolve
        population = evolve(population, fitness_function,
                            selection_parameters, crossover_parameters, mutation_parameters, mutation_parameters)
        number_of_generations += 1

    if output_filename is not None:
        generations_printer.close_file()

    end_time = time.time()

    best_individual = sorted(population, key=fitness_function, reverse=True)[0]
    best_individual_fitness = fitness_function(best_individual)

    return Summary(best_individual, best_individual_fitness, end_time - start_time)
