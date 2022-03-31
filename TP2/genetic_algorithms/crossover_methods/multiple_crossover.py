import random

from TP2.models.Chromosome import Chromosome

def multiple_crossover(first_parent, second_parent, genes_length, number_of_points):

    random_indexes = sorted(random.sample(range(1, genes_length - 2), number_of_points))

    last_index = 0
    first_child_genes = []
    second_child_genes = []
    swap = False

    for random_index in random_indexes:
        if swap:
            first_child_genes += second_parent[last_index:random_index] 
            second_child_genes += first_parent[last_index:random_index] 
        else:
            first_child_genes += first_parent[last_index:random_index] 
            second_child_genes += second_parent[last_index:random_index]
        last_index = random_index
        swap = not swap

    return Chromosome(first_child_genes), Chromosome(second_child_genes)   