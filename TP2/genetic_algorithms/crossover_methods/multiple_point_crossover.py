import random
from ...models.Chromosome import Chromosome


def multiple_point_crossover(first_parent, second_parent, genes_length,crossover_parameters):

    random_indexes = sorted(random.sample(
        range(1, genes_length - 2), crossover_parameters.multiple_point_crossover_points))

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
    
    if swap:
        first_child_genes += second_parent[last_index:]
        second_child_genes += first_parent[last_index:]
    else:
        first_child_genes += first_parent[last_index:]
        second_child_genes += second_parent[last_index:]
    
    return Chromosome(first_child_genes), Chromosome(second_child_genes)
