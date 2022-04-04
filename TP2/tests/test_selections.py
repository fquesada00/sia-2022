from cmath import log
import math
import random
import unittest
from unittest import mock
from genetic_algorithms.selection_methods import uniform_selection, truncate_selection, tournament_selection

from models.Chromosome import Chromosome


class TestSelectionMethods(unittest.TestCase):

    def setUp(self):
        population = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 0
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 1
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # 2
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],  # 3
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],  # 4
            [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # 5
            [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        ]
        self.population = population
        self.fitness_function = lambda x: sum(x)
        self.selection_size = 3

    def tearDown(self):
        self.first_parent = None
        self.second_parent = None
        self.genes_length = None

    @mock.patch('random.randint')
    def test_uniform(self, randint_mock):
        randint_mock.side_effect = [8, 5, 1]
        selection = uniform_selection(
            self.population, self.fitness_function, self.selection_size)
        self.assertEqual(
            selection, [self.population[8], self.population[5], self.population[1]])

    @mock.patch('random.randint')
    def test_truncate(self, randint_mock):
        randint_mock.side_effect = [2, 3, 0]
        k = 5
        selection = truncate_selection(
            self.population, self.fitness_function, self.selection_size, k)
        sorted_populations = sorted(self.population, key=self.fitness_function)
        self.assertEqual(
            selection, [sorted_populations[k+2], sorted_populations[k+4], sorted_populations[k+0]])

    @mock.patch('random.choice')
    @mock.patch('random.random')
    def test_tournament(self, random_choice, choice_mock):
        choice_mock.side_effect = [0, 3, 1, 2]
        randoms = [0.7]
        randoms.append(0.8)
        randoms.append(0.5)
        randoms.append(0.9)

        random_choice.side_effect = randoms

        selection = tournament_selection(
            self.population, self.fitness_function, 1)
        self.assertEqual(
            selection, [self.population[0]])


if __name__ == '__main__':
    unittest.main()
