from cmath import log
import unittest
from unittest import mock
from genetic_algorithms.selection_methods import truncate_selection, tournament_selection, roulette_selection, rank_selection, elite_selection, boltzmann_selection
from models.Chromosome import Chromosome


class TestSelectionMethods(unittest.TestCase):

    def setUp(self):
        population = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 0 -> 55 || -> 0.1
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 1 -> 65 || -> 0.2
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # 2 -> 75 || -> 0.3
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],  # 3 -> 85 || -> 0.4
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],  # 4 -> 95 || -> 0.5
            [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # 5 -> 105 || -> 0.6
            [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # 6 -> 115 || -> 0.7
            [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],  # 7 -> 125 || -> 0.8
            [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],  # 8 -> 135 || -> 0.9
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  # 9 -> 145 || -> 1.0
            #                                        total -> 1000 || -> 5.5
        ]
        self.population = population
        self.fitness_function = lambda x: sum(x)
        self.selection_size = 3

    def tearDown(self):
        self.first_parent = None
        self.second_parent = None
        self.genes_length = None

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

    @mock.patch('random.random')
    def test_roulette(self, random_mock):
        random_mock.side_effect = [0.49]
        selection = roulette_selection(
            self.population, self.fitness_function, 1)
        self.assertEqual(selection, [self.population[6]])

    @mock.patch('random.random')
    def test_rank(self, random_mock):
        random_mock.side_effect = [0.019]
        selection = rank_selection(
            self.population, self.fitness_function, 1)
        self.assertEqual(selection, [self.population[1]])

    def test_elite_selection(self):
        selection = elite_selection(
            self.population, self.fitness_function, self.selection_size)
        population_copy = self.population.copy()
        population_copy.reverse()
        self.assertEqual(selection, population_copy[:self.selection_size])

    @mock.patch("genetic_algorithms.number_of_generations", 1)
    @mock.patch("random.random")
    def test_boltzmann_selection(self, mock_random):
        mock_random.side_effect = [0.35]
        selection = boltzmann_selection(
            self.population, self.fitness_function, 1)
        self.assertEqual(selection, [self.population[4]])

    @mock.patch("genetic_algorithms.number_of_generations", 1)
    @mock.patch("random.random")
    def test_boltzmann_selection_fail(self, mock_random):
        mock_random.side_effect = [0.38]
        selection = boltzmann_selection(
            self.population, self.fitness_function, 1)
        self.assertNotEqual(selection, [self.population[4]])


if __name__ == '__main__':
    unittest.main()
