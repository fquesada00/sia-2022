from cmath import log
import unittest
from unittest import mock
from genetic_algorithms.crossover_methods import simple_crossover, multiple_crossover, uniform_crossover

from models.Chromosome import Chromosome


class TestCrossoverMethods(unittest.TestCase):

    def setUp(self):
        self.first_parent = Chromosome(genes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.second_parent = Chromosome(
            genes=[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])
        self.genes_length = len(self.first_parent)

    def tearDown(self):
        self.first_parent = None
        self.second_parent = None
        self.genes_length = None

    @mock.patch('random.randint')
    def test_simple(self, randint_mock):
        randint_mock.return_value = 4
        first_child, second_child = simple_crossover(
            self.first_parent, self.second_parent, self.genes_length)
        self.assertEqual(first_child, [-1, -2, -3, -4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(second_child, [1, 2, 3, 4, -5, -6, -7, -8, -9, -10])

    @mock.patch('random.sample')
    def test_multiple(self, sample_mock):
        number_of_points = 3
        sample_mock.return_value = [2, 8, 5]
        first_child, second_child = multiple_crossover(
            self.first_parent, self.second_parent, self.genes_length, number_of_points)
        self.assertEqual(first_child, [1, 2, -3, -4, -5, 6, 7, 8, -9, - 10])
        self.assertEqual(second_child, [-1, -2, 3, 4, 5, -6, -7, -8, 9, 10])

    @mock.patch('random.random')
    def test_uniform(self, random_mock):
        random_mock.side_effect = [0.1, 0.9, 0.2,
                                   0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5]
        first_child, second_child = uniform_crossover(
            self.first_parent, self.second_parent, self.genes_length)

        self.assertEqual(first_child, [-1, 2, -3, 4, -5, 6, -7, 8, -9, -10])
        self.assertEqual(second_child, [1, -2, 3, -4, 5, -6, 7, -8, 9, 10])


if __name__ == '__main__':
    unittest.main()
