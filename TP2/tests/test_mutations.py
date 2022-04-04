import unittest
from unittest import mock

from genetic_algorithms.mutation_methods.normal_mutation import normal_mutation


class TestSelectionMethods(unittest.TestCase):

    def setUp(self):
        genes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.genes = genes
        self.mutation_rate = 0.1

    def tearDown(self):
        self.genes = None
        self.mutation_rate = None

    @mock.patch('random.random')
    @mock.patch('random.normalvariate')
    def test_normal_mutation(self, normal_mock, random_mock):
        # mutate genes 1 2 6 7
        random_mock.side_effect = [0.5, 0.05, 0.01,
                                   0.7, 0.8, 0.9, 0.09, 0.02, 0.4, 0.3]
        normal_mock.side_effect = [-4.9760e-2, -
                                   3.4850e-2, 1.5750e-1, 4.0010e-2]

        genes_copy = self.genes.copy()
        genes_copy[1] += -4.9760e-2
        genes_copy[2] += -3.4850e-2
        genes_copy[6] += 1.5750e-1
        genes_copy[7] += 4.0010e-2

        self.assertEqual(normal_mutation(
            self.genes, self.mutation_rate), genes_copy)


if __name__ == '__main__':
    unittest.main()
