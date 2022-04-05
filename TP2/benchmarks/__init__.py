
class BenchmarkParameters():

    def __init__(self, selection_parameters, crossover_parameters, mutation_parameters, cut_condition_parameters, population_size, min_real, max_real, bounds, mutation_rates, multiple_points, crossover_output_filename, mutation_output_filename, cut_condition_output_filename, selection_output_filename, initial_bounds_output_filename, mutation_rate_output_filename):
        self._selection_parameters = selection_parameters
        self._crossover_parameters = crossover_parameters
        self._mutation_parameters = mutation_parameters
        self._cut_condition_parameters = cut_condition_parameters
        self._population_size = population_size
        self._min_real = min_real
        self._max_real = max_real
        self._bounds = bounds
        self._mutation_rates = mutation_rates
        self._multiple_points = multiple_points
        self._crossover_benchmark_output_filename = crossover_output_filename
        self._mutation_benchmark_output_filename = mutation_output_filename
        self._cut_condition_benchmark_output_filename = cut_condition_output_filename
        self._selection_benchmark_output_filename = selection_output_filename
        self._initial_bounds_output_filename = initial_bounds_output_filename
        self._mutation_rate_output_filename = mutation_rate_output_filename

    @property
    def selection_parameters(self):
        return self._selection_parameters

    @selection_parameters.setter
    def selection_parameters(self, value):
        self._selection_parameters = value

    @property
    def crossover_parameters(self):
        return self._crossover_parameters

    @crossover_parameters.setter
    def crossover_parameters(self, value):
        self._crossover_parameters = value

    @property
    def mutation_parameters(self):
        return self._mutation_parameters

    @mutation_parameters.setter
    def mutation_parameters(self, value):
        self._mutation_parameters = value

    @property
    def cut_condition_parameters(self):
        return self._cut_condition_parameters

    @cut_condition_parameters.setter
    def cut_condition_parameters(self, value):
        self._cut_condition_parameters = value

    @property
    def multiple_points(self):
        return self._multiple_points
    
    @multiple_points.setter
    def multiple_points(self, value):
        self._multiple_points = value

    @property
    def population_size(self):
        return self._population_size

    @population_size.setter
    def population_size(self, value):
        self._population_size = value

    @property
    def min_real(self):
        return self._min_real

    @min_real.setter
    def min_real(self, value):
        self._min_real = value

    @property
    def max_real(self):
        return self._max_real

    @max_real.setter
    def max_real(self, value):
        self._max_real = value

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        self._bounds = value

    @property
    def crossover_benchmark_output_filename(self):
        return self._crossover_benchmark_output_filename

    @crossover_benchmark_output_filename.setter
    def crossover_benchmark_output_filename(self, value):
        self._crossover_benchmark_output_filename = value

    @property
    def mutation_benchmark_output_filename(self):
        return self._mutation_benchmark_output_filename

    @mutation_benchmark_output_filename.setter
    def mutation_benchmark_output_filename(self, value):
        self._mutation_benchmark_output_filename = value

    @property
    def cut_condition_benchmark_output_filename(self):
        return self._cut_condition_benchmark_output_filename

    @cut_condition_benchmark_output_filename.setter
    def cut_condition_benchmark_output_filename(self, value):
        self._cut_condition_benchmark_output_filename = value

    @property
    def selection_benchmark_output_filename(self):
        return self._selection_benchmark_output_filename

    @selection_benchmark_output_filename.setter
    def selection_benchmark_output_filename(self, value):
        self._selection_benchmark_output_filename = value

    @property
    def initial_bounds_output_filename(self):
        return self._initial_bounds_output_filename

    @initial_bounds_output_filename.setter
    def initial_bounds_output_filename(self, value):
        self._initial_bounds_output_filename = value

    @property
    def mutation_rate_output_filename(self):
        return self._mutation_rate_output_filename

    @mutation_rate_output_filename.setter
    def mutation_rate_output_filename(self, value):
        self._mutation_rate_output_filename = value

    @property
    def mutation_rates(self):
        return self._mutation_rates

    @mutation_rates.setter
    def mutation_rates(self, value):
        self._mutation_rates = value
