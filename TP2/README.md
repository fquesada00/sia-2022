# Genetic Algorithms

## Sistemas de Inteligencia Articial - TP2

### ITBA - 2022 1Q

## Authors

- [Serpe, Octavio](github.com/serpe)
- [Quesada, Francisco](github.com/fquesada00)
- [Arca, Gonzalo](github.com/gonzaloarca)

## Dependencies

To run the project, you need to install the following dependencies:

- [Python 3.7](https://www.python.org/downloads/)
- [pipenv](https://docs.pipenv.org/en/latest/)

## Usage

To run the project, you first need to install the project's dependencies. From the `TP2` directory, run:

```bash
pipenv install
```

Then, you can run the project:

```bash
pipenv shell
python -m TP2.main
```

This will run the main program with the configuration from `main_config.json`. If you want to make changes to the configuration, you can do so by editing the file `main_config.json` or passing your own configuration file by runnig:

```bash
python -m TP2.main --config_file=<path_to_config_file>
```

### Main config parameters

```json
{

	"min_real": float,
	"max_real": float,
	"population_size": int,
	"output_filename":string,
	"crossover": {
		"method": "single_point" | "multiple_point" | "uniform",
		"multiple_point_number": int
	},
	"selection": {
		"method": "roulette" | "tournament" | "boltzmann" | "truncation" | "elite" | "rank",
		"boltzmann_initial_temperature": float,
		"boltzmann_final_temperature": float,
		"boltzmann_decay_rate": float,
		"tournament_threshold": float,
		"truncate_size": int
	},
	"mutation": {
		"method": "normal" | "uniform" | "swap",
		"mutation_rate": float,
		"uniform_mutation_bound": float,
		"normal_mutation_standard_deviation": float
	},
	"cut_condition": {
		"method": "max_generations" | "fitness_value" | "fitness_variation" | "max_time",
		"max_generations": int,
		"min_fitness": float,
		"fitness_threshold": float,
		"fitness_required_generations_repeats": int,
		"max_time": float
	}
}
```

- `min_real`: Minimum value for a gene.
- `max_real`: Maximum value for a gene.
- `population_size`: Size of the population.
- `output_filename`: Name of the file to save the results.
- `crossover.method`: Crossover method.
- `crossover.multiple_point_number`: Number of points to use in the multiple point crossover.
- `selection.method`: Selection method.
- `selection.boltzmann_initial_temperature`: Initial temperature for the boltzmann selection method.
- `selection.boltzmann_final_temperature`: Final temperature for the boltzmann selection method.
- `selection.boltzmann_decay_rate`: Rate of temperature decay for the boltzmann selection method.
- `selection.tournament_threshold`: Threshold for the tournament selection method.
- `selection.truncate_size`: Size of the truncation selection method.
- `mutation.method`: Mutation method.
- `mutation.mutation_rate`: Rate of mutation.
- `mutation.uniform_mutation_bound`: Bound for the uniform mutation method.
- `mutation.normal_mutation_standard_deviation`: Standard deviation for the normal mutation method.
- `cut_condition.method`: Cut condition method.
- `cut_condition.max_generations`: Maximum number of generations for the max generations cut condition.
- `cut_condition.min_fitness`: Minimum fitness value for the fitness value cut condition.
- `cut_condition.fitness_threshold`: Threshold for the fitness variation cut condition.
- `cut_condition.fitness_required_generations_repeats`: Number of times to repeat the fitness variation cut condition.
- `cut_condition.max_time`: Maximum time for the max time cut condition.

You can also set to save the best individual of each generation:

```bash
python -m TP2.main --config_file=<path_to_config_file> --output_data=<path_to_output_file>
```

### Benchmarks

From the root of the project, run the following command:

```bash
pipenv shell

python -m TP2.benchmarks.${benchmark_module}
```

This will run the benchmarks with the configuration from `benchmarks_config.json`. If you want to make changes to the configuration, you can do so by editing the file `benchmarks_config.json` or passing your own configuration file by runnig:

```bash
python -m TP2.benchmarks.${benchmark_module} --config_file=<path_to_config_file>
```

### Benchmarks config parameters

```json
{

	"min_real": float,
	"max_real": float,
	"population_size": int,
	"crossover": {
		"method": "single_point" | "multiple_point" | "uniform",
		"multiple_point_number": int,
		"benchmark_output_filename": string
	},
	"selection": {
		"method": "roulette" | "tournament" | "boltzmann" | "truncation" | "elite" | "rank",
		"boltzmann_initial_temperature": float,
		"boltzmann_final_temperature": float,
		"boltzmann_decay_rate": float,
		"tournament_threshold": float,
		"truncate_size": int,
		"benchmark_output_filename": string

	},
	"mutation": {
		"method": "normal" | "uniform" | "swap",
		"mutation_rate": float,
		"uniform_mutation_bound": float,
		"normal_mutation_standard_deviation": float,
		"benchmark_output_filename": string
	},
	"cut_condition": {
		"method": "max_generations" | "fitness_value" | "fitness_variation" | "max_time",
		"max_generations": int,
		"min_fitness": float,
		"fitness_threshold": float,
		"fitness_required_generations_repeats": int,
		"max_time": float,
		"benchmark_output_filename": string
	},
	"initial_bounds": {
		"benchmark_output_filename": string,
		"bounds": [
			{
				"min_real": float,
				"max_real": float
			}
		]
	},
	"mutation_rates":{
		"benchmark_output_filename": string,
		"input":[ float ]
	}

}
```

- `min_real`: Minimum value for a gene.
- `max_real`: Maximum value for a gene.
- `population_size`: Size of the population.
- `output_filename`: Name of the file to save the results.
- `crossover.method`: Crossover method.
- `crossover.multiple_point_number`: Number of points to use in the multiple point crossover.
- `crossover.benchmark_output_filename`: Name of the file to save the results of the benchmark.
- `selection.method`: Selection method.
- `selection.boltzmann_initial_temperature`: Initial temperature for the boltzmann selection method.
- `selection.boltzmann_final_temperature`: Final temperature for the boltzmann selection method.
- `selection.boltzmann_decay_rate`: Rate of temperature decay for the boltzmann selection method.
- `selection.tournament_threshold`: Threshold for the tournament selection method.
- `selection.truncate_size`: Size of the truncation selection method.
- `selection.benchmark_output_filename`: Name of the file to save the results of the benchmark.
- `mutation.method`: Mutation method.
- `mutation.mutation_rate`: Rate of mutation.
- `mutation.uniform_mutation_bound`: Bound for the uniform mutation method.
- `mutation.normal_mutation_standard_deviation`: Standard deviation for the normal mutation method.
- `mutation.benchmark_output_filename`: Name of the file to save the results of the benchmark.
- `cut_condition.method`: Cut condition method.
- `cut_condition.max_generations`: Maximum number of generations for the max generations cut condition.
- `cut_condition.min_fitness`: Minimum fitness value for the fitness value cut condition.
- `cut_condition.fitness_threshold`: Threshold for the fitness variation cut condition.
- `cut_condition.fitness_required_generations_repeats`: Number of times to repeat the fitness variation cut condition.
- `cut_condition.max_time`: Maximum time for the max time cut condition.
- `cut_condition.benchmark_output_filename`: Name of the file to save the results of the benchmark.
- `initial_bounds.benchmark_output_filename`: Name of the file to save the results.
- `initial_bounds.bounds`: List of bounds for each gene.
- `mutation_rates.benchmark_output_filename`: Name of the file to save the results.
- `mutation_rates.input`: List of mutation rates.

You can also set to save the best individual of each generation:

```bash
python -m TP2.main --config_file=<path_to_config_file> --output_data=<path_to_output_file>
```
