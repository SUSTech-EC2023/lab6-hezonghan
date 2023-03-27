
import numpy as np
import matplotlib.pyplot as plt

from hzh_v2.objective import objective_function
from hzh_v2.common_operations import create_population, selection, crossover, mutation


def plot_population(lower_bound, upper_bound, population, population_scatter_label):
    x = np.linspace(lower_bound, upper_bound, 1000)
    y = objective_function(x)
    plt.plot(x, y, label="Objective function")

    plt.scatter(population, objective_function(population), color="red", label=population_scatter_label)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


class MyAlgV2:

    def __init__(self):
        pass

    def adjust_fitness(self, population, original_fitness):
        raise Exception('Not implemented in base class.')
        # return adjusted_fitness

    def run_genetic_algorithm(self, population_size, generations, lower_bound, upper_bound, mutation_rate):

        # Initialize population
        population = create_population(population_size, lower_bound, upper_bound)

        # Run GA
        for generation in range(generations):
            original_fitness = objective_function(population)
            adjusted_fitness = self.adjust_fitness(population, original_fitness)

            new_population = []
            for _ in range(population_size):
                # TODO: Implement evolutionary process
                pass

        plot_population(lower_bound, upper_bound, population, 'Final population')

        return population


class MyAlgV2FitnessSharing(MyAlgV2):

    def __init__(self, sigma_share, alpha):
        super().__init__()
        self.sigma_share = sigma_share  # Sharing distance
        self.alpha = alpha  # Sharing exponent

    def adjust_fitness(self, population, original_fitness):
        # TODO: Implement fitness sharing
        adjusted_fitness = None
        return adjusted_fitness


class MyAlgV2Crowding(MyAlgV2):

    def __init__(self):
        super().__init__()


# ========================================================


# GA parameters

population_size = 100
generations = 100
lower_bound = 0
upper_bound = 1
mutation_rate = 0.1

sigma_share = 0.1
alpha = 1


if __name__ == '__main__':

    # my_alg = MyAlgV2FitnessSharing(sigma_share, alpha)
    my_alg = MyAlgV2Crowding()

    population = my_alg.run_genetic_algorithm(population_size, generations, lower_bound, upper_bound, mutation_rate)
