
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

            if (generation + 1) % 10 == 0:
                plot_population(lower_bound, upper_bound, population, f"Population (Gen {generation})")

        plot_population(lower_bound, upper_bound, population, 'Final population')

        return population


class MyAlgV2FitnessSharing(MyAlgV2):

    def __init__(self, sigma_share, alpha, distance_p, raw_fitness_scaling_beta):
        super().__init__()

        self.distance_p = distance_p
        self.sigma_share = sigma_share  # Sharing distance
        self.alpha = alpha  # Sharing exponent
        self.raw_fitness_scaling_beta = raw_fitness_scaling_beta

    def adjust_fitness(self, population: np.ndarray, original_fitness: np.ndarray):
        # TODO: Implement fitness sharing

        population_size = original_fitness.shape[0]

        # distance_matrix = np.matmul(
        #     population.reshape([population_size, 1]),
        #     population.reshape([1, population_size]),
        # )
        denominator = np.zeros(shape=(population_size,))
        for i1 in range(0, population_size):
            for i2 in range(i1+1, population_size):
                distance = np.power(np.sum(np.power(population[i1] - population[i2], self.distance_p)), 1.0/self.distance_p)
                if distance < self.sigma_share:
                    sh = 1 - np.power(distance / sigma_share, self.alpha)
                    denominator[i1] += sh
                    denominator[i2] += sh

        adjusted_fitness = np.power(original_fitness, self.raw_fitness_scaling_beta) / denominator

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
distance_p = 2  # Lp distance
raw_fitness_scaling_beta = 1


if __name__ == '__main__':

    # my_alg = MyAlgV2FitnessSharing(sigma_share, alpha, distance_p, raw_fitness_scaling_beta)
    my_alg = MyAlgV2Crowding()

    population = my_alg.run_genetic_algorithm(population_size, generations, lower_bound, upper_bound, mutation_rate)
