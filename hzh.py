import numpy as np
import matplotlib.pyplot as plt


def objective_function(x):
    return np.sin(10 * np.pi * x) * x + np.cos(2 * np.pi * x) * x


class MyAlg:

    def __init__(self):
        pass

    def create_population(self, size, lower_bound, upper_bound):
        return np.random.uniform(lower_bound, upper_bound, size)

    # TODO: Implement your selection operator
    def selection(self):
        # your code here
        return

    # TODO: Implement your crossover operator
    def crossover(self, parent1, parent2):
        # your code here
        return

    # TODO: Implement your mutation operator
    def mutation(self):
        # your code here
        return

    # TODO: Implement main genetic algorithm process with crowding
    def run_genetic_algorithm(self, population_size, generations, lower_bound, upper_bound, mutation_rate):
        pass

    def plot_population(population, generation):
        x = np.linspace(lower_bound, upper_bound, 1000)
        y = objective_function(x)
        plt.plot(x, y, label="Objective function")

        plt.scatter(population, objective_function(population), color="red", label=f"Population (Gen {generation})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()


class MyAlgFitnessSharing(MyAlg):

    def __init__(self, sigma_share, alpha):
        super().__init__()
        self.sigma_share = sigma_share  # Sharing distance
        self.alpha = alpha  # Sharing exponent

    # TODO: Implement fitness sharing
    def fitness_sharing(self, population, fitness, sigma_share, alpha):
        '''
        Parameters:
        population (np.array): Array of population values
        fitness (np.array): Array of fitness values
        sigma_share (float): Sharing distance
        alpha (float): Sharing exponent

        Return:
        shared_fitness
        '''
        # your code here
        return

    # TODO: Implement main genetic algorithm process
    def run_genetic_algorithm(self, population_size, generations, lower_bound, upper_bound, mutation_rate):
        '''
        Return:
        final_population (np.array): Array of final population values
        '''
        # Initialize population
        population = self.create_population(population_size, lower_bound, upper_bound)

        # Run GA
        for generation in range(generations):
            fitness = objective_function(population)
            shared_fitness = self.fitness_sharing(population, fitness, self.sigma_share, self.alpha)

            new_population = []
            for _ in range(population_size):
                # TODO: Implement evolutionary process
                pass

        return


class MyAlgCrowding(MyAlg):

    def __init__(self):
        pass

    # TODO: Implement main genetic algorithm process with crowding
    def run_genetic_algorithm(self, population_size, generations, lower_bound, upper_bound, mutation_rate):
        population = self.create_population(population_size, lower_bound, upper_bound)

        for generation in range(generations):
            fitness = objective_function(population)
            # your code here

        return population


# ===============================================================


# GA parameters
population_size = 100
generations = 100
lower_bound = 0
upper_bound = 1
mutation_rate = 0.1

sigma_share = 0.1
alpha = 1


if __name__ == '__main__':

    # my_alg = MyAlgFitnessSharing(sigma_share, alpha)
    my_alg = MyAlgCrowding()

    population = my_alg.run_genetic_algorithm(population_size, generations, lower_bound, upper_bound, mutation_rate)

    # Plot results
    x = np.linspace(lower_bound, upper_bound, 1000)
    y = objective_function(x)
    plt.plot(x, y, label="Objective function")

    plt.scatter(population, objective_function(population), color="red", label="Final population")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
