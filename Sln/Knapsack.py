import numpy as np
import random
import matplotlib.pyplot as plt
import cProfile
from pstats import Stats, SortKey
import timeit

np.random.seed(42)


class KnapSackProblem:
    def __init__(self, size):
        order = np.arange(size)
        weights = 2 ** np.random.randn(size)
        values = 2 ** np.random.randn(size)
        self.definition = np.vstack((order, weights, values))
        self.capacity = 0.25 * np.sum(values)
        self.heuristicValue = self.calcHeuristic

    def calcHeuristic(self):
        heuristic = -self.definition[2] / self.definition[1]
        sorted = self.definition[:, heuristic.argsort()]
        index = (np.cumsum(sorted[1, :]) <= self.capacity).argmin()
        return np.sum(sorted[2, :index])


class Individual:
    def __init__(
        self, ProblemToSolve: KnapSackProblem, solution: np.array = None, alpha=0.05
    ) -> None:
        # initialise with a given Knapsack problem
        if solution is None:
            self.maxcapacity = ProblemToSolve.capacity
            self.alpha = 0.05
            self.solution = np.random.permutation(ProblemToSolve.definition.T).T
            self.fitness, self.included = self.CalculateFitness()
            # initialise with a given array
        else:
            self.solution = solution
            self.alpha = alpha
            self.maxcapacity = ProblemToSolve.capacity
            self.fitness, self.included = self.CalculateFitness()

    def CalculateFitness(self) -> int:
        index = (np.cumsum(self.solution[1, :]) <= self.maxcapacity).argmin()
        return np.sum(self.solution[2, :index]), self.solution[0, :index]

    def mutate(self) -> None:
        # if random.uniform(0,1) > self.al   pha:
        indices = np.random.randint(len(self.solution), size=2)
        self.solution[:, indices[0]], self.solution[:, indices[1]] = (
            self.solution[:, indices[1]],
            self.solution[:, indices[0]].copy(),
        )
        return self.CalculateFitness()


def recombination(
    parent1: Individual, parent2: Individual, problem: KnapSackProblem
) -> Individual:
    offspring = np.zeros_like(parent1.solution)
    # we want to create a mix between 2 individuals.
    # We will ensure that the offspring has the elements that both have in their knapsack
    # we will insert the symmetric difference of both with a 0.5 probability
    # We will permute this entire set, then we will permute the remainder and add it to the list.
    intersection = np.intersect1d(parent1.included, parent2.included)
    sym_diff = np.setdiff1d(parent1.included, parent2.included)
    intersection_values = np.in1d(parent1.solution[0], intersection)
    sym_diff_values = np.in1d(parent1.solution[0], sym_diff)

    # copy intersection
    common_part = parent1.solution[:, intersection_values]
    # insert symmetric difference

    for i in range(len(sym_diff)):
        if random.uniform(0, 1) > 0.5:
            common_part = np.vstack(
                (common_part.T, parent1.solution[:, sym_diff_values][:, i])
            ).T
    # Apply permutation on the current set
    common_part = np.random.permutation(common_part.T).T

    # Find remaining, permute and add to offspring
    remaining = np.setdiff1d(parent1.solution[0, :], common_part[0, :])
    remaining_values = np.in1d(parent1.solution[0], remaining)

    remaing_part = parent1.solution[:, remaining_values]
    remaing_part = np.random.permutation(remaing_part.T).T
    offspring = np.vstack((common_part.T, remaing_part.T)).T
    # adjust mutation probability
    beta = (2 * random.uniform(0, 1)) - 0.5
    alpha = parent1.alpha + beta * (parent2.alpha - parent1.alpha)

    return Individual(ProblemToSolve=problem, solution=offspring, alpha=alpha)


def local_search(candidate: Individual, problem: KnapSackProblem) -> Individual:
    initial = np.copy(candidate.solution)
    neighbourhood = np.empty(initial.shape[1], dtype=object)
    for i in range(initial.shape[1]):
        val = initial[:, i].copy()
        candidate.solution = initial.copy()
        new_candidate = np.delete(candidate.solution, i, axis=1)
        new_candidate = np.insert(new_candidate, 0, val, axis=1)
        neighbourhood[i] = Individual(ProblemToSolve=problem, solution=new_candidate)
    highest = np.argmax([ind.fitness for ind in neighbourhood])
    return neighbourhood[highest]


def two_change(candidate: Individual, problem: KnapSackProblem) -> Individual:
    best = candidate
    initial = np.copy(candidate.solution)
    # neighbourhood = np.empty(initial.shape[1], dtype=object)
    for i in range(initial.shape[1]):
        for j in range(i + 1, initial.shape[1]):
            candidate.solution = initial.copy()
            candidate.solution[:, [i, j]] = candidate.solution[:, [j, i]]
            new = Individual(solution=candidate.solution, ProblemToSolve=problem)
            best = new if new.fitness > best.fitness else best
    return best


def selection(population: np.array, k: int) -> Individual:
    selected = np.random.choice(population, k)
    highest = np.argmax([ind.fitness for ind in selected])
    return selected[highest]


# Lambda + mu elimination
def elimination(population: np.array, offspring: np.array, lmbda: int):
    combined = np.concatenate((population, offspring))
    combined = combined[np.argsort([-ind.fitness for ind in combined])]
    return combined[:lmbda]


def EvolutionaryAlgorithm(problem: KnapSackProblem) -> None:

    population_size = 20
    mu = 20
    iterations = 1000
    mean_fitnesses = np.empty(iterations)
    max_fitnesses = np.empty(iterations)

    population = np.empty(shape=population_size, dtype=object)
    for i in range(population_size):
        population[i] = Individual(problem)
    offspring = np.empty_like(population)
    converged_at = 0

    for i in range(iterations):
        # recombination
        for j in range(mu):
            parent1 = selection(population, k=2)
            parent2 = selection(population, k=2)
            offspring[j] = recombination(parent1, parent2, problem)
            # mutate
            offspring[j].mutate()
            offspring[j] = two_change(offspring[j], problem)
        [ind.mutate() for ind in population]
        population = elimination(population, offspring, lmbda=10)
        mean_fitnesses[i] = np.mean([ind.fitness for ind in population])
        max_fitnesses[i] = np.amax([ind.fitness for ind in population])
        converged_at += 1
        if (i > 10) & (max_fitnesses[i - 10] == max_fitnesses[i]):
            break
    print(
        f"The algorithm converged at: {converged_at} with a maximum fitness of {max_fitnesses[i]}"
    )
    plt.figure()
    plt.plot(max_fitnesses[:converged_at], label="max fitness")
    plt.plot(mean_fitnesses[:converged_at], label="mean fitness")
    plt.xlabel("Iterations")
    plt.ylabel("Max fitness")
    plt.legend()
    plt.show()


def main():
    start_time = timeit.default_timer()
    ToSolve = KnapSackProblem(25)
    print(f"The heuristic value is {ToSolve.heuristicValue()}")
    EvolutionaryAlgorithm(ToSolve)
    elapsed = timeit.default_timer() - start_time
    print(f"The algorithm took {elapsed} seconds to complete.")


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()

    with open("profiling_stats.txt", "w") as stream:
        stats = Stats(pr, stream=stream)
        stats.sort_stats("time")
        stats.print_stats()
