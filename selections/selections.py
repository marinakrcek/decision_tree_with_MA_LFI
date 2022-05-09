from parameters import ParameterSet
from typing import List
import numpy as np
import operator

# SelectionFunction = Callable[[List[ParameterSet], int], Tuple[List[ParameterSet], List[ParameterSet]]]


def roulette_wheel(population: List[ParameterSet], elite_size=2):
    """
    Function performs roulette wheel selection of parents from the population and returns chosen parents.
    Returns a list of parents, where the sizes are equal to the size of population without the elite size.
    Population has to be evaluated before calling the selection.
    :param elite_size:
    :param population: list of solutions, population
    :return: two lists representing solutions chosen as parents for the following crossover operation
    """
    fits = np.array([float(s.fitness) for s in population])
    # roulette does not work for negative values, so we shift all values with the minimum fitness
    if np.min(fits) < 0:
        fits -= np.min(fits)
    fits /= np.sum(fits)

    pop_size = len(fits)
    parents1 = np.random.choice(population, size=pop_size - elite_size, p=fits)
    parents2 = np.random.choice(population, size=pop_size - elite_size, p=fits)
    return parents1, parents2


def ktournament(k=4):
    def tournament(population: List[ParameterSet], elite_size=2):
        pop_size = len(population)
        if k < 2 or k >= pop_size:
            raise ValueError("Tournament size should be smaller than the population size, but larger than 2, because "
                             "2 candidates are taken from each tournament.")
        parents1 = []
        parents2 = []
        for i in range(0, pop_size-elite_size):
            candidates = np.random.choice(population, size=k)
            candidates = sorted(candidates, key=operator.attrgetter('fitness'))
            parents1.append(candidates[-1])
            parents2.append(candidates[-2])
        return parents1, parents2
    return tournament
