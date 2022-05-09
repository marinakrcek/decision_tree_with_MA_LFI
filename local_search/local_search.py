import numpy as np
import copy
from fitness import FAULT
from helper import TabuList
from parameters import ParameterSet


# LocalSearch = Callable[[List[ParameterSet]], List[ParameterSet]]
# EvaluationFunction = Callable[[float, float, float, int, int], Tuple[float, str]]
# EvaluatePopulation = Callable[[List[ParameterSet]], None]

local_search_visited = set()


def get_interesting_points(population):
    max_fitness = max([f.fitness for f in FAULT.values()])
    indexes = np.where(np.array([s.fitness for s in population]) > 0.85 * max_fitness)[0]
    indexes = [i for i in indexes if population[i] not in local_search_visited]

    max_solutions_for_local_search = 3
    if len(indexes) > max_solutions_for_local_search:
        indexes = np.random.choice(indexes, max_solutions_for_local_search, replace=False)
    local_search_visited.update(set(np.array(population)[indexes]))
    return indexes


def hooke_jeeves(population, evaluation_function, TESTED: TabuList):
    """
    Function that performs Hooke-Jeeves algorithm as the local search.
    Solutions that were better after local search replace the starting point of the local search.
    First the algorithm evaluates the population (solutions that are not evaluated already)
    because of reproduction step. This way the solutions are also sorted before starting with the local search.
    :param evaluate_population:
    :param evaluation_function:
    :param population: list of ParameterSet solutions
    :return: population after local search
    """

    def explore(x, Dx):
        """
        Exploration step of the Hooke-Jeeves algorithm.

        :param x: starting point, ParameterSet solution
        :param Dx: local step for all the parameters (dimensions) of the ParameterSet
        :return: local point with better fitness, or the starting point if none were better after exploration
        """
        p = copy.deepcopy(x)
        params = vars(p)
        p.created = 'l'
        param_names = p.get_param_names()
        for i, key in np.random.permutation(list(zip(range(len(param_names)), param_names))):
            i = int(i)
            value = params[key]
            orig_fit, orig_fault = p.fitness, p.fault_class
            p.update(key, value + Dx[i])
            fitness, fault_class = TESTED.get(p, (None, None))
            if fitness is None:
                fitness, fault_class = evaluation_function(copy.deepcopy(p))
            if fitness < orig_fit:
                p.update(key, value - Dx[i])
                fitness, fault_class = TESTED.get(p, (None, None))
                if fitness is None:
                    fitness, fault_class = evaluation_function(copy.deepcopy(p))
                if fitness < orig_fit:
                    p.update(key, value)
                    p.update_fitness(orig_fit, orig_fault)
                    continue
            p.update_fitness(fitness, fault_class)
        return p

    indexes = get_interesting_points(population)
    if len(indexes) < 1:
        # no interesting points
        return population
    indexes_sorted_pop = np.argsort(np.array([ps.fitness for ps in population]))  # from worst to best solutions
    worst_i = 0
    steps = np.array([r[2] for r in ParameterSet.get_param_limits()])
    for i in indexes:
        s = population[i]
        Dx = 2 * steps
        xp = copy.deepcopy(s)
        xb = copy.deepcopy(s)
        while np.all(Dx >= steps):
            xn = explore(xp, Dx)
            if xn.fitness > xb.fitness:  # is xn fitness better than fitness of xb
                xp = 2 * xn - xb  # if yes, we move in that direction, and explore from new xp
                xp.created = 'l'
                fitness, fault_class = TESTED.get(xp, (None, None))
                if fitness is None:
                    fitness, fault_class = evaluation_function(xp)
                xp.update_fitness(fitness, fault_class)
                xb = copy.deepcopy(xn)
            else:
                Dx = Dx / 2.0  # if not, decrease the exploration delta, and reset starting point xp
                xp = copy.deepcopy(xb)
        if xb.fitness > population[i].fitness:
            while indexes_sorted_pop[worst_i] in indexes and indexes_sorted_pop[worst_i] != i:
                worst_i += 1
            population[indexes_sorted_pop[worst_i]] = copy.deepcopy(xb)  # xb is the resulting point in the end
            worst_i += 1
    return population
