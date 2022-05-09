from parameters import *
from typing import List
import operator
import itertools
import sklearn.metrics.pairwise as skpw
import copy


# type hint definition for GA function parameters
# SortingFunction = Callable[[List[ParameterSet]], List[ParameterSet]]


def basic_sort(population: List[ParameterSet]) -> List[ParameterSet]:
    return sorted(population, key=operator.attrgetter('x', 'y', 'intensity'))


def xy_sort(population: List[ParameterSet]) -> List[ParameterSet]:
    return sorted(population, key=operator.attrgetter('x', 'y'))


def xy_snake_sort(population: List[ParameterSet]) -> List[ParameterSet]:
    population.sort(key=operator.attrgetter('x'))
    pop = [sorted(list(g), key=operator.attrgetter('y'), reverse=((i + 1) % 2 == 0)) for i, (k, g) in
           enumerate(itertools.groupby(population, key=operator.attrgetter('x')))]
    return list(itertools.chain(*pop))


def manhattan_distances(current_ps, unvisited_ps):
    return skpw.manhattan_distances([[current_ps.x, current_ps.y]], [[ps.x, ps.y] for ps in unvisited_ps])[0]


def euclidean_distances(current_ps, unvisited_ps):
    return skpw.euclidean_distances([[current_ps.x, current_ps.y]], [[ps.x, ps.y] for ps in unvisited_ps])[0]


# finding shortest path with greedy algorithm using a given distance function
def greedy(population: List[ParameterSet], distance_func=manhattan_distances) -> List[ParameterSet]:
    points = xy_sort(population)
    source = points[0]
    unvisited_points = points[1:]
    route = [source]
    current_point = source

    while unvisited_points:
        # get index of the closest point from unvisited_points
        i_closest_point = np.argmin(distance_func(current_point, unvisited_points))
        current_point = copy.deepcopy(unvisited_points[i_closest_point])
        route.append(current_point)
        unvisited_points.pop(i_closest_point)
    return route


def greedy_manhattan(population: List[ParameterSet]) -> List[ParameterSet]:
    return greedy(population, manhattan_distances)


def greedy_euclidean(population: List[ParameterSet]) -> List[ParameterSet]:
    return greedy(population, euclidean_distances)

