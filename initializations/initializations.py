from parameters import *
import oapackage  # Orthogonal Array Package
import lhsmdu
import pyDOE2

# type hint definition for GA initializations function for generating population
# function takes integer which is the size of the population and returns a list
# of parameter sets which are the solutions for the GA
# InitializationFunction = Callable[[int], List[ParameterSet]]


def random_initialization(pop_size):
    """
    Function that creates a population of solutions (parameter sets) with random values.
    :param pop_size: size of the population
    :return: list of solutions representing population for GA
    """
    pop = set()
    while len(pop) < pop_size:
        pop.add(ParameterSet())
    return list(pop)


class TaguchiInitialization:
    # factor_levels can be int or list of ints (if list should be the size of number of factors)
    # number of factors is number of parameters in parameterSet class
    def __init__(self, number_of_factors=ParameterSet.get_parameter_number(), run_size=16, factor_levels=2, strength=2):
        self.number_of_factors = number_of_factors
        if isinstance(factor_levels, int):
            self.factor_levels = [factor_levels] * self.number_of_factors
        elif isinstance(factor_levels, list):
            len_levels = len(factor_levels)
            if len_levels < self.number_of_factors:
                self.factor_levels = factor_levels + [min(factor_levels)] * (self.number_of_factors - len_levels)
            elif len_levels > self.number_of_factors:
                self.factor_levels = factor_levels[:self.number_of_factors]
            else:
                self.factor_levels = factor_levels
        # lcm = int(np.lcm.reduce(self.factor_levels))
        # self.run_size = closest_multiple(run_size, lcm)
        self.run_size = run_size
        self.strength = strength

    def taguchi_initialization(self, pop_size):
        """
        Creates population using Taguchi method.
        :param pop_size: size of the population
        :return: list of solutions representing population for GA
        """

        arrayclass = oapackage.arraydata_t(self.factor_levels, self.run_size, self.strength, self.number_of_factors)
        arrays = [arrayclass.create_root()]
        for extension_column in range(2, self.number_of_factors):
            if not arrays:
                return random_initialization(self.run_size)
            arrays = oapackage.extend_arraylist([arrays[0]], arrayclass)  # TODO: we can take a random array from the list

        if not arrays:
            return random_initialization(self.run_size)

        oa_pop = np.array(arrays[0])  # TODO: we can change to take the random array from the list
        return oa_to_population(oa_pop)


def taguchi_from_example(example_index: int):
    def taguchi_example_to_population(pop_size: int):
        array = np.array(oapackage.exampleArray(example_index))
        return oa_to_population(array)
    return taguchi_example_to_population


def taguchi_from_file(file_path: str):
    def taguchi_from_file_to_population(pop_size: int):
        array = np.genfromtxt(file_path, delimiter=',')
        return oa_to_population(array)
    return taguchi_from_file_to_population


def index_array_to_population(array):
    return lambda _: oa_to_population(array)


def oa_to_population(array):
    number_of_factors = ParameterSet().get_parameter_number()
    array = array[:, :number_of_factors]
    if array.shape[1] < number_of_factors:
        raise ValueError("Wrong number of parameters in the orthogonal array example used.")
    factor_levels = list(map(lambda col: len(set(col)), array.T))
    return [ParameterSet.get_parameterset_from_indexes(oa, factor_levels) for oa in array]


def latin_hypercube_sampling_mdu(pop_size=15):
    number_of_factors = ParameterSet().get_parameter_number()
    array = np.array(lhsmdu.sample(number_of_factors, pop_size)).T  # Latin Hypercube Sampling with multi-dimensional uniformity
    return [ParameterSet.get_parameterset_from_uniform(ps) for ps in array]


def latin_hypercube_sampling_pydoe2(criterion=None):
    def lhs_pydoe2(pop_size=15):
        number_of_factors = ParameterSet().get_parameter_number()
        array = pyDOE2.lhs(number_of_factors, samples=pop_size)
        return [ParameterSet.get_parameterset_from_uniform(ps) for ps in array]
    return lhs_pydoe2
