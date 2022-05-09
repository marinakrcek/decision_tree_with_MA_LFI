from typing import List, Callable, Tuple
import ga_log
from parameters import *
from sort_algorithms import xy_snake_sort
from initializations import random_initialization, TaguchiInitialization
from crossovers import uniform_crossover
from selections import roulette_wheel
from mutations import uniform_mutation
from local_search import hooke_jeeves
from fitness import fitness, percentage_fitness
from stop_conditions import stop_cond_iterations, stop_cond_fitness
from helper import TabuList
import operator

# type hint definition for GA function parameters
StoppingFunction = Callable[[List[ParameterSet], int], bool]
InitializationFunction = Callable[[int], List[ParameterSet]]
SortingFunction = Callable[[List[ParameterSet]], List[ParameterSet]]
SelectionFunction = Callable[[List[ParameterSet], int], Tuple[List[ParameterSet], List[ParameterSet]]]
CrossoverFunction = Callable[[ParameterSet, ParameterSet], ParameterSet]
MutationFunction = Callable[[ParameterSet, float], ParameterSet]
LocalSearch = Callable[[List[ParameterSet], Callable, dict], List[ParameterSet]]


class GeneticAlgorithm:
    """
    Class that represents the Genetic Algorithm.
    """
    TESTED = TabuList()

    def __init__(self, apply_on_bench_function, carto_set_iteration,
                 pop_size=30, mutation_probability=0.05, elite_size=2,
                 nb_measurements=5,
                 max_iterations=50,
                 parameter_info_file: str = None,
                 initialization_function: InitializationFunction = random_initialization,
                 stop_condition=stop_cond_iterations,
                 sort_function: SortingFunction = None,
                 selection: SelectionFunction = roulette_wheel,
                 crossover: CrossoverFunction = uniform_crossover,
                 mutation: MutationFunction = uniform_mutation,
                 local_search: LocalSearch = hooke_jeeves,
                 fitness_func=percentage_fitness
                 ):
        set_parameter_limits_from_ini_file(parameter_info_file)
        self.apply_on_bench = apply_on_bench_function
        self.set_carto_iteration = carto_set_iteration
        self.pop_size = pop_size
        self.mutation_prob = mutation_probability
        self.elite_size = elite_size
        self.nb_measurements = nb_measurements
        self.max_iterations = max_iterations
        self.initialization = initialization_function
        self.stop_condition = stop_condition
        self.sort = sort_function
        self.selection = selection
        self.crossover = crossover
        self.mutate = mutation
        self.local_search = local_search
        self.fitness = fitness_func

    def generate_population(self):
        """
        Function that generates a population of solutions (parameter sets) with random values with the initializations
        function set by the user.
        :return: list of solutions representing population
        """
        pop = self.initialization(self.pop_size)
        if self.pop_size != len(pop):
            self.pop_size = len(pop)
            print("Because of initialization technique, the population size is updated to size", self.pop_size)
        return pop

    def evaluate_parameter_set(self, parameter_set):
        if parameter_set.fitness is not None:
            return parameter_set.fitness, parameter_set.fault_class

        fitness, fault_class = self.TESTED.get(parameter_set, (None, None))
        if fitness is not None:
            return fitness, fault_class

        fault_classes = list()
        for n in range(0, self.nb_measurements):
            ## Apply bench parameters & LFI
            fault_class = self.apply_on_bench(parameter_set)
            fault_classes.append(fault_class)
        fit, fault = self.fitness(fault_classes)
        self.TESTED.add(parameter_set, (fit, fault))
        return fit, fault

    def evaluate_pop(self, population: List[ParameterSet]):
        """
        Function that evaluates the complete population of solutions.
        Function first sorts the population, then evaluates the solutions.
        Solution fitness is stored in the ParameterSet class, so if the solution was once evaluated, it will not be
        evaluated again (except if a change happens in the parameter values).
        :param population: list of parameter sets, population
        :return: population
        """
        if self.sort is not None:
            population = self.sort(population)
        for parameter_set in population:
            parameter_set.fitness, parameter_set.fault_class = self.evaluate_parameter_set(parameter_set)
        return population

    def reproduce(self, population):
        """
        Function that reproduces and creates a new generation of the population.
        :param population: list of ParameterSet solutions, population
        :return: new population after reproduction
        """
        newpop = []
        parents1, parents2 = self.selection(population, self.elite_size)
        for par1, par2 in zip(parents1, parents2):
            child = self.crossover(par1, par2)
            self.mutate(child, self.mutation_prob)
            newpop += [child]
            child.created = 'e'
        newpop += sorted(population, key=operator.attrgetter('fitness'), reverse=True)[:self.elite_size]
        newpop = set(newpop)
        while len(newpop) < self.pop_size:
            ps = ParameterSet()
            ps.created = 'r'
            newpop.add(ps)
        return list(newpop)

    def one_iteration(self, population):
        """
        Function executes one iteration of the GA algorithm.
        Evaluates the current population, then executes the reproduction step and finishes with a local search.
        :param population: list of solutions
        :return: list of solutions, population
        """
        population = self.reproduce(population)
        population = self.evaluate_pop(population)
        if self.local_search is not None:
            population = self.local_search(population, self.evaluate_parameter_set, self.TESTED)
        return population

    def run(self, log_file_name='logfile.pkl'):
        """
        Function that runs the genetic algorithm until the stop condition is satisfied.
        It creates the population and then iterates the algorithm. It prints out the iteration number.
        :return: final population after the genetic algorithm
        """
        log = ga_log.Log(log_file_name, str(self), ParameterSet.get_param_limits(as_dict=True))
        population = self.generate_population()
        population = self.evaluate_pop(population)
        iteration = 0

        ## Main loop
        while not self.stop_condition(population, iteration, self.max_iterations):
            print("Iteration: ", iteration + 1)
            log.log_generation(iteration, population)
            population = self.one_iteration(population)
            iteration += 1
            self.set_carto_iteration(iteration)

        self.TESTED.clear()
        return population

    def __str__(self):
        return f"{{\"pop_size\" : {self.pop_size}, " \
               f"\"mutation_prob\" : {self.mutation_prob}, " \
               f"\"elite_size\" : {self.elite_size}, " \
               f"\"nb_measurements\" : {self.nb_measurements}, " \
               f"\"max_iterations\" : {self.max_iterations}, " \
               f"\"initialization\" : \"{self.initialization.__name__}\", " \
               f"\"stop_condition\" : \"{self.stop_condition.__name__}\", " \
               f"\"sort\" : \"{str(self.sort.__name__) if self.sort else None}\", " \
               f"\"selection\" : \"{self.selection.__name__}\", " \
               f"\"crossover\" : \"{self.crossover.__name__}\", " \
               f"\"mutate\" : \"{self.mutate.__name__}\", " \
               f"\"local_search\" : \"{self.local_search.__name__ if self.local_search else None}\", " \
               f"\"fitness\" : \"{self.fitness.__name__}\"}}"

    def __repr__(self):
        return self.__str__()

    def info_as_dict(self):
        return {"pop_size": self.pop_size,
                "mutation_prob": self.mutation_prob,
                "elite_size": self.elite_size,
                "nb_measurements": self.nb_measurements,
                "max_iterations": self.max_iterations,
                "initialization": self.initialization.__name__,
                "stop_condition": self.stop_condition.__name__,
                "sort": self.sort.__name__ if self.sort else None,
                "selection": self.selection.__name__,
                "crossover": self.crossover.__name__,
                "mutate": self.mutate.__name__,
                "local_search": self.local_search.__name__ if self.local_search else None,
                "fitness": self.fitness.__name__}


class TaguchiGA(GeneticAlgorithm):
    # runsize is pop size (might change because adjusting to the levels)
    def __init__(self, apply_on_bench_function, carto_set_iteration,
                 pop_size=36, mutation_probability=0.05, elite_size=2,
                 nb_measurements=5,
                 max_iterations=50,
                 parameter_info_file: str = None,
                 stop_condition=stop_cond_iterations,
                 sort_function: SortingFunction = None,
                 selection: SelectionFunction = roulette_wheel,
                 crossover: CrossoverFunction = uniform_crossover,
                 mutation: MutationFunction = uniform_mutation,
                 local_search: LocalSearch = hooke_jeeves,
                 fitness_func=percentage_fitness,
                 factor_levels=2, strength=2, number_of_factors=ParameterSet.get_parameter_number()):
        self.ta = TaguchiInitialization(number_of_factors=number_of_factors, run_size=pop_size,
                                        factor_levels=factor_levels, strength=strength)
        super().__init__(apply_on_bench_function,
                         carto_set_iteration=carto_set_iteration,
                         pop_size=self.ta.run_size,
                         mutation_probability=mutation_probability,
                         elite_size=elite_size,
                         nb_measurements=nb_measurements,
                         max_iterations=max_iterations,
                         parameter_info_file=parameter_info_file,
                         initialization_function=self.ta.taguchi_initialization,
                         stop_condition=stop_condition,
                         sort_function=sort_function,
                         selection=selection, crossover=crossover, mutation=mutation, local_search=local_search,
                         fitness_func=fitness_func)
