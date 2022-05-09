import collections
import json
import numpy as np
import fitness
from helper import converter
from bench_connection import dummy_cartography
from parameters import ParameterSet

NB_DIFF_TESTS = 1000  # how many points (parameter sets) should be tested

########## FAULT INJECTION PARAMETERS ############
NB_MEASUREMENTS = 5  # with the same parameter set (same spot)
parameter_info_file = 'parameter_info.ini'


if __name__ == "__main__":
    fitness.set_fitness_values('fitness.info')

    # run_times defines how many times you want to run the same GA
    run_times = 1

    # construct the cartography class
    carto = dummy_cartography()

    for i in range(run_times):
        all_points = set()
        while len(all_points) < NB_DIFF_TESTS:
            ps = ParameterSet()
            if ps in all_points:
                print('skipping', ps)
                continue
            ## evaluate the parameter set
            fault_classes = list()
            for n in range(0, NB_MEASUREMENTS):
                ## Apply bench parameters & LFI
                fault_classes.append(carto.apply_bench_parameter(ps))
            ps.fitness, ps.fault_class = fitness.percentage_fitness(fault_classes)
            all_points.add(ps)
            print(ps)



