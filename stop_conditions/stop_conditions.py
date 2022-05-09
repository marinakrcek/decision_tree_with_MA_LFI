

def stop_cond_iterations(population, iteration, MAX_ITERATIONS):
    """
    This function should define stopping condition of the GA algorithm.
    The interface has to correspond to what GA is sending (if changed).

    :param iteration: current iteration of the GA
    :return: bool value, True if the stopping condition is satisfied, False otherwise
    """
    if iteration >= MAX_ITERATIONS:
        return True
    return False


def stop_cond_fitness(population, iteration, MAX_ITERATIONS):
    """
    This function should define stopping condition of the GA algorithm.
    The interface has to correspond to what GA is sending (if changed).

    :param population: list of solutions (parameter sets) with fitness values as well
    :param iteration: current iteration of the GA
    :return: bool value, True if the stopping condition is satisfied, False otherwise
    """
    if iteration >= MAX_ITERATIONS:
        return True
    fitness_vals = [ps.fitness for ps in population]
    nb_greater = sum(map(lambda x: x > 9.5, fitness_vals))
    if nb_greater > 0.9*len(fitness_vals):
        return True
    return False