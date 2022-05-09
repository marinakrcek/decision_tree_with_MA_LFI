from parameters import ParameterSet
import numpy as np


# CrossoverFunction = Callable[[ParameterSet, ParameterSet], ParameterSet]

def uniform_crossover(parent1: ParameterSet, parent2: ParameterSet):
    """
    Crossover operator takes two ParameterSet solutions as two parents for creating the offspring.
    This implements the uniform crossover, where for the child creating probability for taking a gene
    from either of the parents is equal.
    :param parent1: ParameterSet solution presenting first parent
    :param parent2: ParameterSet solution presenting second parent
    :return: ParameterSet solution (child of the two parents) created by uniform crossover
    """
    # uniform crossover, each bit is chosen from either parent with equal probability
    child = ParameterSet()
    child.x = np.random.choice([parent1.x, parent2.x])
    child.y = np.random.choice([parent1.y, parent2.y])
    child.delay = np.random.choice([parent1.delay, parent2.delay])
    child.pulse_width = int(np.random.choice([parent1.pulse_width, parent2.pulse_width]))
    child.intensity = np.random.choice([parent1.intensity, parent2.intensity])
    return child


def average_crossover(parent1: ParameterSet, parent2: ParameterSet):
    def get_avg(p1, p2, step):
        return min(p1,p2) + int((abs(p1-p2) / step / 2.0)) * step
    return ParameterSet(x=get_avg(parent1.x, parent2.x, ParameterSet.XSTEP),
                        y=get_avg(parent1.y, parent2.y, ParameterSet.YSTEP),
                        delay=get_avg(parent1.delay, parent2.delay, ParameterSet.DSTEP),
                        pulse_width=int(get_avg(parent1.pulse_width, parent2.pulse_width, ParameterSet.PWSTEP)),
                        intensity=get_avg(parent1.intensity, parent2.intensity, ParameterSet.ISTEP))
