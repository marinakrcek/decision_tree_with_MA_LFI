import collections
import operator


class fault_class:
    def __init__(self, id, name, fitness):
        self.id = id
        self.name = name
        self.fitness = fitness

    def __str__(self):
        return f"{self.name}: id={self.id}, fitness={self.fitness}"

    def __repr__(self):
        return self.__str__()

    def as_dict(self):
        return {'fault': self.name, 'id': self.id, 'fitness': self.fitness}


FAULT = {0: fault_class(0, 'pass', 2), 1: fault_class(1, 'fail', 10),
         2: fault_class(2, 'mute', 5), 'mix': fault_class('mix', 'changing', -1)}


def set_fitness_values(file_name):
    with open(file_name, 'r') as f:
        all_lines = f.readlines()
    FAULT.clear()
    for line in all_lines:
        fault_info = line.strip()
        if fault_info.startswith('#'):
            continue
        (id, name, fitness) = fault_info.split()
        FAULT[int(id)] = fault_class(int(id), name, float(fitness))
    FAULT['mix'] = fault_class('mix', 'changing', -1)


def name_to_fault_id(fault_class_name: str):
    for val in FAULT.values():
        if val.name == fault_class_name:
            return val.id


def id_to_fault_name(fault_class_id):  # int or str
    return FAULT[fault_class_id].name


def maldini_fitness(fault_classes):
    """
    From the fault class returns a corresponding value that presents the fitness for the parameter set for GA
    evaluations.
    This can be changed and new functions can be developed to better present the fitness of parameter set so that the
    GA can perform better.
    :param fault_classes:
    :return: fitness value of the given fault class (numeric value) for GA evaluations
    """
    if len(set(fault_classes)) == 1:
        f = FAULT[fault_classes[0]]
        return f.fitness, f.name
    # changing
    counter = collections.Counter(fault_classes)
    ordered = list(FAULT.values())
    ordered.sort(key=operator.attrgetter('fitness'), reverse=True)
    total = 4
    for coeff, fault in zip([1.2, 0.5, 0.2], ordered):
        total = total + coeff * counter[fault.id]
    return total, FAULT['mix'].name
    # return 4 + 1.2 * counter[fault_type.FAIL.value] + 0.5 * counter[fault_type.MUTE.value] + 0.2 * counter[
    #     fault_type.PASS.value], FAULT['mix'].name


def percentage_fitness(fault_classes):
    """
    :param fault_classes:
    :return: fitness value and string name of fault class
    """
    if len(set(fault_classes)) == 1:
        f = FAULT[fault_classes[0]]
        return f.fitness, f.name
    counter = collections.Counter(fault_classes)
    total = 0
    for key in counter.keys():
        f = FAULT[key]
        total = total + f.fitness * counter[key]
    return total/sum(counter.values()), FAULT['mix'].name
