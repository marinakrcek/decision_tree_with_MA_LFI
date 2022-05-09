import numpy as np
from parameters import ParameterSet
from helper import randrange_float
import random 


def fix_rules_intervals(rule_intervals, param_bounds):
    """
    Rules can have a larger or smaller intervals for the parameters than parameter bounds
    set by the user for the currently running campaign.
    So, if rules are out of bounds for the current campaign, they are not used.
    If the rule intervals are larger, they are clipped to user-defined bounds.
    If rule intervals are smaller but inside the user-defined bounds, than the smaller intervals are used
    for initialization.
    If rules don't cover all parameters, those are added from the parameter bounds.
    Additional parameter (like product and test info) are not changed.
    :param rule_intervals: Expected to be already for expanding (max+step, not just max)
    :param param_bounds: Same, for expanding
    :return:
    """
    for param, limits in param_bounds.items():
        rule_limits = rule_intervals.get(param, None)
        if rule_limits is None:
            rule_intervals[param] = limits  # put the parameter bounds for the parameter that is not in the rule
            continue
        assert (rule_limits[2] == limits[2]), 'step in the rule different than the one defined for the campaign'
        rule_intervals[param][0] = max(limits[0], rule_limits[0])
        rule_intervals[param][1] = min(limits[1], rule_limits[1])
        if rule_intervals[param][0] >= rule_intervals[param][1]:
            rule_intervals[param] = limits


def initialize_from_past_fail_rules(rules_for_fail, use_rules=25):
    """

    :param rules_for_fail: list of dictionaries with parameter bounds for fails
    :param use_rules: number of rules to use
    :return: initialization method using this past knowledge
    """

    def initialize_from_past_fails(pop_size):
        np.random.seed()
        pop = set()
        use = min(use_rules, len(rules_for_fail))
        fail_samples_per_rule = np.array([rule['status']['nb_samples_per_class'][rule['status']['class_index']] for rule in rules_for_fail])
        indexes = np.argsort(fail_samples_per_rule)[::-1][:use]
        np.random.shuffle(indexes)
        parameter_bounds = ParameterSet.get_param_limits(for_expanding=True, as_dict=True)
        while len(pop) < pop_size:
            i_random_rule = np.random.choice(indexes, p=fail_samples_per_rule[indexes]/np.sum(fail_samples_per_rule[indexes]))
            random_rule = rules_for_fail[i_random_rule]
            fix_rules_intervals(random_rule, parameter_bounds)
            x = randrange_float(*random_rule['x'])
            y = randrange_float(*random_rule['y'])
            delay = randrange_float(*random_rule['delay'])
            pulse_width = random.randrange(*random_rule['pulse_width'])
            intensity = randrange_float(*random_rule['intensity'])
            pop.add(ParameterSet(x=x, y=y, delay=delay, pulse_width=pulse_width, intensity=intensity))
        return list(pop)

    return initialize_from_past_fails
