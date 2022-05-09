import numpy as np
from enum import Enum
from fitness import FAULT


class fault_type(Enum):
    PASS = 0
    FAIL = 1
    MUTE = 2
    CHANGING = 3


class dummy_cartography:
    def apply_bench_parameter(self, parameter_set):
        x_min = 0
        x_max = 2000
        x = parameter_set.x
        y = parameter_set.y
        if x < x_min or x > x_max:
            raise ValueError(x, "is not in the allowed limits: [", x_min, ", ", x_max, "].")
        if x < x_max * 0.2:
            return np.random.choice([fault_type.PASS.value, fault_type.MUTE.value])
        if x - x_max * 0.1 < y < x + x_max * 0.1:
            return fault_type.FAIL.value
        if x > x_max * 0.5 and (x_max - x - x_max * 0.1 < y < x_max - x + x_max * 0.1):
            return fault_type.MUTE.value
        return fault_type.PASS.value
