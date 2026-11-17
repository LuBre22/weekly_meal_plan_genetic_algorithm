"""Day-level crossover."""

import random

from .meals import DAYS_PER_WEEK, Plan


def crossover(first: Plan, second: Plan, rng: random.Random) -> Plan:
    if len(first) != DAYS_PER_WEEK or len(second) != DAYS_PER_WEEK:
        raise ValueError("Both parents must contain seven days")
    return tuple(rng.choice((first_day, second_day)) for first_day, second_day in zip(first, second))
