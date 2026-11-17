"""Extensible mutation entry point."""

import math
import random

from .meals import Day, MEAL_TYPES, MealCatalog, Plan


def replace_meal(plan: Plan, catalog: MealCatalog, rate: float, rng: random.Random) -> Plan:
    """Independently replace at most one meal in each day with probability rate."""
    days = []
    for day in plan:
        if rng.random() >= rate:
            days.append(day)
            continue
        slot = rng.randrange(len(MEAL_TYPES))
        eligible = catalog.eligible[MEAL_TYPES[slot]]
        if len(eligible) < 2:
            days.append(day)
            continue
        # Choose uniformly from all eligible IDs except the current one.
        current_index = catalog.eligible_index[MEAL_TYPES[slot]][day[slot]]
        replacement_index = rng.randrange(len(eligible) - 1)
        if replacement_index >= current_index:
            replacement_index += 1
        updated = list(day)
        updated[slot] = eligible[replacement_index]
        new_day: Day = (updated[0], updated[1], updated[2])
        days.append(new_day)
    return tuple(days)


def mutate(plan: Plan, catalog: MealCatalog, rate: float, rng: random.Random) -> Plan:
    if not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("Mutation rate must be between 0 and 1")
    return replace_meal(plan, catalog, rate, rng)
