"""Fitness evaluation and tournament parent selection."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence

from .meals import Plan

Fitness = Callable[[Plan], float]


@dataclass(frozen=True)
class ScoredPlan:
    plan: Plan
    score: float


def evaluate(plan: Plan, fitness: Fitness) -> ScoredPlan:
    score = fitness(plan)
    if not math.isfinite(score):
        raise ValueError("Fitness must be a finite number")
    return ScoredPlan(plan, score)


def tournament_select(
    population: Sequence[ScoredPlan], tournament_size: int, rng: random.Random
) -> ScoredPlan:
    if tournament_size < 1 or tournament_size > len(population):
        raise ValueError("Tournament size must be between 1 and the population size")
    return max(rng.sample(population, tournament_size), key=lambda candidate: candidate.score)
