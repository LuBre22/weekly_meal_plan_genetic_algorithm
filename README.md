# Diet genetic algorithm components

This project provides lightweight Python standard-library components for evolving
seven-day meal plans. Each plan is an immutable tuple of seven days, each
containing `(breakfast_id, lunch_id, dinner_id)`. IDs in `meals.csv` are stable
identifiers: do not change or reuse them when editing or reordering meals.
Nutrition values are approximate per-meal estimates.
The catalogue contains 200 illustrative options, including higher-calorie
meals. These are not verified recipes or nutrition data: portion sizes and
ingredients are not specified, so verify values against actual ingredients
and serving sizes before relying on a plan.

The low-level algorithm accepts a caller-provided fitness function. The main
program maximizes weekly protein subject to user-provided nutrition limits. Repeats
are permitted and each meal represents one fixed serving.

## Run with nutrition constraints

Set the `DAILY_MIN_*` and `WEEKLY_MAX_*` variables near the top of
`main.py` to your daily minimums and weekly maximums for calories (kcal),
protein, fat, and carbs (grams). A `None` setting requires a command-line
value; no calorie targets are assumed. After setting them, run `python main.py`.

Alternatively, supply any or all limits as command-line options; these override
the corresponding file settings. Every limit must be set by one of the two
methods. For example, the following limits are illustrative test inputs,
**not nutrition recommendations**:

```powershell
python main.py --daily-min-calories 1000 --daily-min-protein 50 --daily-min-fat 30 --daily-min-carbs 100 --weekly-max-calories 20000 --weekly-max-protein 1200 --weekly-max-fat 500 --weekly-max-carbs 2000
```

Run `python main.py --help` for the options. Edit the settings near
the top of `main.py` to change the population
size, number of generations, mutation rates, phase-switch settings, random
seed, and other parameters. Set `INCLUDE_STATISTICS = True` to print the
generation history. The meal file is resolved relative to `main.py`, so it
works from another working directory too.

Every feasible plan outranks every infeasible plan; feasible plans are ranked
by weekly protein in grams. Infeasible plans are ranked by the sum of relative
constraint violations, using each limit (or 1 when the limit is below 1)
as the denominator. The displayed fitness is a negative penalty for an
infeasible plan, not protein in grams. If no feasible plan is found, the
report summarizes violations immediately below the weekly total as daily
shortfalls summed over the week and weekly excesses, separately for calories
and each macro. A day above its minimum does not offset another day's shortfall.
It exits with code 2; that does not
prove no feasible plan exists. Limits must be finite and nonnegative; each
weekly maximum must be at least seven times its daily minimum.

The constraints don't limit daily maximums, enforce variety, or
ensure nutritional adequacy. Nutrition values are estimates and fixed servings;
choose appropriate targets with qualified guidance before relying on results.
Set a fixed `RANDOM_SEED` for reproducible runs; use `None` for a new random sequence.

## Low-level API example

The following example deliberately uses an unconstrained fitness function rather
than the main program's macro constraints:

```python
from pathlib import Path
from random import Random

from diet_ga.meals import MealCatalog
from diet_ga.population import initialize_population
from diet_ga.selection import evaluate
from diet_ga.report import format_report, run_with_report

catalog = MealCatalog.from_csv(Path("meals.csv"))
rng = Random(42)

# Demonstration only, not a recommended nutrition objective.
def fitness(plan):
    return sum(catalog.by_id[meal_id].protein_g for day in plan for meal_id in day)

population = [
    evaluate(plan, fitness) for plan in initialize_population(200, catalog, rng)
]
report = run_with_report(
    population, catalog, fitness, rng,
    generations=100, min_exploration_generations=20,
    max_exploration_generations=70, stagnation_generations=10,
    improvement_threshold=1.0,
    exploration_mutation_rate=0.5, exploitation_mutation_rate=0.05,
    elite_count=5, immigrant_count=10, tournament_size=3,
    include_statistics=True,
)
print(format_report(report))
```

`initialize_population` samples each eligible slot independently. Tournament
selection samples without replacement within each tournament; parents may be the
same individual. Day-level crossover takes each entire day from one parent.
`mutate` independently attempts a replacement for each day at the supplied rate,
choosing one random slot and a different eligible meal. A slot with only one
eligible meal is unchanged. Add future mutation operations to the `mutate`
entry point without changing the generation loop.

`next_generation` retains the highest-scoring elites, evaluates newly created
children, and evaluates independently generated random immigrants. Its output
has the same size as its input. `elite_count` must be at least one, and elite
and immigrant counts together must not exceed the population size. This reserves
at least one slot for the best candidate from the previous generation, even if
all other slots are immigrants. Set a larger elite count to reserve more slots.
The caller must supply scores evaluated with the same fitness function and
catalogue and reevaluate the population if either changes.

`run_evolution` applies `next_generation` exactly `generations` times.
Exploration lasts at least `min_exploration_generations` and at most
`max_exploration_generations`, both integer counts. The maximum must be less
than the total, leaving at least one exploitation generation; zero is allowed
for either bound. Set both bounds to the same value for a fixed switch.

After each exploration generation, the best score is compared with the score
at the last significant improvement (initially the initial population's best).
A strictly positive cumulative gain of at least `improvement_threshold` resets
the stagnation counter and updates that baseline; otherwise the counter grows.
Once the minimum is met and the counter reaches `stagnation_generations`, the
next generation uses the lower exploitation mutation rate. Stagnation is
counted from the start, including the minimum period. If it never occurs, the
maximum forces a switch. A threshold of zero still requires a positive gain.
The threshold is an absolute fitness-score difference, so tune it to the scale
of the fitness function. The counter must be positive and the threshold finite
and nonnegative. Mutation rates must satisfy
`0 <= exploitation_mutation_rate <= exploration_mutation_rate <= 1`.

## Reports

`run_with_report` uses the same evolution parameters as `run_evolution` and
returns a `WeeklyReport` containing the best candidate's score, seven named
days with meal names and IDs, daily nutrition totals, weekly totals, and daily
averages (weekly totals divided by seven). Values retain their full precision;
`format_report` rounds them for display. Meal values are estimates per serving.
Pass `violations=assessment.violations` to `format_report` to include the
condensed constraint summary directly below the weekly total. Individual
violations remain available in `assessment.violations`.
To serialize the structured report as JSON, use
`json.dumps(dataclasses.asdict(report), indent=2)` after importing those modules.

Set `include_statistics=True` to record the initial population (generation 0)
and every subsequent generation's phase, mutation rate, best and mean fitness,
and number of distinct whole plans. The report also includes the first
exploitation generation and all generations where the best score strictly
exceeded the previous best, independently of the switch's improvement
threshold. The initial entry has no mutation rate; the first exploitation
generation is one-based. Statistics are omitted by default, avoiding history
collection. `format_report` includes the history only when requested.
This option does not change the search or random sequence. The lower-level
`run_evolution` still returns the final population and supports an optional
`on_generation` callback receiving the generation number, phase, mutation
rate, and immutable population snapshot after each generation.

These are per-day replacement probabilities, not per-slot probabilities. Random
immigrants are still introduced in both phases. This schedule encourages broad
search and then refinement, but does not guarantee a global optimum. Stagnation
may signal a local optimum rather than readiness to exploit; the current trigger
does not measure population diversity or increase mutation after the switch.
Its rates, bounds, and stagnation settings need experimental tuning. Guaranteed
elitism preserves the best fitness seen
when the fitness function and catalogue do not change; compare several random
seeds. For more complex adaptation, call `next_generation` directly in a loop.

Run all tests from the project directory with `python -m unittest discover -s tests -v`.
Individual test files can also be run directly, for example
`python tests\test_strategy.py`. Direct execution adds the project root to
Python's import path so that the local `diet_ga` package can be found.
