# Parallelizing genetic algorithms with DEAP

In this repo I will show how to parallelize genetic algorithms (GAs) using Python library DEAP (https://github.com/DEAP/deap).

The whole set of genetic algorithms must define the following components:

* Individual structure: structure of a solution
* Fitness function: measure individual quality
* Crossover operator: how individuals crossover each other
* Mutation operator: how an individual can mutate
* Selection operator: tournaments, best... there are a lot of criteria

DEAP allows us to define and run a GA very quickly.

In order to parallelize one execution, DEAP only allows to switch between cores. This example tell us how to force to run different iterations of GAs concurrently.