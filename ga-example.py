from deap import algorithms, base, creator, tools
import numpy as np
import random
from pprint import pprint
import multiprocessing

random.seed(7)

#Definition of function fitness
def my_fitness_function(individual, items, max_weight, min_amount):
	weight=0.0
	amount=0.0
	for i,catch in enumerate(individual):
		weight+=catch*items[i][0]
		amount+=catch*items[i][1]
	if(weight>max_weight or amount< min_amount):
		weight=max_weight
		amount=min_amount
	#print("{} ------> {}".format(individual,(weight,amount)))
	return (weight,amount)

#Crossover operator
def my_crossover_op(ind1, ind2):
	cross_point=random.randint(0,len(ind1)-1)
	tmp=ind1[:]
	ind1[cross_point:]=ind2[cross_point:]
	ind2[:cross_point]=tmp[:cross_point]
	return (ind1,ind2)

#Mutation operator
def my_mutation_op(ind1):
		max_val=len(ind1)-1
		obj=random.randint(0,max_val)
		#prob=1/max_val * obj
		#if(prob<prob_mut):
		#print("El gen muta")
		ind1[obj]=1-ind1[obj]
		return (ind1,)

#Selection operator could be defined or we can use the one in deap.algorithms


if __name__ == '__main__':
	
	####Variables that defines our problem (Knapsack)

	n_items=100
	items={i:(random.randint(1,10),random.randint(1,100)) for i in range(n_items)}
	print(items)

	max_weight_allowed=7*n_items
	min_amount_allowed=20*n_items
	population_size=10*n_items
	
	####################################


	####Variables that configures genetic algorithm
	n_iter=5 #Number of launchs of genetic algorith over the problem
	
	LAMBDA=population_size #The number of children to produce at each generation.
	NGEN=30 #Number of generations
	MU=population_size #The number of individuals to select for the next generation.
	CXPB=0.7 #The probability that an offspring is produced by crossover.
	MUTPB = 0.3 #The probability that an offspring is produced by mutation.
	hof=tools.HallOfFame(maxsize=1) #A HallOfFame object that will contain the best individuals of each iteration, optional.

	#Container building
		#Fitness function will minimize first objective (weight -> -1.0) and maximize the second (amount -> 1.0)
	creator.create("FitnessMinMax",base.Fitness,weights=(-1.0,1.0))
		#Each individual will be a list that will be evaluated through FitnessMinMax fitness function 
	creator.create("Individual",list,fitness=creator.FitnessMinMax)

	#Tool building
	toolbox = base.Toolbox()
		#Each attribute of an individual will be an integer 0 or 1
	toolbox.register("attribute", random.randint,0,1)
		#An individual will be contained into Individual class. It is build from n_items attributes
	toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.attribute, n=n_items)
		#A population is a list of individuals
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		
		#Register fitness function
	toolbox.register("evaluate", my_fitness_function,items=items,max_weight=max_weight_allowed,min_amount=min_amount_allowed)
		#Register crossover operator
	toolbox.register("mate", my_crossover_op)
		#Register mutate operator
	toolbox.register("mutate", my_mutation_op)
		#Register selection operator
	toolbox.register("select", tools.selNSGA2)

	#Building statistic for each GA run
	stats = tools.Statistics(lambda individual: individual.fitness.values)
	stats.register("avg", np.mean, axis=0)
	stats.register("std", np.std, axis=0)
	stats.register("min", np.min, axis=0)
	stats.register("max", np.max, axis=0)

	

	

	
	#We define each GA execution into a function that receives the number of iteration in order to parallelize executions
	def iteration(i):
		initial_population=toolbox.population(n=MU)
		pop,result=algorithms.eaMuPlusLambda(initial_population, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame=hof,verbose=False)
		#Show results
		best_ind=hof[0]
		best_fit=my_fitness_function(best_ind, items, max_weight_allowed, min_amount_allowed)
		print("Iter {}: individual: {}, fitness: {}".format(i,best_ind,best_fit))
		return (i,best_ind,best_fit,stats)
		



	#Configure how many cores will execute the set of tasks and create a Pool
	number_of_cores = 4
	pool = multiprocessing.Pool(number_of_cores)
	#We are goint to execute n_iter task, so we say it
	total_tasks = n_iter
	tasks = range(total_tasks)
	#We assign and start tasks to each core.
	results = pool.map_async(iteration, tasks)

	#We close the pool and wait all the processes to finish
	pool.close()
	pool.join()


	#We can here print stats for each iteration, order individuals by quality, etc.