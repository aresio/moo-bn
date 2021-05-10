from deap import base, creator, tools
from numpy import savetxt, array
import random
from bn import *
from ga import GA
import getopt
import sys
import os


def convert_candidates(candidates, nodes):
	new_candidates=[]
	for c in candidates:
		new_bn = BN(nodes, initial=c)
		new_candidates.append(new_bn)
	return new_candidates

def mutate_offspring(candidates, nodes=None):
	new_candidates = convert_candidates(candidates, nodes)
	ret_pop = G.mutate_population(new_candidates)
	for c in range(len(candidates)):
		for i in range(nodes*nodes):
			candidates[c][i] = ret_pop[c].adjacency_matrix.reshape(1, nodes*nodes)[0][i]
	return candidates

def generator(random, args): 
	ret = G.generate_individual(mutations=args['initial_mutations'])
	return ret  
	
def evaluate_all(candidates, args, verbose=False, nodes=None):
	new_candidates = convert_candidates(candidates, nodes)
	fit = G.evaluate_fitnesses(new_candidates)
	#print fit
	#exit()
	#fit2 = [len(can.nxrep.edges()) for can in new_candidates]
	#ret = zip(fit, fit2)
	return fit

def print_help():
	pass

if __name__ == '__main__':

	TESTS = [ [1.0, -1.0] ]

	for test in TESTS:

		TGT1 = test[0]
		TGT2 = test[1]


		creator.create("FitnessMulti", base.Fitness, weights=(TGT1, TGT2))
		creator.create("Individual", list, fitness=creator.FitnessMulti)

		# defaults
		IND_SIZE = 10*10
		NGEN = 100
		CXPB = 0.9
		MUPB = 0.05
		INDIVIDUALS = 128
		USE_PRIOR = False
		DATASET = 0
		OFFSET = 0

		# processing command line arguments
		try:
			opts, args = getopt.getopt(sys.argv[1:], "hd:p:m:c:s:g:o:", ["help", "dataset", "popsize", "mutationrate", "crossoverrate", "usepriori", "mpi", "offset", "generations"])
		except getopt.GetoptError:
			print ("ERROR:") 
			print_help()
			exit(-2)

		for opt, arg in opts:
			if opt in ("-h", "--help"):
				print_help()
				sys.exit()
			elif opt in ("-d", "--dataset"):
				DATASET = int(arg)
			elif opt in ("-s", "--popsize"):
				INDIVIDUALS = int(arg)
			elif opt in ("-m", "--mutationrate"):
				MUPB = float(arg)
			elif opt in ("-c", "--crossoverrate"):
				CXPB = float(arg)
			elif opt in ("-p", "--usepriori"):
				USE_PRIOR = (arg=="1")			
			elif opt in ("-g", "--generations"):
				NGEN = int(arg)
			elif opt in ("-o", "--offset"):
				OFFSET = int(arg)
			elif opt in ("--mpi"):
				USE_MPI = True

		output_directory = "results_nsga2_%d" % OFFSET
		debug_file = output_directory+"/__convergence_"+str(TGT1)+"_"+str(TGT2)
		
		try:
			os.mkdir(output_directory)
		except:
			pass

		# create DEAP's toolbox
		toolbox = base.Toolbox()

		# the individuals are composed of boolean values
		toolbox.register("attr_bool", random.randint, 0, 1)
		toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=IND_SIZE)
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)

		#  two-point crossover
		toolbox.register("mate", tools.cxTwoPoint)

		# selection mechanism: Non-dominated Sorting Genetic Algorithm v2
		toolbox.register("select", tools.selNSGA2)	
		# toolbox.register("select", tools.selSPEA2)	

		# create population
		pop_deap = toolbox.population(n=INDIVIDUALS)

		# analyze datasets to retrieve information about the number of nodes
		if USE_PRIOR:
			bnl = BNloader("./prior/file_"+str(DATASET)+".txt", delimiter="\t", skiprows=0)
			G = GA(bnl, individuals=INDIVIDUALS, mutation_rate=MUPB, priori=bnl)			
		else:
			bnl = BNloader("./dataset/file_"+str(DATASET)+".txt", delimiter=",", skiprows=1)
			G = GA(bnl, individuals=INDIVIDUALS, mutation_rate=MUPB)
		G.DATASET = DATASET

		debug_file +=  "_dataset"+str(DATASET)

		IND_SIZE = G.BN.nodes**2

		# use independent scores (NSGA-2)
		G.SCRIPT_SCORE ="MAIN_return_the_score_nobic.R"

		# let's generate almost empty individuals
		for i, individual in enumerate(pop_deap):

			new_individual = G.generate_individual(mutations=1)
			new_individual_list = list(new_individual.adjacency_matrix.reshape((1,IND_SIZE))[0])

			#print (individual)
			#print (new_individual_list)
			#exit()

			for j in range(IND_SIZE):
				individual[j] = new_individual_list[j]

		# first evaluation of the fitness of all individuals
		fitnesses = evaluate_all(pop_deap, None, nodes=bnl.nodes)
		for ind, fit in zip(pop_deap, fitnesses):
			ind.fitness.values = fit[0],fit[1]
			#print ind.fitness.values
		#exit()

		with open(debug_file, "w") as fo:

			# Begin the evolution
			for gen in range(NGEN):
				
				print ("-- Generation %i --" % gen)

				# Select the next generation individuals
				offspring = toolbox.select(pop_deap, INDIVIDUALS)

				# Clone the selected individuals
				offspring = list(map(toolbox.clone, offspring))

				# Apply crossover and mutation on the offspring		
				for child1, child2 in zip(offspring[::2], offspring[1::2]):
					if random.random() < CXPB:
						toolbox.mate(child1, child2)
						del child1.fitness.values
						del child2.fitness.values


				offspring = mutate_offspring(offspring, nodes=bnl.nodes)

				pop_deap[:] = offspring

				# re-evaluate fitnesses
				fitnesses = evaluate_all(pop_deap, None, nodes=bnl.nodes)

				fo.write("\t".join(map(str, fitnesses))+"\n")

				for ind, fit in zip(pop_deap, fitnesses):
					ind.fitness.values = fit

				print (fitnesses)

		final = convert_candidates(pop_deap, G.BN.nodes)
		for n,ind in enumerate(final):		
			ind.export_file(output_directory+"/dataset-"+str(DATASET)+"_prior-"+str(USE_PRIOR)+"_paretoind-"+str(n))

		for n,ind in enumerate(pop_deap):
			savetxt(output_directory+"/dataset-"+str(DATASET)+"_prior-"+str(USE_PRIOR)+"_paretoind-"+str(n)+"_fitness", ind.fitness.values)
