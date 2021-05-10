# from pylab import *
from numpy import *
from scipy import * 
from bn import *
from copy import deepcopy
from subprocess import check_output
from itertools import tee#, izip
import threading
import operator 
import getopt
import sys
import os

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class myThread(threading.Thread):

	def __init__(self, threadID, params):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.params = params
		self.result = None		

	def run(self):
		# print " * Starting thread individual", self.threadID
		if os.name == 'posix':
			simpath = "Rscript"
		else:
			simpath = "D:\\Program Files\\R\\R-4.0.2\\bin\\x64\\Rscript.exe" # to be changed

		command = [simpath, self.params['script'], self.params['path'], self.params['dataset']]
		res = check_output(command)
		res = list(map(float, res.split()))
		self.result = res

class GA(object):

	def __init__(self, BN, individuals=100, priori=None, mutation_rate=0.1):
		self.individuals = individuals
		self.fitness_population = []
		self.population = []
		self.BN = BN
		self.SCRIPT_SCORE = "MAIN_return_the_score.R"
		self.priori = priori
		if self.priori!=None:
			self.priori.generate_nx()
		self.mutation_rate = mutation_rate
		self.DATASET = 1
		print (" * New GA created with", self.individuals, "individuals")
		print (BN, "added to the genetic algorithm class",)
		if not priori==None: print ("using a priori knowledge")
		else: print ("not using a priori knowledge")

	def mutate_population(self, pop):
		#print " * Performing mutation of individuals"
		for net in pop:
			self.mutate(net)
			if self.priori==None: self.fix(net)
		return pop

	def mutate(self, individual):
		nodes = len(individual.adjacency_matrix)
		if self.priori==None:
			#print " * Random mutation without a priori knowledge"
			for i in range(nodes):
				for j in range(nodes):
					if self.mutation_rate>random.random():
						individual.switch_arc(i,j)

		else:
			# print " * Random mutation considering a priori knowledge"
			while(1):
				sel = random.randint(0,nodes,2)
				if individual.is_connected(sel[0], sel[1]):
					individual.remove_arc(sel[0], sel[1])
					break
				else:
					if self.priori.is_connected(sel[0], sel[1]):
						individual.add_arc(sel[0], sel[1])
						break
			
		return individual

	def crossover_population(self, pop):
		print (" * Performing crossover of individuals")
		for n1, n2 in pairwise(pop):
			# n1.show_matrix()
			n1, n2 = self.single_point_crossover(n1, n2)
			self.fix(n1)
			self.fix(n2)
			# n1.show_matrix()
		return pop			

	def single_point_crossover(self, net1, net2):
		allnodes = len(net1.adjacency_matrix)			
		cut = random.randint(0, allnodes*allnodes)		
		net1left =  reshape(array(net1.adjacency_matrix), (allnodes*allnodes))[0:cut]
		net1right =  reshape(array(net1.adjacency_matrix), (allnodes*allnodes))[cut:]
		net2left =  reshape(array(net2.adjacency_matrix), (allnodes*allnodes))[0:cut]
		net2right =  reshape(array(net2.adjacency_matrix), (allnodes*allnodes))[cut:]
		offspring1 =reshape(append(net1left,net2right), (allnodes, allnodes))
		offspring2 =reshape(append(net2left,net1right), (allnodes, allnodes))
		net1.adjacency_matrix=offspring1
		net2.adjacency_matrix=offspring2
		net1.generate_nx()
		net2.generate_nx()
		return net1, net2

	def generate_individual(self, mutations=10):
		nodes = self.BN.nodes
		newbn = BN(nodes, verbose=False)
		for x in range(mutations):
			newbn = self.mutate(newbn)
			newbn = self.fix(newbn)
		return newbn

	def generate_individuals(self, mutations=10):		
		for i in range(self.individuals):		
			newbn = self.generate_individual(mutations=mutations)
			self.population.append(deepcopy(newbn))				
			self.fitness_population.append(0) # we are maximizing

	def fix(self, BN):
		while(BN.is_cycle()):
			(r,c) = BN.pick_random_arc()
			BN.switch_arc(r,c)
		return BN

	def select(self, method="roulette"):
		# print " * Starting selection of population"
		new_population=[]
		new_pop_fitnesses=[]
	
		if method=="roulette":
			integral = sum(self.fitness_population)
			individuals = len(self.population)			
			for x in range(individuals):
				selected = 0 
				cum = self.fitness_population[0]
				rnd = random.random()*integral
				#print " * Selected random number:", rnd
				while(cum>rnd):
					selected+=1
					cum+=self.fitness_population[selected]
				new_population.append(deepcopy(self.population[selected]))
				new_pop_fitnesses.append(self.fitness_population[selected])
		elif method=="ranking":
			dict_fitnesses = {}
			individuals = len(self.population)			
			for n,f in enumerate(self.fitness_population):
				dict_fitnesses[n]=f
			dict_fitnesses =  sorted(dict_fitnesses.items(), key=operator.itemgetter(1), reverse=True)

			rang= range(individuals,0,-1)			
			integral = rang[0]**2/2
			
			for x in range(individuals):
				selected = 0 
				cum = rang[0]
				rnd = random.random()*integral
				#print " * Selected random number:", rnd
				while(cum<rnd):
					selected+=1
					cum+=rang[selected]
				#print " * Selected", selected
				rev_index = dict_fitnesses[selected][0]
				# print " * Corresponding to", rev_index
				new_population.append(deepcopy(self.population[rev_index]))
				new_pop_fitnesses.append(self.fitness_population[rev_index])

		return new_population#, new_pop_fitnesses

	"""
	def fitness(self, BN):
		BN.export_file("./curr_solution_auto.txt")
		ret = check_output(["c:\\Program Files\\R\\R-3.2.3\\bin\\Rscript.exe", 
		"./MAIN_return_the_score.R", 
		"./curr_solution_auto.txt"])
		return float(ret)
	"""

	def evaluate_fitnesses(self, popo):
		# print " * Starting fitness evaluations"

		threads  = []
		for n, individual in enumerate(popo):
			# self.fitness_population[n] = self.fitness(individual)
			if self.priori!=None:
				temporary_filename = "workfiles"+str(self.DATASET)+"_prior"+"/curr_solution_auto"+str(n)+".txt"
			else:
				temporary_filename = "workfiles"+str(self.DATASET)+"/curr_solution_auto"+str(n)+".txt"
			individual.export_file(temporary_filename)
			params = { 	'path': temporary_filename, 
						'dataset': str(self.DATASET),
						'script': self.SCRIPT_SCORE
					 }
			thread = myThread(n, params)
			thread.start()
			threads.append(thread)
		for t in threads:
			t.join()
		print (" * Threads have completed execution")
		ret_fit = []
		for n,t in enumerate(threads):
			# self.fitness_population[n]=t.result
			ret_fit.append(t.result)
			# print " * Fitness for individual",n,"is",self.ret_fit[n]
		#exit()
		return ret_fit
			

	def combine_populations(self, p1, p2, fp1, fp2):		

		#print "NEW GENERATION"
		#print zip(p1, fp1)
		#print zip(p2, fp2)

		newpop = []
		newfit = []
		p1.extend(p2)
		fp1.extend(fp2)
		A =  zip(fp1,range(len(fp1)))
		filtered =  sorted(A, reverse=True)[:len(A)/2]
		#print filtered
		vals, indices = zip(*filtered)
		for i in indices:
			newpop.append(p1[i])
			newfit.append(fp1[i])

		#print zip(newpop, newfit)

		#exit()
		return newpop, newfit


	def evolve(self, iterations=100, selection="ranking"):
		print (" * Optimization started")

		self.fitness_population = self.evaluate_fitnesses(self.population)[:]
		print (self.fitness_population)


		for it in range(iterations):
			print ("*"*100)
			print (" * GA iteration", it+1)
			new_population = self.select(method=selection)
			new_population = self.crossover_population(new_population)
			new_population = self.mutate_population(new_population)
			new_pop_fitnesses = self.evaluate_fitnesses(new_population)

			self.population, self.fitness_population = self.combine_populations(
				self.population, 
				new_population, 
				self.fitness_population,
				new_pop_fitnesses
				)

			print (" * Best individual has fitness", min(self.fitness_population))
			print (" * Iterating...")

def print_help():
	print ("ga.py [-d <dataset>] [-s <population size>] [-m <mutation rate>] [-p <prior enabled>] [-g <generations>] [-o <offset dataset>] [--mpi]")


if __name__ == '__main__':

	DATASET = 1
	USE_PRIOR = False
	MUTATION_RATE = 0.01
	INDIVIDUALS = 32
	ITERATIONS = 100
	OFFSET = 0
	USE_MPI = False

	try:
		opts, args = getopt.getopt(sys.argv[1:], "hd:p:m:s:g:o:", ["help", "dataset", "popsize", "mutationrate", "usepriori", "mpi", "offset", "generations"])
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
			MUTATION_RATE = float(arg)
		elif opt in ("-p", "--usepriori"):
			USE_PRIOR = (arg=="1")
			#USE_PRIOR = True
		elif opt in ("-g", "--generations"):
			ITERATIONS = int(arg)
		elif opt in ("-o", "--offset"):
			OFFSET = int(arg)
		elif opt in ("--mpi"):
			USE_MPI = True


	if USE_MPI:
		from mpi4py import MPI 
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()
		DATASET = rank
	
	DATASET += OFFSET
	print (" * Using dataset", DATASET)

	try:
		os.mkdir("workfiles"+str(DATASET))
	except:
		pass

	if USE_PRIOR:
		bnl = BNloader("./prior/file_"+str(DATASET)+".txt")
		G = GA(bnl, individuals=INDIVIDUALS, mutation_rate=MUTATION_RATE, priori=bnl)
	else:
		bnl = BNloader("./dataset/file_"+str(DATASET)+".txt")
		G = GA(bnl, individuals=INDIVIDUALS, mutation_rate=MUTATION_RATE)


	G.DATASET = DATASET	
	G.generate_individuals(mutations=1)
	G.evolve(iterations=ITERATIONS, selection="ranking")

	print (" * Best solution found:")

	dict_fitnesses = {}
	for n,f in enumerate(G.fitness_population):
		dict_fitnesses[n]=f
	dict_fitnesses =  sorted(dict_fitnesses.items(), key=operator.itemgetter(1), reverse=True)
	best = dict_fitnesses[0][0]
	print (G.population[best].adjacency_matrix)

	try: 
		os.mkdir("results")
	except:
		pass
	G.population[best].export_file("results/dataset-"+str(DATASET)+"_prior-"+str(USE_PRIOR))


	if os.name != 'posix':
		G.population[best].plot("prova_ga.png")

	
