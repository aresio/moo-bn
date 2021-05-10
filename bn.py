import numpy
#import sa
import networkx as nx
try:
	import matplotlib.pyplot as plt
except:
	#print "MATPLOTLIB not available"
	pass

class BN(object):

	def __init__(self, nodes, verbose=False, initial=None):	
		assert nodes>2
		if verbose: print (" * New empty BN with", nodes, "nodes created")
		self.nodes= nodes
		self.nxrep = nx.DiGraph()
		if initial==None:
			self.adjacency_matrix = numpy.zeros((nodes, nodes), dtype=int)
		else:
			self.adjacency_matrix = numpy.array(initial).reshape((nodes, nodes))
			#print "Created adjacency matrix:", self.adjacency_matrix
			self.generate_nx()

	def is_cycle(self):
		cycles = len(list(nx.simple_cycles(self.nxrep)))
		return cycles>0

	def pick_random_arc(self):
		"""
		list_arcs=[]
		for r in xrange(self.nodes):
			for c in xrange(self.nodes):
				if self.is_connected(r,c):
					list_arcs.append((r,c))
		"""
		import random
		ret =  random.sample(self.nxrep.edges(),1)
		return ret[0]

	def is_connected(self, n1, n2):
		return self.adjacency_matrix[n1][n2]!=0

	def switch_arc(self, n1, n2, verbose=False):
		if verbose: print (" * Switching arc", n1, n2)
		if self.is_connected(n1,n2):
			self.remove_arc(n1, n2, verbose)
		else:
			self.add_arc(n1, n2, verbose)

	def add_arc(self, n1, n2, verbose=False):
		if (self.adjacency_matrix[n1][n2]!=0):
			if verbose:	print ("WARNING: cannot add arc between", n1, "and", n2, ": arc is already there")
		else:
			if self.adjacency_matrix[n2][n1]!=0:
				if verbose: print ("WARNING: cannot add arc between", n1, "and", n2, "because the inverse arc already exists")
			else:
				self.adjacency_matrix[n1][n2]=1
				self.nxrep.add_edge(n1, n2)
				if verbose: print (" * New arc between", n1, "and", n2, "added")

	def remove_arc(self, n1, n2, verbose=False):
		if (self.adjacency_matrix[n1][n2]==0):
			if verbose: print ("WARNING: cannot remove arc between", n1, "and", n2, ": arc is already there")
		else:
			self.adjacency_matrix[n1][n2]=0
			self.nxrep.remove_edge(n1,n2)
			if verbose: print (" * Arc between", n1, "and", n2, "removed")

	def show_matrix(self):
		for r in range(self.nodes):
			for c in range(self.nodes):
				print (self.adjacency_matrix[r][c])
			print ()
		print()

	def generate_nx(self):
		self.nxrep = nx.DiGraph()
		for r in range(self.nodes):
			for c in range(self.nodes):
				if self.is_connected(r,c):
					self.nxrep.add_edge(r,c)

	def export_file(self, path, noheader=True, delimiter='\t'):
		assert path!=""
		first_line = ["node_"+str(x) for x in range(self.nodes)]
		first_line = "\t".join(first_line)		
		if noheader:
			try:
				numpy.savetxt(path, self.adjacency_matrix, delimiter=delimiter, fmt="%d", comments="")
			except:
				import os
				os.mkdir(os.path.dirname(path))				
				numpy.savetxt(path, self.adjacency_matrix, delimiter=delimiter, fmt="%d", comments="")
		else:
			try:
				numpy.savetxt(path, self.adjacency_matrix, delimiter=delimiter, fmt="%d", header=first_line, comments="")
			except:
				import os
				os.mkdir(os.path.dirname(path))
				numpy.savetxt(path, self.adjacency_matrix, delimiter=delimiter, fmt="%d", header=first_line, comments="")				

	
	def __repr__(self):
		return "Bayes network"

	def plot(self, output_file):				
		pos = nx.graphviz_layout(self.nxrep, prog='dot')
		nx.draw(self.nxrep, pos)		
		plt.show()

class BNloader(BN):

	def __init__(self, input_file=None, delimiter="\t", skiprows=0):
		assert not input_file==None
		print (" * Import input BN", input_file)
		with open (input_file) as fi:
			nodes = len(fi.readline().split(delimiter))
		print (" * Will create new BN with",nodes, "nodes")
		BN.__init__(self, nodes)
		self.adjacency_matrix = numpy.loadtxt (input_file, skiprows=skiprows, delimiter=delimiter)
		self.nodes = nodes


if __name__ == '__main__':

	pass
	b = BN(3)
	# b.plot("prova.png")
	# b.show_matrix()
	# b.import_prior("priori")
	# b.export_file("prova")

	# b.put_arc(from=0, to=0, assuming=
