################################################################################
# cam.py
################################################################################
'''
This is the main script for this project. It contains helper functions and the 
model itself, which is a discrete state hopfield neural network.
'''

# LIBRARIES
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
from imageio import imread
import networkx as nx

#SEEDs
# np.random.seed(123456789)     #numpy seed
# random.seed(123456789) #let's do lib random to REALLY make sure seed's fixed

################################################################################
# FUNCTIONS
################################################################################
def plot_graph(G):
	pass


def hamming_distance(a,b):
	return (a != b).sum()


def euclid_distance(a,b):
	return np.sum((a-b)**2)


def states_distance(memories,state):
	print("\nNetwork state vs memory states")
	for i,m in enumerate(memories):
		d = hamming_distance(m,state)
		print("Hamming distance to memory %i: %i" % (i,d))


def compute_weights(S):
	# M = S.shape[0]          #nr of memories
	# W = np.zeros((N,N))     #empty NxN matrix	
	# for s in S:             #for-each state
	# 	W += np.outer(s,s)    #outer product
	# W = W/M                 #average over nr of states
	# W -= np.eye(N)          #subtract identity
	#------------------------------------------------- alternatively...
	M = S.shape[0]            #nr of memories
	N = S.shape[1]            #nr of neurons/nodes in gr  aph
	b = np.sum(S,axis=0) / M  #bias is avg node activity throughout memories
	W = (S.T @ S) / M - np.eye(N)	
	return W,b


def delta_E(x,W,b):
	'''
	Return the partial energies (delta E's) for each node
	'''
	E_i = x  @ W - b
	return E_i


def global_energy(x,W,b):
	'''
	Return the calculated energy for a state (row) vector x, given W and b.
	'''
	E = -0.5*(x@W)@x.T - b@x.T
	return E


def sgn(x):
	'''
	Return a vector based on the threshold rule: flip the states to 
	the corresponding 1's and -1's by >0 or <0. 
	'''
	# x_new = np.zeros_like(x)
	x[x > 0] =  1
	x[x < 0] = -1
	return x


def binary_decision(x,W,b):
	# x_new = np.zeros_like(x)
	x_new = x @ W - b
	x_new[x_new < 0] = -1
	x_new[x_new >= 0] = 1
	return x_new.astype(int)


def iterate(probe,orig,n_iter=10):
	x = copy.deepcopy(probe)
	dists,states,iters, energy = [],[],[],[]
	dists.append(hamming_distance(x,orig))
	states.append(x)
	energy.append(global_energy(x,W,b))

	for i in range(n_iter):
		x_new = binary_decision(x,W,b)
		x = x_new
		states.append(x)
		dists.append(hamming_distance(orig,x))
		energy.append(global_energy(x,W,b))

	print("Iter | Energy "+" "*8+ " | Hamming | X/State ")
	print("-"*80)
	for i,d in enumerate(dists):
		print("%4i | %15i | %7i | " % (i,energy[i],d),end='')
		print_state(states[i])
	return x

def letters_dataset(): # VERY SMALL RESOLUTION LETTERS
	letters = []
	letters.append(imread('../img/A.png'))
	letters.append(imread('../img/B.png'))
	letters.append(imread('../img/C.png'))
	letters.append(imread('../img/E.png'))
	M = len(letters)               #nr predefined states/memories
	N = len(letters[0].flatten())  #nr of pixels per letter/neurons
	X = np.zeros((M,N))
	for i,img in enumerate(letters):
		X[i,:] = sgn(np.array(img.flatten() - 0.5)) #[0,1] to [-1,1]
	return X


def plot_small_letters(letters,path,labels=None):
	plt.figure(figsize=(10,4))
	n = len(letters)

	for p in range(n):
		if labels is not None:
			plt.subplot(1,n,p+1,title=labels[p])
		else:
			plt.subplot(1,n,p+1)
		plt.imshow(np.reshape(letters[p],((6,6))))
		plt.axis('off')
		plt.savefig(path)


def mnist_dataset(n_samples=3):
	path = "../img/mnist_10.csv"
	digits = np.loadtxt(path,delimiter=",")
	digits = digits[:,1:] #1st number is the label, remove it
	digits = digits/255
	digits[digits > 0] =  1
	digits[digits < 1] = -1
	if n_samples > 10:
		print("File only contains 10 samples!")
		return digits
	return digits[0:n_samples]


def plot_mnist(digits,path,labels=None):
	# path = "../img/digits.png"
	plt.figure(figsize=(9,4))
	for p in range(3):
		if labels is not None:
			plt.subplot(130+1+p,title=labels[p])
		else:
			plt.subplot(130+1+p)
		plt.imshow(digits[p].reshape((28,28)),cmap=plt.get_cmap("gray"))
	plt.savefig(path)

def orthogonal_dataset():
	# DISCRETE ORTHOGONAL STATES -- 4 states of 16 neurons
	ortho_states = [[1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1],
			        [1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1],
			  		[1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1],
				    [1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1]]
	# ortho_states = [[1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1],
			  	    # [1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1]]
			  		
	S = np.array(ortho_states) #numpy array instead of python array
	return S


def cover(x,c=0.5):
	'''
	Takes in single state vector 'x' and returns half it covered.
	'''
	x_new = np.copy(x)
	x_new[int(len(x)*c):] = -1

	#print details
	print()
	print("Original vector:  ", end=' ')
	print_state(x)
	print("Randomized vector:",end=' ')
	print_state(x_new)
	print("Hamming Distance:   %i" % hamming_distance(x,x_new))
	print("Euclidean Distance: %.3f" % euclid_distance(x,x_new),end='\n\n')

	return x_new


def randomize(x,p=0.4):
	'''
	"Turn off" (set to -1) a fraction p of nodes in the 1-d array x.  
	'''

	#shuffle the indexes using a uniform dist.
	idxs        = np.random.permutation(len(x))[:int(len(x)*p)]
	x_noise       = copy.deepcopy(x)
	x_noise[idxs] = -1

	#print a bunch of details
	print()
	print("Original vector:  ", end=' ')
	print_state(x)
	print("Randomized vector:",end=' ')
	print_state(x_noise)
	print("Hamming Distance:   %i" % hamming_distance(x,x_noise))
	print("Euclidean Distance: %.3f" % euclid_distance(x,x_noise),end='\n\n')
	return x_noise


def random_vector(p=0.5,size=5):
	'''
	Return a completely random vector of 1's and -1's.
	'''
	x = np.random.choice(2,size,p=p)
	x[x < 1] = -1
	return x


def hinton_example():
	x = np.array([1,0,1,0,0])
	W = [[ 0,-4, 3, 2, 0],
		 [-4, 0, 0, 3, 3],
		 [ 3, 0, 0,-1, 0],
		 [ 2, 3,-1, 0,-1],
		 [ 0, 3, 0,-1, 0]]
	W = np.array(W)
	b = copy.deepcopy(x)
	return x,W,b


def print_state(x_array):
	if len(x_array) > 17:
		for j in x_array[0:10]:
			print("%2i" % j,end=' ')
		print(" ... %2i %2i %2i" % (x_array[-3],x_array[-2],x_array[-1]))
	else:
		for j in x_array:
			print("%2i" % j,end=' ')
		print("")


def plot_graph(V,E):
	pass

	G = nx.Graph()
	for i,v in enumerate(V):
		if v > 0:
			c = 'blue'
		else:
			c = 'red'
		G.add_node("X"+str(i),color=c)

	for i in range(E.shape[0]):
		for j in range(E.shape[1]):
			if i < j:
				G.add_edge(i,j,weight=E[i,j])

################################################################################
# MAIN
################################################################################
if __name__ == "__main__":

	#SIMPLE VECTORS
	S    = orthogonal_dataset()
	W,b  = compute_weights(S)
	orig = S[0]
	test = randomize(orig,p=0.4)
	# test = cover(orig)
	final_state = iterate(test,orig,n_iter=10)
	states_distance(S,final_state)

	#LETTERS
	# S    = letters_dataset()
	# # plot_small_letters([orig,test],"../img/test_letters.png")
	# W,b  = compute_weights(S)
	# orig = S[3]
	# p    = 0.8
	# test = randomize(orig,p=p)
	# final_state = iterate(test,orig)
	# labels = ["Original","Randomized p=%.1f" % p,"Recovered"]
	# plot_small_letters([orig,test,final_state],"../img/letters_result.png",labels)

	#MNIST
	# S = mnist_dataset(n_samples=4)
	# # plot_mnist(S,"../img/digits.png")
	# W,b  = compute_weights(S)
	# orig = S[0]
	# p = 0.5
	# test = randomize(orig,p=p)
	# # test = cover(orig,c=p)
	# final_state = iterate(test,orig,n_iter=30)
	# labels = ["Original","Randomized p=%.1f" % p,"Recovered"]
	# labels[-2] += "\nH: %i" % hamming_distance(test,orig)
	# labels[-2] += ", E: %i" % global_energy(test,W,b)
	# labels[-1] += "\nH: %i" % hamming_distance(final_state,orig)
	# labels[-1] += ", E: %i" % global_energy(final_state,W,b)
	# plot_mnist([orig,test,final_state],"../img/digits_result_good_random.png",labels)


