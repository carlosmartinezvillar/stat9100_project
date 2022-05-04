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

#SEEDs
np.random.seed(123456789)     #numpy seed
random.seed(123456789) #let's do lib random to REALLY make sure seed's fixed


# FUNCTIONS
def plot_graph(G):
	pass

def hamming_distance(a,b):
	return (a != b).sum()

def euclid_distance(a,b):
	return np.sum((a-b)**2)

def compute_weights(S):
	# M = S.shape[0]          #nr of states
	# W = np.zeros((N,N))     #empty NxN matrix	
	# for s in S:             #for-each state
	# 	W += np.outer(s,s)    #outer product
	# W = W/M                 #average over nr of states
	# W -= np.eye(N)          #subtract identity
	#------------------------------------------------- alternatively...
	M = S.shape[0]            #nr of memories
	N = S.shape[1]            #nr of neurons
	b = np.sum(S,axis=0) / M  #bias is avg node activity throughout memories
	W = S.T @ S / M - np.eye(N)	
	return W,b


def energies(x,W,b):
	'''
	Return the partial energies (delta E's) for each node
	'''
	E_i = x @ W  +  b
	return E_i

def global_energy(x,W,b):
	'''
	Return the calculated energy for a state vector x, given W and b.
	'''
	E = -0.5 * (x@W)@x.T - b@x.T
	return E

def threshold(x):
	'''
	Return a vector based on the threshold rule. That is, flip the states to 
	the corresponding 1's and -1's >0 or <0. 
	'''
	x_new = np.zeros_like(x)
	x_new[x >= 0] =  1
	x_new[x < 0] = -1
	return x_new

def binary_decision(x_old,W,b):
	x_new = np.zeros_like(x_old)
	x_new = x_old @ W + b
	return threshold(x_new)

def iterate(probe,n_iter=10):
	x = copy.deepcopy(probe)
	print("Dist | X/State ")
	print("-------------------------------------------")
	for i in range(n_iter):
		x_new = binary_decision(x,W,b)
		print(" ",hamming_distance(x,x_new)," | ",x_new)
		x     = x_new
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
		X[i,:] = threshold(np.array(img.flatten() - 0.5)) #[0,1] to [-1,1]
	return X

def plot_small_letters(letters,path):
	plt.figure(figsize=(10,4))
	n = len(letters)
	for _ in range(n):
		plt.subplot(1,n,_+1)
		plt.imshow(np.reshape(letters[_],((6,6))))
		plt.axis('off')
	plt.savefig(path)

def mnist_dataset():
	pass

def plot_mnist():
	pass

def orthogonal_dataset():
	# DISCRETE ORTHOGONAL STATES -- 4 states of 16 neurons
	ortho_states = [[1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1],
				    [1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1],
			        [1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1],
			  		[1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1]]
	S = np.array(ortho_states) #numpy array instead of python array
	return S

def cover(x):
	'''
	Takes in single state vector 'x' and returns half it covered.
	'''
	x_new = copy.deepcopy(x)
	x_new[len(x)//2:] = -1
	return x_new

def randomize(x,p=0.4):
	'''
	"Turn off" (set to -1) a fraction p of nodes in the 1-d array x.  
	'''
	# idxs        = np.random.randint(len(x),size=int(len(x)*p))
	idxs        = np.random.permutation(len(x))[:int(len(x)*p)]
	x_new       = copy.deepcopy(x)
	x_new[idxs] = -1

	if len(x) < 17:
		print("Original vector:  ", end=' ')
		print(x)
		print("Randomized vector:",end=' ')
		print(x_new)
	else:
		print("Original vector:  ",end=' ')
		print(x[0:10],"\b ", " ... ",x[-3:])
		print("Randomized vector:",end=' ')
		print(x_new[0:10],"\b ", " ... ",x_new[-3:])
	print("Hamming Distance: ", hamming_distance(x,x_new),end="/")
	print("Euclidean Distance: ",euclid_distance(x,x_new))
	return x_new

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

################################################################################
# MAIN
################################################################################
if __name__ == "__main__":

	#Simple vectors
	# S    = orthogonal_dataset()
	# W,b  = compute_weights(S)
	# test = randomize(S[1])
	# iterate(test)

	#Small letters
	S    = letters_dataset()
	W,b  = compute_weights(S)
	x    = S[1]
	test = randomize(x)
	plot_small_letters([x,test],"../img/test_letters.png")
	result = iterate(test)
	plot_small_letters([test,result],"../img/test_letters_result.png")

