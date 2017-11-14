import itertools
import numpy as np
from scipy.stats import entropy

def determine_bias_baseline(T, largest_values):
	"""
	Finds the baseline bias parameter value for the intrinsic plasticity rule, 
	which is used to shift bias update sizes.
	largest_values: a list of maximal values of each predictor in this SAM.
	"""

	num_predictors = len(largest_values)
	return (2.5 / T) * (-num_predictors * np.log(0.2) - np.sum([np.log(m + 1) for m in largest_values]) + np.log(0.02))


def generate_distribution(num_vars, num_discrete_values, randomiser=np.random.RandomState()):
	"""
	Generates a joint discrete distribution with the given number of variables, 
	each of which has the given number of discrete values. This does not distinguish
	between any variable, and distribution parameters are generated randomly. The 
	value 0 is skipped, i.e. variables can take the values [1, num_discrete_values].
	randomiser: a numpy.random.RandomState
	Returns a dictionary of tuple:probability pairs, e.g:
	{
	 (1, 1):0.27,
	 (1, 2):0.31,
	 (2, 1):0.41,
	 (2, 2):0.01
	}
	"""

	var_values = range(1, num_discrete_values + 1)
	possibilities = list(itertools.product(var_values, repeat=num_vars))
	probabilities = np.array([randomiser.random_sample() for p in possibilities])
	probabilities = probabilities / probabilities.sum()
	
	return dict(zip(possibilities, probabilities))


def draw_from_distribution(distribution, complete=True, randomiser=np.random.RandomState()):
	"""
	Draws a complete or incomplete sample from the given distribution. A sample consists
	of a tuple whose elements are drawn from the supplied distribution. If the sample is 
	incomplete, the last element is not returned, so the tuple has n-1 dimensions, where
	n is the number of variables of the distribution.
	distribution: a dictionary generated by generate_distribution() or of similar layout
	randomiser: a numpy.random.RandomState
	Returns a tuple drawn from the given distribution, e.g. (2, 2)
	"""

	p = randomiser.random_sample()
	sum = 0.0
	for key, value in distribution.items():
		sum = sum + value
		if p < sum:
			return key if complete else tuple(key[i] for i in range(len(key) - 1))


def get_KL_divergence(estimate, target):
	"""
	Returns the KL divergence KL(estimate||target), where both distributions are
	a dictionary of tuple:probability pairs.
	"""
	tuples = list(estimate.keys())
	p = [estimate[k] for k in tuples]
	q = [target[k] for k in tuples]
	
	return entropy(p, q)
