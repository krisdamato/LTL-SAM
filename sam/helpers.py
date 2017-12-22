import matplotlib.pyplot as plt
import itertools
import numpy as np
import re
import os
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import entropy
from time import gmtime, strftime


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


def get_KL_divergence(estimate, target, min_prob=1e-5, exclude_invalid_states=False):
	"""
	Returns the KL divergence KL(estimate||target), where both distributions are
	a dictionary of tuple:probability pairs.
	"""
	# Take the union of all keys (the estimate distribution may be missing some
	# possibilities).
	tuples = target.keys() if exclude_invalid_states else list(set().union(estimate.keys(), target.keys())) 
	p = [target[k] if k in target else 0.0 for k in tuples]
	q = [max(estimate[k], min_prob) if k in estimate else min_prob for k in tuples]
	
	return entropy(p, q)


def compute_joint_distribution(eqn, num_discrete_values, *dists):
	"""
	Given a decomposition of a joint distribution, e.g. "p(y1,y2,y3) = p(y1)p(y2)p(y3|y1,y2)",
	this computes the full joint distribution dictionary with all possibilities. The equation
	must be passed as a string, using '|' to indicate conditionality and ',' between variables.
	The passed distributions must correspond to the order on the RHS of the equation. 
	Note: this assumes that the order of RVs in the passed distributions corresponds to the 
	variable indices, e.g. "p(y3|y1,y2)" assumes that the tuples in the passed dictionary 
	of this distribution are ordered as (y1,y2,y3). Also, it is assumed that the distributions 
	match the type and number of variables indicated in the equation (e.g. "p(y3|y1,y2)" is a 
	3-variable conditional distribution).
	"""
	# Parse string.
	split = eqn.replace(' ', '').split('=')
	lhs = split[0]
	rhs = split[1]

	# Find the list of variables.
	vars = sorted(re.search(r'\((.*?)\)', lhs).group(1).split(','))

	# Find all distributions on the RHS, and their variables.
	rhs_ps = re.findall(r'\((.*?)\)', rhs)
	rhs_vars = [sorted(p.replace('|', ',').split(',')) for p in rhs_ps]

	# For each possibility on the LHS, work out the probability using the passed distribtions.
	var_values = range(1, num_discrete_values + 1)
	possibilities = list(itertools.product(var_values, repeat=len(vars)))
	joint = {}
	for p in possibilities:
		# Work out the probabilities of each RHS distribution for this combination of RV values.
		probability = 1.0
		for dist_vars, dist in zip(rhs_vars, dists):
			# Find the RV tuple indices of the RVs in this dist.
			indices = [int(s[1:]) - 1 for s in dist_vars]

			# Find the RVs at those indices.
			rvs = tuple([p[i] for i in indices])

			# Find the probability at those RVs in this distribution.
			probability *= dist[rvs]

		joint[p] = probability

	return joint


def compute_conditional_distribution(joint, num_discrete_values, dep_index=-1):
	"""
	Given a joint distribution p(x1, x2, ..., xn) in the form of tuple:probability pairs, 
	this computes the conditional distribution p(xi|x1, ..., xi-1, xi+1, ..., xn). xi is 
	assumed to be the last tuple element, with values in {1, ..., num_discrete_values} if 
	the index is not supplied.
	"""
	conditional = dict(joint)
	num_x_vars = len(list(joint.keys())[0]) - 1
	var_values = range(1, num_discrete_values + 1)
	x_possibilities = list(itertools.product(var_values, repeat=num_x_vars))
	dep_index = dep_index if dep_index != -1 else num_x_vars

	for p in x_possibilities:
		p_prepend = tuple([p[i] for i in range(dep_index)])
		p_append = tuple([p[i] for i in range(dep_index, num_x_vars)])
		search_p = [p_prepend + (z, ) + p_append for z in range(1, num_discrete_values + 1)]

		total = 0
		for k, v in joint.items():
			if k in search_p:
				total += v
		for k in search_p:
			conditional[k] /= total 

	return conditional


def compute_marginal_distribution(joint, keep, num_discrete_values):
	"""
	Given a full joint distribution p(y1, ..., yn) in the form of RV tuple:probability pairs,
	this computes the marginal distribution p(yk1, yk2, ..., ykn), where [yk1, ..., ykn] are
	specified in keep as "yi" strings, e.g. "y1". Returns a distribution with the same format
	as the joint distribution, preserving order between variables, except that only a subset
	of variables is retained.
	NOTE: This assumes that the first variable is called "y1", not "y0".
	"""
	# Find which variables to marginalize.
	keep_indices = [int(s[1:]) - 1 for s in keep]
	num_vars = len(list(joint.keys())[0])
	var_indices = list(range(num_vars))
	remove_indices = list(set(var_indices) - set(keep_indices))

	# Create a multi-dimensional array from the joint distribution.
	shape = [num_discrete_values] * num_vars
	joint_array = np.empty(shape)
	for t, p in joint.items():
		# We have to subtract 1 from the tuple components, since values in
		# the distribution start at 1.
		t = tuple(np.array(t) - 1)
		joint_array[t] = p

	# Sum over the distribution to remove all marginalized variables.
	marginal_array = np.sum(joint_array, axis=tuple(remove_indices))

	# Convert the array into a dictionary of tuple:probability pairs.
	marginal = dict()
	var_values = range(1, num_discrete_values + 1)
	possibilities = list(itertools.product(var_values, repeat=len(keep_indices)))
	for p in possibilities:
		t = tuple(np.array(p) - 1)
		marginal[p] = marginal_array[t]

	return marginal


def plot_3d_histogram(target, estimated, num_discrete_values, target_label, estimated_label):
	"""
	Assumes that there are three RVs in each distribution that can take
	values in {1, ..., num_discrete_values} and makes a plot for each 
	value of the last variable.
	Note: does not show to screen.
	"""
	fig = plt.figure(figsize=(6, 8))
	ax1 = fig.add_subplot(211, projection='3d')
	ax2 = fig.add_subplot(212, projection='3d') 

	# Generate grid matrices.
	_x = np.arange(1, num_discrete_values + 1)
	_y = np.arange(1, num_discrete_values + 1)
	_xx, _yy = np.meshgrid(_x, _y)
	x, y = _xx.ravel(), _yy.ravel()

	top1e = [estimated[(i, j, 1)] for j in _y for i in _x]
	top2e = [estimated[(i, j, 2)] for j in _y for i in _x]
	top1t = [target[(i, j, 1)] for j in _y for i in _x]
	top2t = [target[(i, j, 2)] for j in _y for i in _x]

	bottom = np.zeros_like(top1e)
	width = depth = 0.15

	green_proxy = plt.Rectangle((0, 0), 1, 1, fc='g')
	blue_proxy = plt.Rectangle((0, 0), 1, 1, fc='b')

	ax1.bar3d(x - width, y, bottom, width, depth, top1e, shade=True, color='b')
	ax1.bar3d(x, y, bottom, width, depth, top1t, shade=True, color='g')
	ax1.view_init(elev=35, azim=-65)
	ax1.grid(False)
	ax1.set_xticks(_x)
	ax1.set_yticks(_y)
	ax1.set_title('z = 1')
	ax1.legend([green_proxy, blue_proxy], [target_label, estimated_label])

	ax2.bar3d(x - width, y, bottom, width, depth, top2e, shade=True, color='b')
	ax2.bar3d(x, y, bottom, width, depth, top2t, shade=True, color='g')
	ax2.view_init(elev=35, azim=-65)
	ax2.grid(False)
	ax2.set_xticks(_x)
	ax2.set_yticks(_y)
	ax2.set_title('z = 2')
	ax2.legend([green_proxy, blue_proxy], [target_label, estimated_label])

	return fig


def plot_histogram(target, estimated, num_discrete_values, target_label, estimated_label, renormalise_estimated_states=True):
	"""
	Given a distribution in the form of a dictionary, this plots a histogram
	of states, labeled by state vector.
	Note: does not show to the screen.
	"""
	fig = plt.figure(figsize=(18, 3))
	ax = fig.add_subplot(111)

	# Renormalise legal states.
	truncated = {k:estimated[k] for k in target}
	if renormalise_estimated_states:
		total = np.sum(list(truncated.values()))
		for k in truncated:
			truncated[k] /= total

	# Get the states in order.
	ordered_target = OrderedDict(sorted(target.items()))
	ordered_truncated = OrderedDict(sorted(truncated.items()))

	width = 0.3
	x = np.arange(len(ordered_target))
	ax.bar(x, ordered_truncated.values(), width, color='g')
	ax.bar(x + width, ordered_target.values(), width, color='black')
	ax.set_xticks(x + width / 2)
	ax.set_xticklabels(ordered_target.keys(), rotation=45)
	ax.set_ylabel('p')
	plt.tight_layout()

	return fig


def plot_all(multimeter, spikereader):
	"""
	Plots the spike trace and voltage traces of a single neuron on the same figure.
	"""
	# Get spikes and plot.
	spikes = nest.GetStatus(spikereader, keys='events')[0]
	senders = spikes['senders']
	times = spikes['times']
	dmm = nest.GetStatus(multimeter)[0]
	Vms = dmm["events"]["V_m"]
	bias = dmm["events"]["bias"]
	ts = dmm["events"]["times"]

	plt.figure()
	plt.plot(ts, Vms)
	plt.plot(ts, bias)
	plt.plot(times, senders, '|')
	plt.show()


def plot_spikes(self, spikereader):
	"""
	Plots the spike trace from all neurons the spikereader was connected to during
	the simulation.
	"""
	# Get spikes and plot.
	spikes = nest.GetStatus(spikereader, keys='events')[0]
	senders = spikes['senders']
	times = spikes['times']

	# Plot
	plt.figure()
	plt.plot(times, senders, '|')
	plt.show()


def get_dictionary_string(d):
	"""
	Outputs a nicely formatted string with a dictionary's content.
	"""
	dict_string = ""
	for k, v in d.items():
		dict_string += "{}: {}\n".format(k,v)

	return dict_string


def create_directory(directory):
	"""
	Creates a directory if it does not exist.
	"""
	if not os.path.exists(directory):
		os.makedirs(directory)


def save_text(text, path):
	"""
	Creates a text file at specified path and saves the provided
	text to it.
	"""
	f = open(path,'w')
	f.write(text)
	f.close()


def get_now_string():
	"""
	Returns a string of the current timestamp.
	"""
	return strftime("%Y-%m-%d %H:%M:%S", gmtime())