import numpy as np

def determine_bias_baseline(T, largest_values):
	'''
	Finds the baseline bias parameter value for the intrinsic plasticity rule, 
	which is used to shift bias update sizes.
	largest_values: a list of maximal values of each predictor in this SAM.
	'''

	num_predictors = len(largest_values)
	return (2.5 / T) * (-num_predictors * np.log(0.2) - np.sum([np.log(m + 1) for m in largest_values]) + np.log(0.02))
