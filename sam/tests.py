import helpers
import itertools
import nest
import numpy as np
import matplotlib.pyplot as plt
from sam import *

def test_sample_draws():
	"""
	Tests whether drawing repeated samples from the distribution
	does reconstruct the target distribution approximately.
	"""
	# Use the distribution from Peceveski et al.
	distribution = {
		(1,1,1):0.04,
		(1,1,2):0.04,
		(1,2,1):0.21,
		(1,2,2):0.21,
		(2,1,1):0.04,
		(2,1,2):0.21,
		(2,2,1):0.21,
		(2,2,2):0.04
	}

	# Create module.
	sam = SAMModule(randomise_seed=True)
	sam.create_network(num_x_vars=2, 
		num_discrete_vals=2, 
		num_modes=2,
		distribution=distribution)

	# Draw repeated samples from distribution and estimate.
	test_dist = {
		(1,1,1):0,
		(1,1,2):0,
		(1,2,1):0,
		(1,2,2):0,
		(2,1,1):0,
		(2,1,2):0,
		(2,2,1):0,
		(2,2,2):0
	}

	for i in range(12000):
		t = sam.draw_random_sample()
		test_dist[t] = test_dist[t] + 1

	total = np.sum(list(test_dist.values()))
	for k, v in test_dist.items():
		test_dist[k] = v / total

	print("KL divergence from actual to estimate distribution is", helpers.get_KL_divergence(test_dist, distribution))


def run_single_random_sample(plot_iteration_number, neuron_index=0):
	"""
	Draws a single sample from the distribution, presents it to the SAM module
	and plots the spikes and voltages after a specified number of iterations.
	"""
	# Use the distribution from Peceveski et al.
	distribution = {
		(1,1,1):0.04,
		(1,1,2):0.04,
		(1,2,1):0.21,
		(1,2,2):0.21,
		(2,1,1):0.04,
		(2,1,2):0.21,
		(2,2,1):0.21,
		(2,2,2):0.04
	}

	# Create module.
	sam = SAMModule(randomise_seed=True)
	sam.create_network(num_x_vars=2, 
		num_discrete_vals=2, 
		num_modes=2,
		distribution=distribution)
	for i in range(plot_iteration_number - 1):
		sam.present_random_sample(100.0)
	mm1 = sam.connect_multimeter(sam.all_neurons[neuron_index])
	sr1 = sam.connect_reader([sam.all_neurons[neuron_index]])
	sr  = sam.connect_reader(sam.all_neurons)
	sam.present_random_sample(100.0) # Inject a current for some time.
	sam.plot_all(mm1, sr1)
	sam.plot_spikes(sr)


def run_trained_test(sam):
	"""
	Simulates the given SAM module against all input possibilites,
	counting output spikes.
	"""
	implicit = sam.compute_implicit_distribution() 
	print("Implicit distribution is {}.\nKL divergence = {}".format(implicit, helpers.get_KL_divergence(implicit, sam.distribution)))

	var_values = range(1, sam.num_discrete_vals + 1)
	possibilities = list(itertools.product(var_values, repeat=sam.num_vars - 1))

	for p in possibilities:
		print("Testing SAM on input:", p)

		# Present evidence.
		sr = sam.connect_reader(sam.all_neurons)
		sam.clear_currents()
		sam.present_input_evidence(duration=2000.0, sample=p)

		# Plot evidenced activity.
		sam.plot_spikes(sr)


def run_pecevski_experiment(plot_intermediates=False):
	"""
	Reconstructs the SAM module task from Pecevski et al 2016, 
	presenting a particular multivariate distribution as target.
	Plots intermediate results.
	"""
	# Use the distribution from Peceveski et al.
	distribution = {
		(1,1,1):0.04,
		(1,1,2):0.04,
		(1,2,1):0.21,
		(1,2,2):0.21,
		(2,1,1):0.04,
		(2,1,2):0.21,
		(2,2,1):0.21,
		(2,2,2):0.04
	}

	# Create module.
	sam = SAMModule(randomise_seed=True, num_threads=1)
	sam.create_network(num_x_vars=2, 
		num_discrete_vals=2, 
		num_modes=2,
		distribution=distribution)

	# Simulate and collect KL divergences.
	t = 0
	i = 0
	skip = 10	
	kls = []

	while t < sam.params['first_learning_phase']:
		sam.present_random_sample() # Inject a current for some time.
		sam.clear_currents()
		if i % skip == 0:
			implicit = sam.compute_implicit_distribution() 
			kls.append(helpers.get_KL_divergence(implicit, distribution))
		t += sam.params['sample_presentation_time']
		i += 1
	sam.set_intrinsic_rate(sam.params['second_bias_rate'])
	while t < (sam.params['first_learning_phase'] + sam.params['second_learning_phase']):
		sam.present_random_sample() # Inject a current for some time.
		sam.clear_currents()
		if i % skip == 0:
			implicit = sam.compute_implicit_distribution() 
			kls.append(helpers.get_KL_divergence(implicit, distribution))
		t += sam.params['sample_presentation_time']
		i += 1

	print(sam.get_neuron_biases(sam.alpha))

	sam.set_intrinsic_rate(0.0)
	sam.set_plasticity_learning_time(0)

	# Present evidence.
	run_trained_test(sam)

	# Plot KL divergence plot.
	plt.figure()
	plt.plot(np.array(range(len(kls))) * skip * sam.params['sample_presentation_time'] * 1e-3, kls)
	plt.show()