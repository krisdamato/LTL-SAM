import copy 
import itertools
import matplotlib.pyplot as plt
import nest
import numpy as np
import sam.helpers as helpers
from collections import defaultdict
from sam.sam import *

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
	nest.ResetKernel()
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
	nest.ResetKernel()
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
	implicit_conditional = helpers.compute_conditional_distribution(implicit, sam.num_discrete_vals)
	target_conditional = helpers.compute_conditional_distribution(sam.distribution, sam.num_discrete_vals)
	print("Theoretical implicit distribution is {}.\nKL d. (joint) = {}\nKL d. (cond.) = {}".format(implicit, 
		helpers.get_KL_divergence(implicit, sam.distribution),
		helpers.get_KL_divergence(implicit_conditional, target_conditional)))

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


def measure_experimental_cond_distribution(sam, duration):
	"""
	Measures the conditional distribution of the provided SAM module
	experimentally, i.e. by counting output spikes directly.
	Note: For best results, stop all plasticity effects.
	"""
	var_values = range(1, sam.num_discrete_vals + 1)
	possibilities = list(itertools.product(var_values, repeat=sam.num_vars - 1))
	conditional = dict(sam.distribution)

	for p in possibilities:
		# print("Measuring experimental conditional distribution on input:", p)

		# Present evidence.
		sr = sam.connect_reader(sam.all_neurons)
		sam.clear_currents()
		sam.present_input_evidence(duration=duration, sample=p)

		# Get spikes.
		spikes = nest.GetStatus(sr, keys='events')[0]
		senders = spikes['senders']
		times = spikes['times']

		# Count spikes per output neuron.
		counts = defaultdict(int)
		for node in senders:
			counts[node] += 1 if node in sam.zeta else 0

		# Calculate conditional probabilities.
		total = np.sum(list(counts.values()))
		for z in var_values:
			conditional[p + (z,)] = counts[sam.zeta[z - 1]] / total

	return conditional


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
	nest.ResetKernel()
	sam = SAMModule(randomise_seed=False)
	sam.create_network(num_x_vars=2, 
		num_discrete_vals=2, 
		num_modes=2,
		distribution=distribution,
		params={'num_threads':1})

	# Get the conditional of the module's target distribution.
	distribution = sam.distribution
	conditional = helpers.compute_conditional_distribution(sam.distribution, 2)

	# Simulate and collect KL divergences.
	t = 0
	i = 0
	skip_kld = 10	
	skip_exp_cond = 1000
	kls_joint = []
	kls_cond = []
	kls_cond_exp = []
	set_second_rate = False
	last_set_intrinsic_rate = sam.params['first_bias_rate']
	extra_time = 0

	while t <= sam.params['learning_time'] :
		# Inject a current for some time.
		sam.present_random_sample() 
		sam.clear_currents()
		t += sam.params['sample_presentation_time']

		# Compute theoretical distributions and measure KLD.
		if i % skip_kld == 0:
			implicit = sam.compute_implicit_distribution()
			implicit_conditional = helpers.compute_conditional_distribution(implicit, 2)
			kls_joint.append(helpers.get_KL_divergence(implicit, distribution))
			kls_cond.append(helpers.get_KL_divergence(implicit_conditional, conditional))

		# Measure experimental conditional distribution from spike activity.
		if i % skip_exp_cond == 0:
			# Stop plasticity for testing.
			sam.set_intrinsic_rate(0.0)
			sam.set_plasticity_learning_time(0)
			experimental_conditional = measure_experimental_cond_distribution(sam, duration=2000.0)
			kls_cond_exp.append(helpers.get_KL_divergence(experimental_conditional, conditional))

			# Restart plasticity.
			extra_time += 2000
			sam.set_intrinsic_rate(last_set_intrinsic_rate)
			sam.set_plasticity_learning_time(int(sam.params['stdp_time_fraction'] * sam.params['learning_time'] + extra_time))
			sam.clear_currents()

		# Set different intrinsic rate.
		if t >= sam.params['learning_time'] * sam.params['intrinsic_step_time_fraction'] and set_second_rate == False:
			set_second_rate = True
			last_set_intrinsic_rate = sam.params['second_bias_rate']
			sam.set_intrinsic_rate(last_set_intrinsic_rate)
		
		i += 1

	print(sam.get_neuron_biases(sam.alpha))

	sam.set_intrinsic_rate(0.0)
	sam.set_plasticity_learning_time(0)

	# Present evidence.
	run_trained_test(sam)

	# Plot KL divergence plot.
	plt.figure()
	plt.plot(np.array(range(len(kls_cond))) * skip_kld * sam.params['sample_presentation_time'] * 1e-3, kls_cond, label="KLd p(z|x)")
	plt.plot(np.array(range(len(kls_joint))) * skip_kld * sam.params['sample_presentation_time'] * 1e-3, kls_joint, label="KLd p(x,z)")
	plt.plot(np.array(range(len(kls_cond_exp))) * skip_exp_cond * sam.params['sample_presentation_time'] * 1e-3, kls_cond_exp, label="Exp. KLd p(z|x)")
	plt.legend(loc='upper center')
	plt.show()