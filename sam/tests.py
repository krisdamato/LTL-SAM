import nest
from sam import *

def test_sample_draws():
	"""
	Tests whether drawing repeated samples from the distribution
	does reconstruct the target distribution approximately.
	"""
	nest.Install('sammodule')

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

	for i in range(10000):
		t = sam.draw_random_sample()
		test_dist[t] = test_dist[t] + 1

	total = np.sum(list(test_dist.values()))
	for k, v in test_dist.items():
		test_dist[k] = v / total

	print(test_dist)


def run_single_random_sample(plot_iteration_number, neuron_index=0):
	"""
	Draws a single sample from the distribution, presents it to the SAM module
	and plots the spikes and voltages after a specified number of iterations.
	"""
	nest.Install('sammodule')
	nest.SetKernelStatus({'resolution':0.01})

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


def run_pecevski_experiment(plot_intermediates=False):
	"""
	Reconstructs the SAM module task from Pecevski et al 2016, 
	presenting a particular multivariate distribution as target.
	Plots intermediate results.
	"""
	nest.Install('sammodule')
	#nest.SetKernelStatus({'resolution':0.01})

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

	# Simulate.
	for i in range(6000):
		if i == 3000 and plot_intermediates:
			mm1 = sam.connect_multimeter(sam.alpha[0])
			mm2 = sam.connect_multimeter(sam.alpha[1])
			sr1 = sam.connect_reader(sam.all_neurons)
		sam.present_random_sample(100.0) # Inject a current for some time.
		sam.clear_currents()
		if i == 3000 and plot_intermediates:
			sam.plot_potential_and_bias(mm1)
			sam.plot_potential_and_bias(mm2)
			sam.plot_spikes(sr1)
	sam.set_intrinsic_rate(0.02)
	for i in range(6000):
		if i == 3000 and plot_intermediates:	
			mm1 = sam.connect_multimeter(sam.alpha[0])
			mm2 = sam.connect_multimeter(sam.alpha[1])
			sr1 = sam.connect_reader(sam.all_neurons)
		sam.present_random_sample(100.0) # Inject a current for some time.
		sam.clear_currents()
		if i == 3000 and plot_intermediates:
			sam.plot_potential_and_bias(mm1)
			sam.plot_potential_and_bias(mm2)
			sam.plot_spikes(sr1)

	sam.set_intrinsic_rate(0.0)
	sam.set_plasticity_learning_time(0)

	# Present evidence.
	sr = sam.connect_reader(sam.all_neurons)
	sam.clear_currents()
	sam.present_input_evidence(duration=2000.0, sample=(2,2,1))

	# Plot spontaneous activity.
	sam.plot_spikes(sr)