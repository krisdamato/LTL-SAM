import nest
from sam import *

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

# Simulate.
for i in range(6000):
	sam.present_random_sample(100.0) # Inject a current for some time.
sam.set_intrinsic_rate(0.02)
for i in range(6000):
	sam.present_random_sample(100.0) # Inject a current for some time.

sam.set_intrinsic_rate(0.0)
sam.set_plasticity_learning_time(0)

# Present evidence.
sr = sam.connect_reader(sam.all_neurons)
sam.clear_currents()
sam.present_input_evidence(duration=2000.0, sample=(2,2,2))

# Run without input.
#sam.simulate_without_input(20000.0) # Simulate for some more time.

# Plot spontaneous activity.
sam.plot_spikes(sr)