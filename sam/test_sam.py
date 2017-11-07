import nest
from sam import *

nest.Install('sammodule')

sam = SAMModule(randomise_seed=True)
sam.create_network(num_x_vars=2, 
	num_discrete_vals=2, 
	num_modes=2)
sr = sam.connect_reader()
for i in range(1200):
	sam.present_random_sample(100.0) # Inject a current for some time.
	sam.clear_currents()
nest.Simulate(10000.0) # Simulate for some more time.
sam.plot_spikes(sr)