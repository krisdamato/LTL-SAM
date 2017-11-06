import nest
from sam import *

nest.Install('sammodule')

sam = SAMModule(randomise_seed=False)
sam.create_network(num_x_vars=2, 
	num_discrete_vals_x=2, 
	num_discrete_vals_z=2, 
	num_modes=2)
sam.inject_random_current(1000.0)