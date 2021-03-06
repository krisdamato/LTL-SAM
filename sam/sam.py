import copy
import logging
import nest
import numpy as np
import pylab
import sam.helpers as helpers
import time
from collections import defaultdict
from mpi4py import MPI


class SAMModule:
	"""
	Encapsulates a Stochastic Association Module (SAM), as described in Pecevski et al, 2016. A SAM is a winner-take-all module
	that performs unsupervised learning of a target multinomial distribution (density estimation). Although in the sense of 
	density estimation it is an unsupervised algorithm (during the learning phase samples are simply presented from the 
	target distribution, with no labelling or splitting into input/output), the module learns the structure of variable 
	interdependencies, so that withholding one variable and presenting samples from the rest to the module causes it to 
	estimate the target conditional probability of the withheld variable.
	"""

	def __init__(self, randomise_seed=False):
		"""
		Initialises basic parameters.
		"""
		self.seed = int(time.time()) if randomise_seed else 705
		self.initialised = False


	@staticmethod
	def parameter_spec():
		"""
		Returns a dictionary of param_name:(min, max) pairs, which describe the legal
		limit of the parameters.
		"""
		param_spec = {
			'initial_stdp_rate':(0.0, 0.01),
			'final_stdp_rate':(0.0, 0.01),
			'weight_baseline':(-10.0, 0.0),
			'T':(0.0, 1.0),
			'first_bias_rate':(0.0, 0.1),
			'second_bias_rate':(0.0, 0.1),
			'bias_baseline':(-40.0, 0.0),
			'exp_term_prob':(0.0, 1.0),
			'exp_term_prob_scale':(0.0, 5.0),
			'relative_bias_spike_rate':(1e-5, 1.0)
			}

		return param_spec


	def set_sam_defaults(self, params={}):
		"""
		Sets SAM defaults to Pecevski et al. 2016 values.
		params: dictionary of parameters to set.
		"""
		# Set common properties.
		tau = 15.0 if 'tau' not in params else params['tau']
		delay = 0.05 if 'delay' not in params else params['delay']

		# Set all derived and underived properties.
		self.params = {
			'initial_stdp_rate':0.002,
			'final_stdp_rate':0.0006,
			'stdp_time_fraction':0.5,
			'intrinsic_step_time_fraction':0.5,
			'initial_weight':3.0,
			'max_weight':4.0,
			'min_weight':0.0,
			'weight_baseline':2.5 * np.log(0.2),
			'tau':tau,
			'delay':delay,
			'T':0.58,
			'use_rect_psp_exc':True,
			'use_rect_psp_inh':True,
			'inhibitors_use_rect_psp_exc':True,
			'amplitude_exc':2.0,
			'amplitude_inh':2.0,
			'external_current':0.0,
			'tau_membrane':delay/100,
			'tau_alpha':8.5,
			'first_bias_rate':0.01,
			'second_bias_rate':0.02,
			'relative_bias_spike_rate':0.02,
			'bias_baseline':0.0,
			'max_bias':5.0,
			'min_bias':-30.0,
			'default_random_bias':True,
			'initial_bias':5.0,
			'dead_time_random':False,
			'linear_term_prob':0.0,
			'exp_term_prob':0.5/tau,
			'exp_term_prob_scale':1.0,
			'weight_chi_alpha_mean': 4.0 / 3,
			'weight_chi_alpha_std':0.1,
			'weight_alpha_inhibitors':80.0,
			'weight_alpha_zeta':20.0,
			'weight_inhibitors_alpha':-7.0,
			'bias_inhibitors':-10.0,
			'bias_chi':-10.0,
			'bias_zeta':-10.0,
			'bias_alpha_mean':5.0,
			'bias_alpha_std':0.1,
			'nu_current_plus':30.0,
			'nu_current_minus':-30.0,
			'alpha_current_minus':-80.0,
			'alpha_current_plus':0.0,
			'learning_time':300000,
			'sample_presentation_time':100.0, # 100 ms.
			'chi_neuron_type':'srm_pecevski_alpha',
			'alpha_neuron_type':'srm_pecevski_alpha',
			'zeta_neuron_type':'srm_pecevski_alpha',
			'inhibitors_neuron_type':'srm_pecevski_alpha',
			'chi_alpha_synapse_type':'stdp_pecevski_synapse',
			'alpha_zeta_synapse_type':'static_synapse',
			'alpha_inhibitors_synapse_type':'static_synapse',
			'inhibitors_alpha_synapse_type':'static_synapse',
			'num_inhibitors':5,
			'delay_alpha_inhibitors':delay,
			'delay_inhibitors_alpha':delay,
			'delay_chi_alpha':delay,
			'delay_alpha_zeta':delay,
			'devices_delay':delay,
			'max_depress_tau_multiplier':100000.0,
			'use_renewal':False
		}

		# Update defaults.
		self.params.update(params)


	def set_nest_defaults(self):
		"""
		Clears and sets the NEST defaults from the parameters.
		NOTE: Any change of the network parameters needs a corresponding call to
		this function in order to update settings.
		"""
		# Set NEST defaults.
		n = nest.GetKernelStatus(['total_num_virtual_procs'])[0]

		logging.info("Randomising {} processes using {} as main seed.".format(n * 2 + 1, self.seed))

		# Seed Python, global, and per-process RNGs.
		self.rngs = [np.random.RandomState(s) for s in range(self.seed, self.seed + n)]
		nest.SetKernelStatus({
			'grng_seed' : self.seed + n,
			'rng_seeds' : range(self.seed + n + 1, self.seed + 2 * n + 1)
			})

		# Reduce NEST verbosity.
		nest.set_verbosity('M_ERROR')
		nest.SetDefaults('static_synapse', params={'weight':self.params['initial_weight']})

		nest.SetDefaults('stdp_pecevski_synapse', params={
			'eta_0':self.params['initial_stdp_rate'],
			'eta_final':self.params['final_stdp_rate'], 
			'learning_time':int(self.params['stdp_time_fraction'] * self.params['learning_time']),
			'max_weight':self.params['max_weight'],
			'min_weight':self.params['min_weight'],
			'weight':self.params['initial_weight'],
			'w_baseline':self.params['weight_baseline'],
			'tau':self.params['tau'],
			'T':self.params['T'],
			'depress_multiplier':self.params['max_depress_tau_multiplier']
			})

		nest.SetDefaults('srm_pecevski_alpha', params={
			'rect_exc':self.params['use_rect_psp_exc'], 
			'rect_inh':self.params['use_rect_psp_inh'],
			'e_0_exc':self.params['amplitude_exc'], 
			'e_0_inh':self.params['amplitude_inh'],
			'I_e':self.params['external_current'],
			'tau_m':self.params['tau_membrane'], # Membrane time constant (only affects current injections). Is this relevant?
			'tau_exc':self.params['tau'],
			'tau_inh':self.params['tau'], 
			'tau_bias':self.params['tau'],
			'eta_bias':0.0, # eta_bias is set manually.
			'rel_eta':self.params['relative_bias_spike_rate'],
			'b_baseline':self.params['bias_baseline'],
			'max_bias':self.params['max_bias'],
			'min_bias':self.params['min_bias'],
			'bias':self.params['initial_bias'],
			'dead_time':self.params['tau'], # Abs. refractory period.
			'dead_time_random':self.params['dead_time_random'],
			'c_1':self.params['linear_term_prob'], # Linear part of transfer function.
			'c_2':self.params['exp_term_prob'], # The coefficient of the exponential term in the transfer function.
			'c_3':self.params['exp_term_prob_scale'], # Scaling coefficient of effective potential in exponential term.
			'T':self.params['T'],
			'use_renewal':self.params['use_renewal']
			})

		nest.SetDefaults('stdp_synapse', params={
			'tau_plus':self.params['tau_alpha'],
			'Wmax':self.params['max_weight'],
			'mu_plus':0.0,
			'mu_minus':0.0,
			'lambda':self.params['final_stdp_rate'],
			'weight':self.params['initial_weight']
			})

		# Create layer model specialisations.
		# Input population model.
		if self.params['chi_neuron_type'] == 'srm_pecevski_alpha':
			self.chi_neuron_params={'bias':self.params['bias_chi']}
		else:
			raise NotImplementedError

		# Hidden population model.
		if self.params['alpha_neuron_type'] == 'srm_pecevski_alpha':
			# We will set alpha neuron biases during network construction.
			self.alpha_neuron_params={'eta_bias':self.params['first_bias_rate']}
		else:
			raise NotImplementedError

		# Output population model.
		if self.params['zeta_neuron_type'] == 'srm_pecevski_alpha':
			self.zeta_neuron_params={'bias':self.params['bias_zeta']}
		else:
			raise NotImplementedError

		# Inhibitor population model.
		if self.params['inhibitors_neuron_type'] == 'srm_pecevski_alpha':
			self.inhibitors_neuron_params={
				'bias':self.params['bias_inhibitors'],
				'rect_exc':self.params['inhibitors_use_rect_psp_exc'],
				'tau_exc':self.params['tau']
				}
		else:
			raise NotImplementedError

		# Create synapse model specialisations.
		# Input-alpha synapses.
		if self.params['chi_alpha_synapse_type'] == 'stdp_pecevski_synapse':
		 	self.chi_alpha_synapse_params={
		 		'model':'stdp_pecevski_synapse',
				'weight':{
					'distribution':'normal_clipped',
					'low':self.params['min_weight'],
					'high':self.params['max_weight'],
					'mu':self.params['weight_chi_alpha_mean'],
					'sigma':self.params['weight_chi_alpha_std']
					},
				'delay':self.params['delay_chi_alpha']
				}
		elif self.params['chi_alpha_synapse_type'] == 'stdp_synapse':
			self.chi_alpha_synapse_params={
		 		'model':'stdp_synapse',
				'weight':{
					'distribution':'normal_clipped',
					'low':self.params['min_weight'],
					'high':self.params['max_weight'],
					'mu':self.params['weight_chi_alpha_mean'],
					'sigma':self.params['weight_chi_alpha_std']
					},
				'delay':self.params['delay_chi_alpha']
				}
		else:
			raise NotImplementedError

		# Alpha-inhibitors synapses.
		if self.params['alpha_inhibitors_synapse_type'] == 'static_synapse':
			self.alpha_inhibitors_synapse_params={
				'model':'static_synapse',
				'weight':self.params['weight_alpha_inhibitors'],
				'delay':self.params['delay_alpha_inhibitors']
				}
		else:
			raise NotImplementedError

		# Inhibitors-alpha synapses.
		if self.params['inhibitors_alpha_synapse_type'] == 'static_synapse':
			self.inhibitors_alpha_synapse_params={
				'model':'static_synapse',
				'weight':self.params['weight_inhibitors_alpha'],
				'delay':self.params['delay_inhibitors_alpha']
				}
		else:
			raise NotImplementedError

		# Alpha-zeta synapses.
		if self.params['alpha_zeta_synapse_type'] == 'static_synapse':
			self.alpha_zeta_synapse_params={
				'model':'static_synapse',
				'weight':self.params['weight_alpha_zeta'],
				'delay':self.params['delay_alpha_zeta']
				}
		else:
			raise NotImplementedError


	def create_network(self, num_x_vars=2, num_discrete_vals=2, num_modes=2, distribution=None, dep_index=-1, params={}, create_input_layer=True):
		"""
		Creates a SAM module with the specified architecture.
		distribution: Users can specify the distribution themselves, in which case they need to take care that the 
			other parameters are within the constraints set by this distribution (e.g. num_x_vars).
		dep_index: Specifies the index (within the distribution tuples) of the output variable (-1 = last).
		
		"""
		# Set parameter defaults.
		self.set_sam_defaults(params)

		# Set bias baseline if not given externally.
		if 'bias_baseline' not in params.keys():
			self.params['bias_baseline'] = helpers.determine_bias_baseline(self.params['T'], [num_discrete_vals for i in range(num_x_vars)])
			logging.info("Setting bias baseline to {}".format(self.params['bias_baseline']))

		# Set NEST defaults. 
		# Note: This has to happen after SAM defaults are set.
		self.set_nest_defaults()
			
		# Create input layer if the module is standalone.
		if create_input_layer:
			self.created_input_layer = True
			self.chi = nest.Create(self.params['chi_neuron_type'], n=num_discrete_vals * num_x_vars, params=self.chi_neuron_params)
		else:
			self.chi = () # Needs to be set separately.

		# Create other layers.
		self.alpha = nest.Create(self.params['alpha_neuron_type'], n=num_modes * num_discrete_vals, params=self.alpha_neuron_params)
		self.zeta = nest.Create(self.params['zeta_neuron_type'], n=num_discrete_vals, params=self.zeta_neuron_params)
		self.inhibitors = nest.Create(self.params['inhibitors_neuron_type'], n=self.params['num_inhibitors'], params=self.inhibitors_neuron_params)

		# Track all neurons.
		self.all_neurons = self.chi + self.alpha + self.zeta + self.inhibitors

		# Connect layers.
		# Chi-alpha connectivity: all-to-all.
		if create_input_layer:
			nest.Connect(self.chi, self.alpha, conn_spec='all_to_all', syn_spec=self.chi_alpha_synapse_params)

		# Alpha-inhibitors connectivity: all-to-all.
		nest.Connect(self.alpha, self.inhibitors, conn_spec='all_to_all', syn_spec=self.alpha_inhibitors_synapse_params)

		# Inhibitors-alpha connectivity: all-to-all.
		nest.Connect(self.inhibitors, self.alpha, conn_spec='all_to_all', syn_spec=self.inhibitors_alpha_synapse_params)

		# Alpha bias randomisation.
		self.set_random_biases(self.alpha, 
			self.params['bias_alpha_mean'], 
			self.params['bias_alpha_std'],
			self.params['min_bias'],
			self.params['max_bias'])

		# Alpha-zeta connectivity: subpopulation-to-subpopulation-index.
		for i in range(len(self.alpha)):
			j = i // num_modes
			nest.Connect([self.alpha[i]], [self.zeta[j]], syn_spec=self.alpha_zeta_synapse_params)

		# If no distribution has been passed, generate one using the passed parameters.
		self.output_index = dep_index if dep_index >= 0 else num_x_vars
		self.distribution = distribution if distribution is not None else self.generate_distribution(num_x_vars + 1, num_discrete_vals)
		if len(self.distribution) != pow(num_discrete_vals, num_x_vars + 1):
			raise Exception("The number of variables in the distribution and parameters must match. " + 
				"Given a distribution of length {} but {} combinations of values and variables.".format(len(self.distribution), pow(num_discrete_vals, num_x_vars + 1)))

		# Set other flags/metainfo.
		self.num_vars = num_x_vars + 1
		self.num_discrete_vals = num_discrete_vals
		self.num_modes = num_modes

		# The network is initialised only if it its fully constructed.
		self.initialised = True if create_input_layer else False


	def set_input_layer(self, nodes):
		"""
		If the module uses output from other modules as its input layer, the existing neuron indices can be passed 
		here. The user must take care to pass the neurons in the right order. Overwrites any existing input neurons
		if already specified and connects new input layer to alpha layer.
		"""
		if len(nodes) != self.num_discrete_vals * (self.num_vars - 1):
			raise Exception("The number of input neurons must match the network parameters. Provided with an array " +
				"of pre-existing neurons of length {}".format(len(nodes)))
		
		self.uses_preexisting_input = True
		self.chi = nodes

		# Reset the all_neurons aggregate.
		self.all_neurons = self.chi + self.alpha + self.zeta + self.inhibitors

		# Connect chi to alpha neurons: all-to-all.
		nest.Connect(self.chi, self.alpha, conn_spec='all_to_all', syn_spec=self.chi_alpha_synapse_params)

		# Only now is the network fully initialised.
		self.initialised = True


	def clone(self):
		"""
		Returns a network with the same architecture and equal weights and biases.
		Note: We pass the params to create_network because the function overwrites 
		the bias baseline unless the bias baseline is passed. Also, note that
		if the network was built using pre-existing inputs, it will not clone
		the input layer, but will instead leave the input layer undefined and
		unconnected.
		"""
		new_module = copy.deepcopy(self)
		new_module.create_network(
			num_x_vars=self.num_vars - 1,
			num_discrete_vals=self.num_discrete_vals,
			num_modes=self.num_modes,
			distribution=self.distribution,
			dep_index=self.output_index,
			params=self.params,
			create_input_layer=self.created_input_layer)

		# Flag the module uninitialised if the input layer was provided externally.
		if not self.created_input_layer:
			new_module.initialised = False
		
		# Copy bias values.
		if new_module.params['alpha_neuron_type'] == 'srm_pecevski_alpha':
			alpha_biases = nest.GetStatus(self.alpha, 'bias')
			nest.SetStatus(new_module.alpha, 'bias', alpha_biases)

		# Copy synapse weights.
		if self.created_input_layer:
			chi_alpha_conns = nest.GetConnections(self.chi, self.alpha)
			chi_alpha_weights = nest.GetStatus(chi_alpha_conns, 'weight')
			chi_alpha_conns_new = nest.GetConnections(new_module.chi, new_module.alpha)
			nest.SetStatus(chi_alpha_conns_new, 'weight', chi_alpha_weights)

		return new_module


	def copy_dynamic_properties(self, module):
		"""
		Copies the dynamic properties such as weights and biases from the
		specified module. 
		Note: Assumes both modules are fully constructed and initialised.
		"""
		# Copy all biases.
		if module.params['chi_neuron_type'] == 'srm_pecevski_alpha':
			biases = nest.GetStatus(module.chi, 'bias')
			nest.SetStatus(self.chi, 'bias', biases)

		if module.params['alpha_neuron_type'] == 'srm_pecevski_alpha':
			biases = nest.GetStatus(module.alpha, 'bias')
			nest.SetStatus(self.alpha, 'bias', biases)

		if module.params['zeta_neuron_type'] == 'srm_pecevski_alpha':
			biases = nest.GetStatus(module.zeta, 'bias')
			nest.SetStatus(self.zeta, 'bias', biases)

		# Copy all weights.
		conns = nest.GetConnections(module.all_neurons, module.all_neurons)
		weights = nest.GetStatus(conns, 'weight')
		conns_new = nest.GetConnections(self.all_neurons, self.all_neurons)
		nest.SetStatus(conns_new, 'weight', weights)


	def set_bias_baseline(self, baseline):
		"""
		Sets the alpha layer bias baseline.
		"""
		if not self.initialised:
			raise Exception("SAM module not initialised yet.")

		# Update alpha layer baselines.
		if self.params['alpha_neuron_type'] == 'srm_pecevski_alpha':
			logging.info("Setting bias baseline to {}".format(self.params['bias_baseline']))
			nest.SetStatus(self.alpha, {'b_baseline':baseline})
		else:
			logging.warning("Cannot set a bias baseline in neurons that don't support it. Returning with no effect.")


	def get_local_nodes(self, neurons):
		"""
		Returns a list of (node ID, process ID) pairs for all nodes that were
		created by this process.
		"""
		node_info = nest.GetStatus(neurons)
		return [(ni['global_id'], ni['vp']) for ni in node_info if ni['local']]


	def are_nodes_local(self, neurons):
		"""
		Returns a list of True/False values for all nodes, indicating whether the 
		nodes are local.
		"""
		node_info = nest.GetStatus(neurons)
		return [ni['local'] for ni in node_info]


	def generate_distribution(self, num_vars, num_discrete_vals):
		"""
		Generates a distribution randomly (see helpers documentation). This uses
		the rank 0 process RNG to generate the random numbers.
		"""
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()

		# Broadcast the distribution to all MPI processes.
		if rank == 0:
		    dist = helpers.generate_distribution(num_vars, num_discrete_vals, self.rngs[0])
		    logging.info("Process 0 generated distribution: %s", dist)
		else:
			dist = None
			logging.info("Process {} receiving distribution.".format(rank))

		comm.bcast(dist, root=0)

		return dist


	def draw_random_sample(self):
		"""
		Uses the rank 0 process to draw a random sample from the target distribution.
		See the documentation in helpers for more details on how this works.
		"""
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()

		# Broadcast the sample to all MPI processes.
		if rank == 0:
		    sample = helpers.draw_from_distribution(self.distribution, complete=True, randomiser=self.rngs[0])
		else:
			sample = None

		comm.bcast(sample, root=0)

		return sample


	def draw_normal_values(self, n, mu, sigma, min, max):
		"""
		Uses the rank 0 process to draw a random sequence of numbers from a normal
		distribution with the specified parameters. 
		"""
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()

		# Broadcast the sample to all MPI processes.
		if rank == 0:
			# Do this element by element, since numpy does not offer trunc norm.
			sample = []
			while len(sample) < n:
				r = self.rngs[0].normal(mu, sigma)
				if r >= min and r <= max:
					sample.append(r)
		else:
			sample = None

		comm.bcast(sample, root=0)

		return sample


	def set_random_biases(self, nodes, mu, sigma, min, max):
		"""
		Sets the biases in the passed neurons to a random value drawn from 
		a normal distribution with the specificed mean and std.
		Note: Assumes that the neurons support bias.
		"""
		normal_biases = self.draw_normal_values(len(nodes), mu, sigma, min, max)	
		locals = self.are_nodes_local(nodes)
		for i in range(len(nodes)):
			if locals[i]:
				nest.SetStatus([nodes[i]], {'bias':normal_biases[i]})


	def set_input_currents(self, state):
		"""
		Inhibits/forces the input layer neurons according to the provided state.
		state: a tuple containing the values to be encoded by the input population
		code, one value for each subpopulation of chi neurons.
		Note: This function is not abstracted away because SAMGraph needs access
		to currents layer-by-layer.
		"""
		locals = self.are_nodes_local(self.chi)
		for i in range(len(self.chi)):
			var_index = i // self.num_discrete_vals
			node_value = (i % self.num_discrete_vals) + 1 # The + 1 is necessary since the neuron values are 1-offset.
			inhibit = state[var_index] != node_value
			if locals[i]:
				current = self.params['nu_current_minus'] if inhibit else self.params['nu_current_plus']
				nest.SetStatus([self.chi[i]], {'I_e':current})


	def set_hidden_currents(self, value):
		"""
		Inhibits/forces the alpha layer neurons according to the provided value.
		value: a value to be encoded by the output population.
		Note: This function is not abstracted away because SAMGraph needs access
		to currents layer-by-layer.
		"""
		locals = self.are_nodes_local(self.alpha)
		for i in range(len(self.alpha)):
			node_value = (i // self.num_modes) + 1 # The + 1 is necessary since the neuron values are 1-offset.
			inhibit = value != node_value
			if locals[i]:
				current = self.params['alpha_current_minus'] if inhibit else self.params['alpha_current_plus']
				nest.SetStatus([self.alpha[i]], {'I_e':current})


	def set_output_currents(self, value):
		"""
		Inhibits/forces the output layer neurons according to the provided value.
		value: a value to be encoded by the output population.
		Note: This function is not abstracted away because SAMGraph needs access
		to currents layer-by-layer.
		"""
		locals = self.are_nodes_local(self.zeta)
		for i in range(len(self.zeta)):
			node_value = i + 1 # The + 1 is necessary since the neuron values are 1-offset.
			inhibit = value != node_value
			if locals[i]:
				current = self.params['nu_current_minus'] if inhibit else self.params['nu_current_plus']
				nest.SetStatus([self.zeta[i]], {'I_e':current})


	def set_currents(self, state, inhibit_alpha=False):
		"""
		Sets excitatory/inhibitory currents to the network for the given state.
		This does not simulate the network.
		"""		
		# Extract input values.
		input_values = [state[i] for i in range(len(state)) if i is not self.output_index]

		# Set input layer neurons.
		self.set_input_currents(input_values)

		# Set alpha layer neurons.
		if inhibit_alpha:
			self.set_hidden_currents(state[self.output_index])


	def set_intrinsic_rate(self, intrinsic_rate):
		"""
		Sets the learning rate of intrinsic plasticity in the alpha layer.
		Applicable only for SRM Peceveski neurons.
		"""
		if self.params['alpha_neuron_type'] == 'srm_pecevski_alpha':
			nest.SetStatus(self.alpha, {'eta_bias':intrinsic_rate})
		else:
			logging.warning("Cannot set a bias rate in neurons that don't support it. Returning with no effect.")


	def set_plasticity_learning_time(self, learning_time):
		"""
		Sets the STDP learning time in the STDP connections (in ms).
		"""
		if self.params['chi_alpha_synapse_type'] == 'stdp_pecevski_synapse':
			synapses = nest.GetConnections(self.chi, self.alpha)
			nest.SetStatus(synapses, {'learning_time':learning_time}) # We stop learning by setting the learning time to 0.
		else:
			logging.warning("Cannot set a learning time in synapses that don't support it. Returning with no effect.")


	def set_plasticity_learning_rate(self, rate):
		"""
		Toggles vanilla STDP on or off. Has no effect on Pecevski STDP synapses.
		"""
		if self.params['chi_alpha_synapse_type'] == 'stdp_synapse':
			synapses = nest.GetConnections(self.chi, self.alpha)
			nest.SetStatus(synapses, {'lambda':rate})
		else:
			logging.warning("Cannot set a learning rate in synapses that don't support it. Returning with no effect.")


	def simulate_without_input(self, duration):
		"""
		Simulates the network without any input.
		"""
		self.clear_currents()
		nest.Simulate(duration)


	def present_random_sample(self, duration=None):
		"""
		Simulates the network for the given duration while a constant external current
		is presented to each population coding neuron in the network, chosen randomly 
		to be excitatory or inhibitory from a valid state of the distribution.
		Alpha neurons are inhibited if the value they represent does not match the 
		sample value.
		"""
		if not self.initialised:
			raise Exception("SAM module not initialised yet.")

		# Get a random state from the distribution.
		state = self.draw_random_sample()

		# Set currents in input and output layers.
		self.set_currents(state, inhibit_alpha=True)

		# Simulate.
		nest.Simulate(duration if duration is not None else self.params['sample_presentation_time'])


	def present_input_evidence(self, duration=None, sample=None):
		"""
		Presents the given sample state or a random one drawn from the set distribution
		and simulates for the given period. This only activates/inhibits the input 
		layer. 
		"""
		if not self.initialised:
			raise Exception("SAM module not initialised yet.")

		# Get a random state from the distribution if one is not given.
		state = sample if sample is not None else self.draw_random_sample()

		# Set currents in input layer.
		self.set_currents(state, inhibit_alpha=False)

		# Simulate.
		nest.Simulate(duration if duration is not None else self.params['sample_presentation_time'])


	def clear_currents(self):
		"""
		Convenience call that unsets all external currents.
		"""
		nest.SetStatus(self.all_neurons, {'I_e':0.0})


	def connect_multimeter(self, node, multimeter=None):
		"""
		Connects a multimeter to the bias and membrane voltage of the selected node.
		"""
		# Create a multimeter.
		multimeter = nest.Create('multimeter', params={"withtime":True, "record_from":["V_m", "bias"]}) if multimeter is None else multimeter

		# Connect to all neurons.
		nest.Connect(multimeter, [node])

		return multimeter


	def plot_potential_and_bias(self, multimeter):
		"""
		Plots the voltage traces of the cell membrane and bias from all neurons that were
		connected during the simulation.
		"""
		dmm = nest.GetStatus(multimeter)[0]
		Vms = dmm["events"]["V_m"]
		bias = dmm["events"]["bias"]
		ts = dmm["events"]["times"]

		pylab.figure()
		pylab.plot(ts, Vms)
		pylab.plot(ts, bias)
		pylab.show()


	def connect_reader(self, nodes, spikereader=None):
		"""
		Connects a spike reader to the network and returns it.
		"""
		# Create a spike reader.
		spikereader = nest.Create('spike_detector', params={'withtime':True, 'withgid':True}) if spikereader is None else spikereader

		# Connect all neurons to the spike reader.
		nest.Connect(nodes, spikereader, syn_spec={'delay':self.params['devices_delay']})

		return spikereader


	def plot_all(self, multimeter, spikereader):
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

		pylab.figure()
		pylab.plot(ts, Vms)
		pylab.plot(ts, bias)
		pylab.plot(times, senders, '|')
		pylab.show()


	def plot_spikes(self, spikereader):
		"""
		Plots the spike trace from all neurons the spikereader was connected to during
		the simulation.
		"""
		# Get spikes and plot.
		spikes = nest.GetStatus(spikereader, keys='events')[0]
		senders = spikes['senders']
		times = spikes['times']

		# Count spikes per output neuron.
		counts = defaultdict(int)
		for node in senders:
			counts[node] += 1 if node in self.zeta else 0
		for i in self.zeta:
			logging.info("Neuron {} spiked {} times".format(i, counts[i]))

		# Plot
		pylab.figure()
		pylab.plot(times, senders, '|')
		pylab.show()


	def print_network_info(self):
		"""
		Prints network architecture info.
		"""
		logging.info("baseline = {}".format(self.params['bias_baseline']))
		logging.info("%s",self.all_neurons)


	def get_neuron_biases(self, neurons):
		"""
		Convenience function that returns the bias of specified
		neurons, assuming they keep track of bias.
		"""
		return np.array(nest.GetStatus(neurons, 'bias'))


	def get_connection_weight(self, source, target):
		"""
		Convenience function that returns the weight of the
		connection between two neurons.
		"""
		return nest.GetStatus(nest.GetConnections([source], [target]), 'weight')[0]


	def compute_implicit_distribution(self):
		"""
		Returns the distribution encoded by the network's weights and biases
		at this point in time.
		Note: This does not look at spontaneous network activity, but the 
		analytical distribution implicitly encoded by the network architecture,
		as described by Eqn. 5 in Peceveski et. al. 
		"""
		implicit = dict(self.distribution)
		for t in implicit.keys():
			xs = [t[i] for i in range(self.num_vars) if i is not self.output_index]
			z = t[self.output_index]
			p = 0
			
			# For each z value, the joint distribution is only concerned with 
			# alphas that encode for that particular value (i.e. each alpha 
			# neuron in the subpop that codes for that z value), since
			# p(z|a) = 0 for other alphas.
			for i in range(len(self.alpha)):
				if i // self.num_modes != (z - 1): continue
				subtotal = 0
				alpha_neuron = self.alpha[i]

				# Find sum of weights going into this alpha neuron.
				for j in range(len(xs)):
					chi_index = j * self.num_discrete_vals + xs[j] - 1
					chi_neuron = self.chi[chi_index]
					w_hat = self.get_connection_weight(chi_neuron, alpha_neuron) + self.params['weight_baseline']
					subtotal += w_hat

				# Find the bias of this alpha neuron.
				b_hat = self.get_neuron_biases([alpha_neuron])[0] + self.params['bias_baseline']
				subtotal += b_hat

				# Add the exponential of the subtotal to the probability of this 
				# combination of variables.
				p += np.exp(subtotal)			

			implicit[t] = p

		# Normalise distribution.
		total = np.sum(list(implicit.values()))
		for k, v in implicit.items():
			implicit[k] /= total

		return implicit