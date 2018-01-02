import copy
import itertools
import logging
import nest
import numpy as np
import pylab
import random
import sam.helpers as helpers
import time
from collections import Counter, defaultdict, OrderedDict


class SPINetwork:
	"""
	Encapsulates a Sparse Probabilistic Inference (SPI) network, which uses sparse connectivity in a spiking
	network of interconnected excitatory/inhibitory pools of neurons to encode a Bayesian graph.
	"""

	def __init__(self, randomise_seed=False):
		"""
		Initialises basic parameters.
		"""
		self.seed = int(time.time()) if randomise_seed else 705
		self.initialised = False


	@staticmethod
	def basic_parameter_spec():
		"""
		Returns a dictionary of param_name:(min, max) pairs, which describe the legal
		limit of the parameters.
		"""
		param_spec = {
			'stdp_rate_initial':(0.0, 0.01),
			'stdp_rate_final':(0.0, 0.01),
			'T':(0.0, 1.0),
			'bias_rate_1':(0.0, 0.1),
			'bias_baseline':(-40.0, 0.0),
			'prob_exp_term':(0.0, 100.0),
			#'prob_exp_term_scale':(0.0, 5.0),
			'bias_relative_spike_rate':(1e-5, 1.0),
			'connectivity_chi_inh':(0.0, 1.0),
			'connectivity_inh_chi':(0.0, 1.0),
			'connectivity_inh_self':(0.0, 1.0),
			'connectivity_chi_chi':(0.0, 1.0),
			'connectivity_chi_self':(0.0, 1.0),
			#'delay_max':(0.1, 10.0),
			#'delay_min_ratio':(0.0, 1.0),
			'weight_chi_inhibitors':(0.0, 10.0),
			'weight_chi_self':(0.0, 10.0),
			'weight_inhibitors_chi':(-10.0, 0.0),
			'weight_inhibitors_self':(0.0, 10.0),
			'weight_baseline':(-10.0, 0.0),
			'weight_chi_chi_max':(0.01, 5.0)
			}

		return param_spec


	def get_spi_defaults(self, override_params={}):
		"""
		Combines passed SPI parameter overrides with defaults.
		override_params: dictionary of parameters to set.
		"""
		# Set common properties.
		tau = 10.0 if 'tau' not in override_params else override_params['tau']
		delay_fixed = 1.0 if 'delay_fixed' not in override_params else override_params['delay_fixed']
		delay_max = 4.0 if 'delay_max' not in override_params else round(override_params['delay_max'], 1)
		delay_min = 1.0 if 'delay_min_ratio' not in override_params else round(max(override_params['delay_min_ratio'] * delay_max, 0.1), 1)

		# Set all derived and underived properties.
		params = {
			'amplitude_exc':2.0,
			'amplitude_inh':2.0,
			'bias_change_time_fraction':1.0,
			'bias_rate_1':0.01,
			'bias_rate_2':0.02,
			'bias_relative_spike_rate':0.02,
			'bias_baseline':0.0,
			'bias_max':5.0,
			'bias_min':-30.0,
			'bias_initial':5.0,
			'bias_inhibitors':-10.0,
			'bias_input':-10.0,
			'bias_chi_mean':5.0,
			'bias_chi_std':0.1,
			'connectivity_chi_inh':0.575,
			'connectivity_inh_chi':0.6,
			'connectivity_inh_self':0.55,
			'connectivity_chi_chi':1.0,
			'connectivity_chi_self':0.0,
			'current_plus_chi':0.0,
			'current_minus_chi':-20.0,
			'current_minus_input':-30.0,
			'current_plus_input':30.0,
			'dead_time_inhibitors':tau,
			'delay_chi_inhibitors':delay_fixed,
			'delay_inhibitors_chi':delay_fixed,
			'delay_chi_self':delay_fixed,
			'delay_chi_chi_min':delay_min,
			'delay_chi_chi_max':delay_max,
			'delay_inhibitors_self':delay_fixed,
			'delay_devices':delay_min,
			'learning_time':300000,
			'pool_size_excitatory':5,
			'pool_size_inhibitory':10,
			'prob_linear_term':0.0,
			'prob_exp_term':1.0/tau,
			'prob_exp_term_scale':1.0,
			'neuron_type_chi':'srm_pecevski_alpha',
			'neuron_type_inhibitors':'srm_pecevski_alpha',
			'neuron_type_input':'srm_pecevski_alpha',
			'sample_presentation_time':100.0,
			'stdp_rate_initial':0.002,
			'stdp_rate_final':0.0006,
			'stdp_time_fraction':1.0,
			'synapse_type_chi_chi':'stdp_pecevski_synapse',
			'synapse_type_chi_inhibitors':'static_synapse',
			'synapse_type_chi_self':'static_synapse',
			'synapse_type_inhibitors_chi':'static_synapse',
			'synapse_type_inhibitors_self':'static_synapse',
			'T':0.58,
			'tau':tau,
			'tau_membrane':delay_min/100,
			'tau_alpha':tau,
			'tau_multiplier_max':100000.0,
			'use_rect_psp_exc':True,
			'use_rect_psp_inh':True,
			'use_rect_psp_exc_inhibitors':True,
			'use_renewal':False,
			'weight_chi_chi_max':1.0,
			'weight_chi_chi_min':0.0,
			'weight_chi_chi_std':0.1,
			'weight_chi_inhibitors':13.57,
			'weight_chi_self':13.57,
			'weight_inhibitors_chi':-1.86,
			'weight_inhibitors_self':13.57,
			'weight_initial':3.0,		
			'weight_baseline':2.5 * np.log(0.2)
		}

		# Update defaults.
		params.update(override_params)
		return params


	def set_kernel_settings(self):
		"""
		Sets main NEST kernel settings.
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


	def set_nest_defaults(self, params):
		"""
		Clears and sets the NEST defaults from the parameters.
		NOTE: Any change of the network parameters needs a corresponding call to
		this function in order to update settings.
		"""
		nest.SetDefaults('static_synapse', params={'weight':params['weight_initial']})

		nest.SetDefaults('stdp_pecevski_synapse', params={
			'eta_0':params['stdp_rate_initial'],
			'eta_final':params['stdp_rate_final'], 
			'learning_time':int(params['stdp_time_fraction'] * params['learning_time']),
			'weight':params['weight_initial'],
			'w_baseline':params['weight_baseline'],
			'tau':params['tau'],
			'T':params['T'],
			'depress_multiplier':params['tau_multiplier_max']
			})

		nest.SetDefaults('srm_pecevski_alpha', params={
			'rect_exc':params['use_rect_psp_exc'], 
			'rect_inh':params['use_rect_psp_inh'],
			'e_0_exc':params['amplitude_exc'], 
			'e_0_inh':params['amplitude_inh'],
			'I_e':0.0,
			'tau_m':params['tau_membrane'], # Membrane time constant (only affects current injections). Is this relevant?
			'tau_exc':params['tau'],
			'tau_inh':params['tau'], 
			'tau_bias':params['tau'],
			'eta_bias':0.0, # eta_bias is set manually.
			'rel_eta':params['bias_relative_spike_rate'],
			'b_baseline':params['bias_baseline'],
			'max_bias':params['bias_max'],
			'min_bias':params['bias_min'],
			'bias':params['bias_initial'],
			'dead_time':params['tau'], # Abs. refractory period.
			'dead_time_random':False,
			'c_1':params['prob_linear_term'], # Linear part of transfer function.
			'c_2':params['prob_exp_term'], # The coefficient of the exponential term in the transfer function.
			'c_3':params['prob_exp_term_scale'], # Scaling coefficient of effective potential in exponential term.
			'T':params['T'],
			'use_renewal':params['use_renewal']
			})

		nest.SetDefaults('stdp_synapse', params={
			'tau_plus':params['tau_alpha'],
			'Wmax':params['weight_chi_chi_max'],
			'mu_plus':0.0,
			'mu_minus':0.0,
			'lambda':params['stdp_rate_final'],
			'weight':params['weight_initial']
			})

		# Create layer model specialisations.
		# Chi population model.
		if params['neuron_type_chi'] == 'srm_pecevski_alpha':
			self.chi_neuron_params={'eta_bias':params['bias_rate_1']}
		else:
			raise NotImplementedError

		# Inhibitor population model.
		if params['neuron_type_inhibitors'] == 'srm_pecevski_alpha':
			self.inhibitors_neuron_params={
				'bias':params['bias_inhibitors'],
				'dead_time':params['dead_time_inhibitors'],
				'rect_exc':params['use_rect_psp_exc_inhibitors']				}
		else:
			raise NotImplementedError
			
		# Input population model (for conditional networks only).
		if params['neuron_type_input'] == 'srm_pecevski_alpha':
			self.input_neuron_params={'bias':params['bias_input']}
		else:
			raise NotImplementedError
					
		# Create synapse model specialisations.
		# Chi-chi synapses.
		if params['synapse_type_chi_chi'] == 'stdp_pecevski_synapse':
		 	self.chi_chi_synapse_params={
		 		'model':'stdp_pecevski_synapse',
				'weight':{
					'distribution':'normal_clipped',
					'low':params['weight_chi_chi_min'],
					'high':params['weight_chi_chi_max'],
					'mu':params['weight_chi_chi_max'],
					'sigma':params['weight_chi_chi_std']
					},
				'delay':{
					'distribution':'uniform',
					'low':params['delay_chi_chi_min'],
					'high':params['delay_chi_chi_max'],
					},
				'max_weight':params['weight_chi_chi_max'],
				'min_weight':params['weight_chi_chi_min']
				}
		else:
			raise NotImplementedError

		# Chi-inhibitors synapses.
		if params['synapse_type_chi_inhibitors'] == 'static_synapse':
			self.chi_inhibitors_synapse_params={
				'model':'static_synapse',
				'weight':params['weight_chi_inhibitors'],
				'delay':params['delay_chi_inhibitors']
				}
		else:
			raise NotImplementedError

		# Inhibitors-chi synapses.
		if params['synapse_type_inhibitors_chi'] == 'static_synapse':
			self.inhibitors_chi_synapse_params={
				'model':'static_synapse',
				'weight':params['weight_inhibitors_chi'],
				'delay':params['delay_inhibitors_chi']
				}
		else:
			raise NotImplementedError

		# Inhibitors-inhibitors synapses.
		if params['synapse_type_inhibitors_self'] == 'static_synapse':
			self.inhibitors_self_synapse_params={
				'model':'static_synapse',
				'weight':params['weight_inhibitors_self'],
				'delay':params['delay_inhibitors_self']
				}
		else:
			raise NotImplementedError

		# Chi-self synapses.
		if params['synapse_type_chi_self'] == 'static_synapse':
			self.chi_self_synapse_params={
				'model':'static_synapse',
				'weight':params['weight_chi_self'],
				'delay':params['delay_chi_self']
				}
		else:
			raise NotImplementedError

		self.chi_chi_connectivity_params={
				'rule':'pairwise_bernoulli',
				'p':params['connectivity_chi_chi']
			}

		self.chi_inh_connectivity_params={
				'rule':'pairwise_bernoulli',
				'p':params['connectivity_chi_inh']
			}

		self.inh_chi_connectivity_params={
				'rule':'pairwise_bernoulli',
				'p':params['connectivity_inh_chi']
			}

		self.inh_self_connectivity_params={
				'rule':'pairwise_bernoulli',
				'p':params['connectivity_inh_self']
			}

		self.chi_self_connectivity_params={
				'rule':'pairwise_bernoulli',
				'p':params['connectivity_chi_self']
			}

	
	def create_conditional_network(self, dependencies, distribution, num_discrete_vals=2, override_params={}):
		"""
		Creates a recurrent SPI network which codes the conditional distribution passed, 
		assuming only a single dependency.
		"""
		if len(dependencies) > 1:
			raise Exception("There are more than one dependency in the passed dictionary.")
			
		# Set main kernel settings.
		self.set_kernel_settings()

		# Set parameter defaults.
		self.params = self.get_spi_defaults(override_params)
		self.network_type = 'conditional'

		# Remove generic parameters from parameter list.
		for k in self.parameter_repeats(): self.params.pop(k)

		self.dependencies = dependencies
		self.distribution = distribution
		self.chi_pools = dict()
		self.inhibitory_pools = dict()
		self.subnetwork_distributions = dict()
		self.dep_indices = dict()
		self.subnetwork_params = dict()
		self.inputs = dict()

		# Create multiple excitatory pools for the dependency, ignoring input for now.
		for ym in self.__get_variables_ordered():
			# Create the chi pools in order of variable name.
			ys = dependencies[ym]

			subnetwork_params = self.filter_repeat_params(override_params, ym)
			self.subnetwork_params[ym] = subnetwork_params.copy()
			subnetwork_vars = sorted([ym] + ys)
			
			# Set subnetwork parameters.
			self.subnetwork_params[ym] = self.get_spi_defaults(self.subnetwork_params[ym])
			self.set_nest_defaults(self.subnetwork_params[ym])
			
			# Create excitatory and inhibitory pools.
			self.chi_pools[ym] = nest.Create(self.subnetwork_params[ym]['neuron_type_chi'], n=self.subnetwork_params[ym]['pool_size_excitatory'] * num_discrete_vals, params=self.chi_neuron_params)
			self.inhibitory_pools[ym] = nest.Create(self.subnetwork_params[ym]['neuron_type_inhibitors'], n=self.subnetwork_params[ym]['pool_size_inhibitory'], params=self.inhibitors_neuron_params)
			
			# Connect excitatory and inhibitory pools.
			nest.Connect(self.chi_pools[ym], 
				self.inhibitory_pools[ym], 
				conn_spec=self.chi_inh_connectivity_params, 
				syn_spec=self.chi_inhibitors_synapse_params)
			nest.Connect(self.inhibitory_pools[ym], 
				self.chi_pools[ym], 
				conn_spec=self.inh_chi_connectivity_params, 
				syn_spec=self.inhibitors_chi_synapse_params)

			# Self-connect inhibitory pools (autopses)
			nest.Connect(self.inhibitory_pools[ym],
				self.inhibitory_pools[ym],
				conn_spec=self.inh_self_connectivity_params,
				syn_spec=self.inhibitors_self_synapse_params)

			# Self-connect excitatory pools.
			value_pool = [self.get_variable_neurons(ym, x) for x in range(1, num_discrete_vals + 1)]
			if self.subnetwork_params[ym]['connectivity_chi_self'] > 0:
				for pool in value_pool:
					nest.Connect(pool, pool, 
						conn_spec=self.chi_self_connectivity_params,
						syn_spec=self.chi_self_synapse_params)

			# Chi bias randomisation.
			self.set_random_biases(self.chi_pools[ym], 
				self.subnetwork_params[ym]['bias_chi_mean'], 
				self.subnetwork_params[ym]['bias_chi_std'],
				self.subnetwork_params[ym]['bias_min'],
				self.subnetwork_params[ym]['bias_max'])

			# Set marginal distribution of this layer and this variable's index in the distribution.
			self.dep_indices[ym] = subnetwork_vars.index(ym),
			self.subnetwork_distributions[ym] = helpers.compute_marginal_distribution(distribution, subnetwork_vars, num_discrete_vals)
		
		# Specify input pools, making the right recurrent connections.
		for ym in self.__get_variables_ordered():
			ys = dependencies[ym]
			input_vars = sorted(ys)
			for y in input_vars:
				self.inputs[y] = nest.Create(self.subnetwork_params[ym]['neuron_type_chi'], n=self.subnetwork_params[ym]['pool_size_excitatory'] * num_discrete_vals, params=self.input_neuron_params)
			
			# Re-assign NEST defaults since each subnetwork overwrites them.
			self.set_nest_defaults(self.subnetwork_params[ym])

			# Connect chi pools with each other.
			for y in ys:
				nest.Connect(self.inputs[y], 
					self.chi_pools[ym], 
					conn_spec=self.chi_chi_connectivity_params,
					syn_spec=self.chi_chi_synapse_params)

		# Set other metainformation flags.
		self.num_discrete_vals = num_discrete_vals
		self.special_params = {}

		# Track all neurons.
		self.all_neurons = tuple()
		for y in self.inputs:
			self.all_neurons += self.inputs[y]
		for ym in self.__get_variables_ordered():
			self.all_neurons += self.chi_pools[ym] + self.inhibitory_pools[ym]

		logging.info("Created {} neurons.".format(len(self.all_neurons)))
		self.initialised = True
		

	def create_network(self, dependencies, distribution, num_discrete_vals=2, override_params={}, special_params={}):
		"""
		Generates a recurrent SPI network according to the supplied dependencies. 
		This gives an analog to the SAMGraph structure, but it also uses:
		1. Sparse connectivity
		2. Large neuron pools for each variable (no distinction between modes)
		3. No output layer.
		4. A rate-based competition on output state.
		dependencies: A dictionary of ym:[y1, y2, ... , yn] where ym is a string
			naming the dependent variable and yi are strings naming the Markov
			blanket variables. Note: it is up to the user to supply correct
			dependencies. Each string should have the format "yi", e.g. "y1",
			starting from index 1.
		distribution: A joint target distribution that the network is supposed to
			estimate. Supplied as a dictionary of (x1, x2, ..., xk):p pairs, 
			where the xi are values of the random variables, and p is a 
			probability.
		override_params: A dictionary of overriden parameters to be passed to all subnetworks.
		special_params: A dictionary of dictionaries, in the format {"y1":{...}, 
			...}, containing parameters to be sent specifically to individual 
			subnetworks. This overwrites parameters in override_params, even those that are
			meant to evolve on a module-by-module basis initially.
		The other parameters are as in SAMModule.
		"""
		# Set main kernel settings.
		self.set_kernel_settings()

		# Set parameter defaults.
		self.params = self.get_spi_defaults(override_params)
		self.network_type = 'joint'

		# Remove generic parameters from parameter list.
		for k in self.parameter_repeats(): self.params.pop(k)

		self.dependencies = dependencies
		self.distribution = distribution
		self.chi_pools = dict()
		self.inhibitory_pools = dict()
		self.subnetwork_distributions = dict()
		self.dep_indices = dict()
		self.subnetwork_params = dict()

		# Create multiple excitatory pools for each dependency, ignoring input for now.
		for ym in self.__get_variables_ordered():
			# Create the chi pools in order of variable name.
			ys = dependencies[ym]

			subnetwork_params = self.filter_repeat_params(override_params, ym)
			self.subnetwork_params[ym] = {**subnetwork_params, **special_params[ym]} if ym in special_params else subnetwork_params.copy()
			subnetwork_vars = sorted([ym] + ys)
			
			# Set subnetwork parameters.
			self.subnetwork_params[ym] = self.get_spi_defaults(self.subnetwork_params[ym])
			self.set_nest_defaults(self.subnetwork_params[ym])
			
			# Create excitatory and inhibitory pools.
			self.chi_pools[ym] = nest.Create(self.subnetwork_params[ym]['neuron_type_chi'], n=self.subnetwork_params[ym]['pool_size_excitatory'] * num_discrete_vals, params=self.chi_neuron_params)
			self.inhibitory_pools[ym] = nest.Create(self.subnetwork_params[ym]['neuron_type_inhibitors'], n=self.subnetwork_params[ym]['pool_size_inhibitory'], params=self.inhibitors_neuron_params)
			
			# Connect excitatory and inhibitory pools.
			nest.Connect(self.chi_pools[ym], 
				self.inhibitory_pools[ym], 
				conn_spec=self.chi_inh_connectivity_params, 
				syn_spec=self.chi_inhibitors_synapse_params)
			nest.Connect(self.inhibitory_pools[ym], 
				self.chi_pools[ym], 
				conn_spec=self.inh_chi_connectivity_params, 
				syn_spec=self.inhibitors_chi_synapse_params)

			# Self-connect inhibitory pools (autopses)
			nest.Connect(self.inhibitory_pools[ym],
				self.inhibitory_pools[ym],
				conn_spec=self.inh_self_connectivity_params,
				syn_spec=self.inhibitors_self_synapse_params)

			# Self-connect excitatory pools.
			value_pool = [self.get_variable_neurons(ym, x) for x in range(1, num_discrete_vals + 1)]
			if self.subnetwork_params[ym]['connectivity_chi_self'] > 0:
				for pool in value_pool:
					nest.Connect(pool, pool, 
						conn_spec=self.chi_self_connectivity_params,
						syn_spec=self.chi_self_synapse_params)

			# Chi bias randomisation.
			self.set_random_biases(self.chi_pools[ym], 
				self.subnetwork_params[ym]['bias_chi_mean'], 
				self.subnetwork_params[ym]['bias_chi_std'],
				self.subnetwork_params[ym]['bias_min'],
				self.subnetwork_params[ym]['bias_max'])

			# Set marginal distribution of this layer and this variable's index in the distribution.
			self.dep_indices[ym] = subnetwork_vars.index(ym),
			self.subnetwork_distributions[ym] = helpers.compute_marginal_distribution(distribution, subnetwork_vars, num_discrete_vals)
		
		# Specify input pools for each subnetwork, making the right recurrent connections.
		for ym in self.__get_variables_ordered():
			ys = dependencies[ym]
			input_vars = sorted(ys)
			input_neurons = tuple([n for y in input_vars for n in self.chi_pools[y]])
			
			# Re-assign NEST defaults since each subnetwork overwrites them.
			self.set_nest_defaults(self.subnetwork_params[ym])

			# Connect chi pools with each other.
			nest.Connect(input_neurons, 
				self.chi_pools[ym], 
				conn_spec=self.chi_chi_connectivity_params,
				syn_spec=self.chi_chi_synapse_params)

		# Set other metainformation flags.
		self.num_discrete_vals = num_discrete_vals
		self.special_params = special_params

		# Track all neurons.
		self.all_neurons = tuple()
		for ym in self.__get_variables_ordered():
			self.all_neurons += self.chi_pools[ym] + self.inhibitory_pools[ym]

		logging.info("Created {} neurons.".format(len(self.all_neurons)))
		self.initialised = True


	def __get_variables_ordered(self):
		"""
		Returns the variables ordered by name in an array, e.g. ['y1', 'y2']
		"""
		if self.network_type == 'joint':
			return ['y' + str(i) for i in range(1, len(self.dependencies) + 1)]
		else:
			return [list(self.dependencies.keys())[0]]


	@staticmethod
	def parameter_spec(num_modules, network_type='joint'):
		"""
		Returns a dictionary of param_name:(min, max) pairs, which describe the legal
		limit of the parameters.
		"""
		# Get the vanilla spec.
		spec = SPINetwork.basic_parameter_spec()

		# For each variable that appears in the repeat spec, add n variables with the 
		# name, suffixed by '_1', '_2', etc.
		repeat_spec = SPINetwork.parameter_repeats(network_type)
		for k in repeat_spec:
			k_spec = spec[k]
			spec.pop(k)
			new_keys = [k + '_' + str(i) for i in range(1, num_modules + 1)]
			for new_k in new_keys:
				spec[new_k] = k_spec

		return spec


	@staticmethod
	def parameter_repeats(network_type='joint'):
		"""
		Returns a list of parameters that are to be specialised by each subnetwork in the 
		graph network, i.e. that can evolve separately.
		"""
		repeats = ['bias_baseline', 'weight_chi_chi_max'] if network_type == "joint" else []
		return repeats


	def parameter_string(self):
		"""
		Returns a string containing all model parameters, combining the params dictionary
		passed to create_network() as well as the special_params dictionary.
		"""
		params = OrderedDict(sorted(self.params.items()))
		special_params = OrderedDict(sorted(self.special_params.items()))

		return "Params:\n{}\nSpecial params:\n{}".format(helpers.get_dictionary_string(params), helpers.get_dictionary_string(special_params))


	def filter_repeat_params(self, params, var_name):
		"""
		Parses and filters the params dictionary to keep only the variables that are 
		relevant for the specified variable name.
		var_name: a string specifying the variable name, e.g. 'y1'.
		"""
		filtered = {}
		repeats = SPINetwork.parameter_repeats()
		for k in params:
			repeat_var = None
			for r in repeats:
				if r in k: repeat_var = r

			if repeat_var is None: filtered[k] = params[k]
			else:
				# Keep only that which has the same suffix.
				var_suffix = var_name[1:]
				k_truncated = k.replace(repeat_var + '_', '')
				if var_suffix in k_truncated: filtered[repeat_var] = params[k]

		return filtered


	def clone(self):
		"""
		Returns a network with the same architecture and equal weights and biases.
		Note: We pass the params to create_network because the function overwrites 
		the bias baseline unless the bias baseline is passed.
		"""
		new_network = SPINetwork()
		new_network.seed = self.seed
		new_network.network_type = self.network_type
		if new_network.network_type == 'joint':
			new_network.create_network(
				dependencies=self.dependencies,
				num_discrete_vals=self.num_discrete_vals,
				distribution=self.distribution,
				override_params=self.params,
				special_params=self.special_params)
		elif new_network.network_type == 'conditional':
			new_network.create_conditional_network(
				dependencies=self.dependencies,
				num_discrete_vals=self.num_discrete_vals,
				distribution=self.distribution,
				override_params=self.params)
				
		# Copy bias values.
		for ym in new_network.__get_variables_ordered():
			if new_network.subnetwork_params[ym]['neuron_type_chi'] == 'srm_pecevski_alpha':
				chi_biases = nest.GetStatus(self.chi_pools[ym], 'bias')
				nest.SetStatus(new_network.chi_pools[ym], 'bias', chi_biases)

		# Copy synapse weights.
		# NOTE: This only makes sense if the connections are the same, which should
		# be guaranteed because clones have the same seed and should therefore
		# create the same connections.
		conns = nest.GetConnections(self.all_neurons, self.all_neurons)
		weights = nest.GetStatus(conns, 'weight')
		conns_new = nest.GetConnections(new_network.all_neurons, new_network.all_neurons)

		nest.SetStatus(conns_new, 'weight', weights)

		return new_network


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


	def draw_random_sample(self):
		"""
		Draws a random sample from the target distribution.
		See the documentation in helpers for more details on how this works.
		"""
		return helpers.draw_from_distribution(self.distribution, complete=True, randomiser=self.rngs[0])
		

	def draw_normal_values(self, n, mu, sigma, min, max):
		"""
		Draws a random sequence of numbers from a normal
		distribution with the specified parameters. 
		"""
		# Do this element by element, since numpy does not offer trunc norm.
		sample = []
		while len(sample) < n:
			r = self.rngs[0].normal(mu, sigma)
			if r >= min and r <= max:
				sample.append(r)
	
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


	def get_variable_neurons(self, var_name, var_value):
		"""
		Returns the neuron population that rate codes for this particular variable
		and value combination.
		"""
		pool_size_excitatory = self.subnetwork_params[var_name]['pool_size_excitatory']
		return self.chi_pools[var_name][(var_value - 1) * pool_size_excitatory:var_value * pool_size_excitatory]

		
	def get_input_neurons(self, var_name, var_value):
		"""
		Returns the input neuron population that rate codes for this particular variable
		and value combination.
		"""
		ym = list(self.dependencies.keys())[0]
		pool_size_excitatory = self.subnetwork_params[ym]['pool_size_excitatory']
		return self.inputs[var_name][(var_value - 1) * pool_size_excitatory:var_value * pool_size_excitatory]

		
	def set_input_currents(self, state):
		"""
		Inhibits/forces input neurons (for conditional-type networks only).
		"""
		for y in self.inputs:
			var_index = int(y[1:]) - 1
			for x in range(1, self.num_discrete_vals + 1):
				nodes = self.get_input_neurons(y, x)
				inhibit = state[var_index] != x
				current = self.params['current_minus_input'] if inhibit else self.params['current_plus_input']
				nest.SetStatus(nodes, {'I_e':current})
		
		
	def set_chi_currents(self, state):
		"""
		Inhibits/forces the chi neurons according to the provided state.
		state: a tuple containing the values to be encoded by the chi populations,
		one value for each subnetwork of chi neurons.
		"""
		for ym in self.__get_variables_ordered():
			var_index = int(ym[1:]) - 1
			for x in range(1, self.num_discrete_vals + 1):
				nodes = self.get_variable_neurons(ym, x)
				inhibit = state[var_index] != x
				current = self.params['current_minus_chi'] if inhibit else self.params['current_plus_chi']
				nest.SetStatus(nodes, {'I_e':current})
			

	def set_currents(self, state):
		"""
		Sets excitatory/inhibitory currents to the network for the given state.
		This does not simulate the network.
		"""		
		# Set chi pool neurons.
		self.set_chi_currents(state)
		
		if self.network_type == 'conditional':
			self.set_input_currents(state)


	def set_intrinsic_rate(self, intrinsic_rate):
		"""
		Sets the learning rate of intrinsic plasticity in all chi subnetworks.
		Applicable only for SRM Peceveski neurons.
		"""
		for ym in self.__get_variables_ordered():
			if self.subnetwork_params[ym]['neuron_type_chi'] == 'srm_pecevski_alpha':
				nest.SetStatus(self.chi_pools[ym], {'eta_bias':intrinsic_rate})
			else:
				logging.warning("Cannot set a bias rate in neurons that don't support it. Returning with no effect.")


	def set_plasticity_learning_time(self, learning_time):
		"""
		Sets the STDP learning time in the STDP connections (in ms).
		Note: raises an error if the neurons don't support learning time.
		"""
		if self.network_type == 'joint':
			# Get connections between chi pools.
			for ym in self.__get_variables_ordered():
				neurons_to = chi_pools[ym]
				neurons_from = tuple(n for ys in self.dependencies[ym] for n in chi_pools[ys])

				# Update all connections between neurons in these pools.
				synapses = nest.GetConnections(neurons_from, neurons_to)
				if len(synapses) > 0:
					nest.SetStatus(synapses, {'learning_time':learning_time})
		elif self.network_type == 'conditional':
			# Get connections between input and chi pools.
			for ym in self.__get_variables_ordered():
				neurons_to = self.chi_pools[ym]
				neurons_from = tuple(n for ys in self.dependencies[ym] for n in self.inputs[ys])

				# Update all connections between neurons in these pools.
				synapses = nest.GetConnections(neurons_from, neurons_to)
				if len(synapses) > 0:
					nest.SetStatus(synapses, {'learning_time':learning_time})


	def set_plasticity_learning_rate(self, rate):
		"""
		Toggles vanilla STDP on or off. 
		Note: raises an error if the neurons don't support lambda.
		"""
		if self.network_type == 'joint':
			# Get connections between chi pools.
			for ym in self.__get_variables_ordered():
				neurons_to = chi_pools[ym]
				neurons_from = tuple(n for ys in self.dependencies[ym] for n in chi_pools[ys])

				# Update all connections between neurons in these pools.
				synapses = nest.GetConnections(neurons_from, neurons_to)
				if len(synapses) > 0:
					nest.SetStatus(synapses, {'lambda':rate})
		elif self.network_type == 'conditional':
			# Get connections between input and chi pools.
			for ym in self.__get_variables_ordered():
				neurons_to = self.chi_pools[ym]
				neurons_from = tuple(n for ys in self.dependencies[ym] for n in self.inputs[ys])

				# Update all connections between neurons in these pools.
				synapses = nest.GetConnections(neurons_from, neurons_to)
				if len(synapses) > 0:
					nest.SetStatus(synapses, {'lambda':rate})


	def simulate_without_input(self, duration):
		"""
		Simulates the network without any input.
		"""
		self.clear_currents()
		nest.Simulate(duration)


	def present_random_sample(self, use_currents = True, duration=None):
		"""
		Simulates the network for the given duration while a constant external current
		is presented to each rate coding neuron in the network, chosen randomly 
		to be excitatory or inhibitory from a valid state of the distribution.
		use_currents: True to use external forcing currents, False to use Poisson spike
			generators instead of current.
		"""
		if not self.initialised:
			raise Exception("SPI network not initialised yet.")

		# Get a random state from the distribution.
		state = self.draw_random_sample()

		# Set currents in input and output layers.
		if use_currents:
			self.set_currents(state)
		else:
			raise NotImplementedError("Poisson input generation not yet implemented.")

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


	def connect_reader(self, nodes, spikereader=None):
		"""
		Connects a spike reader to the nodes passed and returns it.
		"""
		# Create a spike reader.
		spikereader = nest.Create('spike_detector', params={'withtime':True, 'withgid':True}) if spikereader is None else spikereader

		# Connect all neurons to the spike reader.
		nest.Connect(nodes, spikereader, syn_spec={'delay':self.params['delay_devices']})

		return spikereader


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


	def measure_experimental_joint_distribution(self, duration, timestep=10.0):
		"""
		Lets the network generate spontaneous spikes for a long duration
		and then uses the spike activity to calculate the frequency of network 
		states.
		"""
		logging.info("Starting experimental joint distribution measurement on SPI network.")

		# Attach a spike reader.
		spikereader = self.__stop_activity_attach_reader()

		# Get current time.
		start_time = nest.GetKernelStatus('time')

		# Simulate for duration ms with no input.
		nest.Simulate(duration)

		# Get spikes.
		spikes = nest.GetStatus(spikereader, keys='events')[0]

		return self.get_distribution_from_spikes(spikes, start_time, start_time + duration, 1.5 * self.params['tau'], timestep)


	def measure_experimental_cond_distribution(self, duration):
		"""
		Measures the conditional distribution of this network
		experimentally, i.e. by counting output spikes directly.
		Note: For best results, stop all plasticity effects.
		"""
		if self.network_type != 'conditional':
			raise Exception("Conditional distribution can only be measured for conditional-type networks.")

		var_values = range(1, self.num_discrete_vals + 1)
		possibilities = list(itertools.product(var_values, repeat=len(self.inputs)))
		conditional = dict(self.distribution)
		ym = list(self.dependencies.keys())[0]

		for p in possibilities:
			# print("Measuring experimental conditional distribution on input:", p)

			# Attach a spike reader.
			spikereader = self.__stop_activity_attach_reader()

			# Present input evidence.
			self.__present_input_evidence(duration=duration, sample=p)

			# Get spikes.
			spikes = nest.GetStatus(spikereader, keys='events')[0]
			senders = spikes['senders']

			# Count spikes per output neuron.
			counts = defaultdict(int)
			for node in senders:
				for x in var_values:
					counts[x] += 1 if node in self.get_variable_neurons(ym, x) else 0

			# Calculate conditional probabilities.
			total = np.sum(list(counts.values()))
			for z in var_values:
				conditional[p + (z,)] = counts[z] / total

		return conditional


	def __present_input_evidence(self, duration=None, sample=None):
			"""
			Presents the given sample state or a random one drawn from the set distribution
			and simulates for the given period. This only activates/inhibits the input 
			layer. 
			"""
			if not self.initialised:
				raise Exception("Network not initialised yet.")

			# Get a random state from the distribution if one is not given.
			state = sample if sample is not None else self.draw_random_sample()

			# Set currents in input layer.
			self.set_input_currents(state)

			# Simulate.
			nest.Simulate(duration if duration is not None else self.params['sample_presentation_time'])


	def __stop_activity_attach_reader(self):
		"""
		Stops all plasticity and input and attaches a reader to the excitatory subnetworks, returning it.
		"""
		# Attach a spike reader to all population coding layers.
		spikereader = nest.Create('spike_detector', params={'withtime':True, 'withgid':True})
		for ym in self.__get_variables_ordered():
			nest.Connect(self.chi_pools[ym], spikereader, syn_spec={'delay':self.params['delay_devices']})

		if self.network_type == 'conditional':
			for ys in self.inputs:
				nest.Connect(self.inputs[ys], spikereader, syn_spec={'delay':self.params['delay_devices']})

		# Clear currents.
		self.clear_currents()

		# Stop all plasticity.
		self.set_intrinsic_rate(0.0)
		self.set_plasticity_learning_time(0)

		return spikereader


	def __determine_state(self, spikes):
		"""
		Given a set of spikes of the network, this determines which of
		the network states the network is in, or whether it is in an 
		a zero-state, a zero-state being one in which the rate of both
		variable-value encoding populations is zero.
		Note: uses the first RNG to choose between equal-spike windows.
		"""
		state = [0 for i in range(len(self.dependencies))]
		for ym in self.__get_variables_ordered():
			i = int(ym[1:]) - 1
			variable_neurons = [self.get_variable_neurons(ym, x) for x in range(1, 1 + self.num_discrete_vals)]

			# Count the number of spikes of each subpool.
			spike_counts = Counter(spikes)
			counts = [sum(spike_counts[spike] for spike in value_neurons) for value_neurons in variable_neurons]

			# If both are zero, we have a zero state.
			if all(c == 0 for c in counts): break

			# Otherwise, find the variable value with the highest
			# number of spikes (or if equal, choose randomly).
			max_count = np.amax(counts)
			if counts.count(max_count) == 1: 
				state[i] = counts.index(max_count) + 1 # +1 is necessary because the 0-state is not coded for.
			else:
				indices = [j for j, x in enumerate(counts) if x == max_count]
				state[i] = self.rngs[0].choice(indices) + 1

		if self.network_type == 'conditional':
			for ys in self.inputs:
				i = int(ys[1:]) - 1
				variable_neurons = [self.get_input_neurons(ys, x) for x in range(1, 1 + self.num_discrete_vals)]

				# Count the number of spikes of each subpool.
				spike_counts = Counter(spikes)
				counts = [sum(spike_counts[spike] for spike in value_neurons) for value_neurons in variable_neurons]

				# If both are zero, we have a zero state.
				if all(c == 0 for c in counts): break

				# Otherwise, find the variable value with the highest
				# number of spikes (or if equal, choose randomly).
				max_count = np.amax(counts)
				if counts.count(max_count) == 1: 
					state[i] = counts.index(max_count) + 1 # +1 is necessary because the 0-state is not coded for.
				else:
					indices = [j for j, x in enumerate(counts) if x == max_count]
					state[i] = self.rngs[0].choice(indices) + 1

		return tuple(state)


	def get_distribution_from_spikes(self, spikes, start_time, end_time, averaging_window = None, timestep=1.0):
		"""
		Helper function that returns the distribution represented by the 
		spikes passed.
		"""
		averaging_window = self.params['tau'] if averaging_window is None else averaging_window
		senders = spikes['senders']
		times = spikes['times']

		# Prepare state distribution variables.
		joint = defaultdict(int)
		zeros = 0

		# For every timestep, figure out the network state we are in.
		steps = np.arange(start_time, end_time, timestep)
		for t in steps:
			spike_indices = [i for i, st in enumerate(times) if t - averaging_window < st <= t]
			state_spikes = [senders[i] for i in spike_indices]
			state = self.__determine_state(state_spikes)
			joint[state] += 1
			if state in self.distribution: 
				pass
			else:
				zeros += 1

		# Normalise all values.
		total = np.sum(list(joint.values()))
		for k, v in joint.items():
			joint[k] = v / total
		zeros /= len(steps)

		logging.info("Probability of a zero state: {}".format(zeros))

		return joint


	def draw_stationary_state(self, duration, ax=None):
		"""
		Lets the network spike without external input, and draws the spikes from
		the chi pool neurons.
		"""
		# Attach a spike reader to all rate coding pools.
		spikereader = nest.Create('spike_detector', params={'withtime':True, 'withgid':True})
		for ym in self.__get_variables_ordered():
			nest.Connect(self.chi_pools[ym], spikereader, syn_spec={'delay':self.params['delay_devices']})


		if self.network_type == 'conditional':
			for ys in self.inputs:
				nest.Connect(self.inputs[ys], spikereader, syn_spec={'delay':self.params['delay_devices']})

		# Clear currents.
		self.clear_currents()

		# Stop all plasticity.
		self.set_intrinsic_rate(0.0)
		self.set_plasticity_learning_time(0)

		# Get current time.
		start_time = nest.GetKernelStatus('time')

		# Simulate for duration ms with no input.
		nest.Simulate(duration)

		# Get spikes.
		spikes = nest.GetStatus(spikereader, keys='events')[0]
		senders = spikes['senders']
		times = spikes['times']

		# Map neuron indices to a range starting from 0, for clarity in reproduction.
		neuron_map = {}
		index = 0
		for ym in self.__get_variables_ordered():
			subnetwork = self.chi_pools[ym]
			for z in subnetwork:
				neuron_map[z] = index
				index += 1
		senders = [neuron_map[z] for z in senders]

		# Plot
		if ax is None:
			pylab.figure()
			pylab.plot(times, senders, '|')
			pylab.title('Spontaneous activity after training')
			pylab.show()
		else:
			ax.plot(times, senders, '|')
			ax.set_title('Spontaneous activity after training')