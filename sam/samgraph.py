import copy
import logging
import nest
import numpy as np
import sam.helpers as helpers
from collections import defaultdict, OrderedDict
from mpi4py import MPI
from sam.sam import SAMModule


class SAMGraph:
	"""
	Encapsulates an interconnected group of SAM modules as in Pecevski et al. 2016, 
	second experiment. This construction can be used to simulate a generic graph
	model.
	"""

	def __init__(self, randomise_seed=True):
		"""
		Initialises basic parameters.
		"""
		self.randomise_seed = randomise_seed
		self.initialised = False


	def create_network(self, dependencies, distribution, num_discrete_vals=2, num_modes=2, params={}, special_params={}):
		"""
		Generates a graph of SAM modules interconnected according to the supplied
		dependencies. 
		dependencies: A dictionary of ym:[y1, y2, ... , yn] where ym is a string
			naming the dependent variable and yi are strings naming the Markov
			blanket variables. Note: it is up to the user to supply correct
			dependencies. Each string should have the format "yi", e.g. "y1",
			starting from index 1.
		distribution: A joint target distribution that the network is supposed to
			estimate. Supplied as a dictionary of (x1, x2, ..., xk):p pairs, 
			where the xi are values of the random variables, and p is a 
			probability.
		params: A dictionary of common parameters to be passed to each SAM module.
		special_params: A dictionary of dictionaries, in the format {"y1":{...}, 
			...}, containing parameters to be sent specifically to individual 
			modules. This overwrites parameters in params, even those that are
			meant to evolve on a module-by-module basis initially.
		The other parameters are as in SAMModule.
		"""
		self.sams = dict()
		self.dependencies = dependencies
		self.distribution = distribution

		# Create a SAM module for each dependency, ignoring input layers for now.
		for ym, ys in dependencies.items():
			module_params = self.filter_repeat_params(params, ym)
			module_params = {**module_params, **special_params[ym]} if ym in special_params else module_params.copy()
			module_vars = sorted([ym] + ys)
			num_dependencies = len(module_vars) - 1
			self.sams[ym] = SAMModule(randomise_seed=self.randomise_seed)
			self.sams[ym].create_network(num_x_vars=num_dependencies, 
				num_discrete_vals=num_discrete_vals, 
				num_modes=num_modes, 
				distribution=helpers.compute_marginal_distribution(distribution, module_vars, num_discrete_vals), 
				params=module_params,
				dep_index=module_vars.index(ym),
				create_input_layer=False)

		# Specify input layer for each module, making the right recurrent connections.
		for ym, ys in dependencies.items():
			input_vars = sorted(ys)
			input_neurons = tuple([n for y in input_vars for n in self.sams[y].zeta])
			self.sams[ym].set_input_layer(input_neurons)

		# Set other metainformation flags.
		self.num_discrete_vals = num_discrete_vals
		self.num_modes = num_modes
		self.params = params
		self.special_params = special_params

		# Add generic parameters to parameter list.
		generics = dict(list(self.sams.values())[0].params)
		for k in self.parameter_repeats(): generics.pop(k)
		self.params.update(generics)

		self.initialised = True


	def clone(self):
		"""
		Clones the recurrent network.
		"""
		new_network = SAMGraph(randomise_seed=self.randomise_seed)
		new_network.create_network(
			num_discrete_vals=self.num_discrete_vals,
			num_modes=self.num_modes,
			dependencies=self.dependencies,
			distribution=self.distribution,
			params=self.params,
			special_params=self.special_params)

		# Copy weights and biases of each module.
		for ym in new_network.sams:
			new_network.sams[ym].copy_dynamic_properties(self.sams[ym])

		return new_network


	@staticmethod
	def parameter_spec(num_modules):
		"""
		Returns a dictionary of param_name:(min, max) pairs, which describe the legal
		limit of the parameters.
		"""
		# Get the vanilla spec from the underlying module class.
		spec = SAMModule.parameter_spec()

		# For each variable that appears in the repeat spec, add n variables with the 
		# name, suffixed by '_1', '_2', etc.
		repeat_spec = SAMGraph.parameter_repeats()
		for k in repeat_spec:
			k_spec = spec[k]
			spec.pop(k)
			new_keys = [k + '_' + str(i) for i in range(1, num_modules + 1)]
			for new_k in new_keys:
				spec[new_k] = k_spec

		return spec


	@staticmethod
	def parameter_repeats():
		"""
		Returns a list of parameters that are to be specialised by each module in the 
		graph network, i.e. that can evolve separately.
		"""
		repeats = ['bias_baseline']
		return repeats


	def parameter_string(self):
		"""
		Returns a string containing all model parameters, combining the params dictionary
		passed to create_network() with the params obtainable from the underlying SAMModule
		as well as the special_params dictionary.
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
		repeats = SAMGraph.parameter_repeats()
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


	def draw_random_sample(self):
		"""
		Uses the rank 0 process to draw a random sample from the target distribution.
		See the documentation in helpers for more details on how this works.
		Note: since SAMGraph does not gave RNGs, it uses the first RNG of the first 
		SAM module.
		"""
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()

		# Broadcast the sample to all MPI processes.
		if rank == 0:
			rng = list(self.sams.values())[0].rngs[0]
			sample = helpers.draw_from_distribution(self.distribution, complete=True, randomiser=rng)
		else:
			sample = None

		comm.bcast(sample, root=0)

		return sample


	def present_random_sample(self, duration=None):
		"""
		Simulates the network for the given duration while a constant external current
		is presented to each population coding population in the network, chosen randomly 
		to be excitatory or inhibitory from a valid state of the distribution.
		Alpha neurons are inhibited if the value they represent does not match the 
		sample value.
		"""
		if not self.initialised:
			raise Exception("SAM graph not initialised yet.")

		# Get a random state from the distribution.
		state = self.draw_random_sample()

		# Set the currents of the output and hidden layer of each module.
		for ym in self.sams:
			var_index = int(ym[1:]) - 1
			self.sams[ym].set_hidden_currents(state[var_index])
			self.sams[ym].set_output_currents(state[var_index])

		# Simulate.
		nest.Simulate(duration if duration is not None else self.params['sample_presentation_time'])


	def clear_currents(self):
		"""
		Convenience call that unsets all external currents.
		"""
		for ym in self.sams:
			self.sams[ym].clear_currents()


	def set_intrinsic_rate(self, intrinsic_rate):
		"""
		Convenience call that forwards the rate request to the 
		underlying modules.
		"""
		for ym in self.sams:
			self.sams[ym].set_intrinsic_rate(intrinsic_rate)


	def set_plasticity_learning_time(self, learning_time):
		"""
		Convenience call that forwards the learning time request to the
		underlying modules.
		"""
		for ym in self.sams:
			self.sams[ym].set_plasticity_learning_time(learning_time)


	def determine_state(self, spikes, invalid_to_random=True):
		"""
		Given a set of spikes of the network, this determines which of
		the network states the network is in, or whether it is in an 
		invalid state or a zero-state, an invalid state being one in 
		which more than one output neuron is firing at the same time
		for the same RV.
		"""
		state = [0 for i in range(len(self.sams))]
		rng = list(self.sams.values())[0].rngs[0]
		for i in range(len(self.sams)):
			var_name = "y" + str(i + 1)
			module = self.sams[var_name]

			# Are there any spikes of this module's zeta layer?
			spike_found = False
			for j, n in enumerate(module.zeta):
				encoded_value = (j + 1) if not spike_found else -1
				if n in spikes:
					state[i] = encoded_value
					spike_found = True

		# If any state value is greater than the maximum encoded value,
		# this is an invalid state. 
		if any(s == -1 for s in state):
			if not invalid_to_random:
				state = [-1 for i in range(len(self.sams))]
			else:
				state = [rng.choice(self.num_discrete_vals) + 1 if s == -1 else s for s in state]

		return tuple(state)


	def measure_experimental_joint_distribution(self, duration, timestep=1.0):
		"""
		Lets the network generate spontaneous spikes for a long duration
		and then uses the spike activity to calculate the frequency of network 
		states.
		"""
		logging.info("Starting experimental joint distribution measurement on recurrent SAM network.")

		# Attach a spike reader to all population coding layers.
		spikereader = nest.Create('spike_detector', params={'withtime':True, 'withgid':True})
		for ym in self.sams:
			nest.Connect(self.sams[ym].zeta, spikereader, syn_spec={'delay':self.params['devices_delay']})

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

		# Prepare state distribution variables.
		invalid_state = tuple([-1 for i in range(len(self.sams))])
		joint = defaultdict(int)
		invalids = 0
		zeros = 0

		# For every timestep, figure out the network state we are in.
		tau = self.params['tau']
		steps = np.arange(start_time, start_time + duration, timestep)
		for t in steps:
			spike_indices = [i for i, st in enumerate(times) if t - tau < st <= t]
			state_spikes = [senders[i] for i in spike_indices]
			state = self.determine_state(state_spikes)
			joint[state] += 1
			if state in self.distribution: 
				pass
			elif state == invalid_state:
				invalids += 1
			else:
				zeros += 1

		# Normalise all values.
		total = np.sum(list(joint.values()))
		for k, v in joint.items():
			joint[k] = v / total
		invalids /= len(steps)
		zeros /= len(steps)

		logging.info("Probability of an invalid state: {}".format(invalids))
		logging.info("Probability of a zero state: {}".format(zeros))

		return joint


	def draw_stationary_state(self, duration, ax=None):
		"""
		Lets the network spike without external input, and draws the spikes from
		the output neurons (zeta layers).
		"""
		# Attach a spike reader to all population coding layers.
		spikereader = nest.Create('spike_detector', params={'withtime':True, 'withgid':True})
		for ym in self.sams:
			nest.Connect(self.sams[ym].zeta, spikereader, syn_spec={'delay':self.params['devices_delay']})

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
		for ym in range(len(self.sams)):
			var = 'y' + str(ym + 1)
			module = self.sams[var]
			zeta = module.zeta
			for z in zeta:
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
