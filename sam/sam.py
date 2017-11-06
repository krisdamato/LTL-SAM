import helpers
import nest
import numpy as np
import pylab
import time

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
		Initialises RNGs and network settings with default values.
		"""
		self.set_defaults()
		self.initialised = False

		# Randomise.
		seed = int(time.time()) if randomise_seed else 705
		n = nest.GetKernelStatus(['total_num_virtual_procs'])[0]

		print("Randomising {} processes using {} as main seed.".format(n * 2 + 1, seed))

		# Seed Python, global, and per-process RNGs.
		self.rngs = [np.random.RandomState(s) for s in range(seed, seed + n)]
		nest.SetKernelStatus({'grng_seed' : seed + n})
		nest.SetKernelStatus({'rng_seeds' : range(seed + n + 1, seed + 2 * n + 1)})


	def set_defaults(self, params={}):
		"""
		Sets SAM defaults to Pecevski et al. 2016 values.
		params: dictionary of parameters to set.
		"""
		# Set defaults.
		tau = 15.0
		learning_phase_1 = 600000 # 600 s
		learning_phase_2 = 600000 

		self.params = {
			'initial_stdp_rate':0.002,
			'final_stdp_rate':0.0006,
			'stdp_time':learning_phase_1, # 600 s
			'initial_weight':3.0,
			'max_weight':5.0,
			'min_weight':0.0,
			'weight_baseline':2.5 * np.log(0.2),
			'tau':tau,
			'T':0.4,
			'use_rect_psp_exc':False,
			'use_rect_psp_inh':True,
			'amplitude_exc':2.8,
			'amplitude_inh':2.8,
			'external_current':0.0,
			'tau_membrane':15.0,
			'tau_exc':8.5,
			'first_bias_rate':0.01,
			'second_bias_rate':0.02,
			'bias_baseline':0.0,
			'max_bias':5.0,
			'min_bias':-30.0,
			'default_random_bias':False,
			'initial_bias':5.0,
			'dead_time_random':False,
			'linear_term_prob':0.0,
			'exp_term_prob':1/tau,
			'exp_term_prob_scale':1.0,
			'weight_chi_alpha_mean':3.0,
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
			'first_learning_phase':learning_phase_1,
			'second_learning_phase':learning_phase_2,
			'sample_presentation_time':100.0, # 100 ms.
			'chi_neuron_type':'srm_pecevski_alpha',
			'alpha_neuron_type':'srm_pecevski_alpha',
			'zeta_neuron_type':'srm_pecevski_alpha',
			'inhibitors_neuron_type':'srm_pecevski_alpha',
			'chi_alpha_synapse_type':'stdp_pecevski_synapse',
			'alpha_zeta_synapse_type':'static_synapse',
			'alpha_inhibitors_synapse_type':'static_synapse',
			'inhibitors_alpha_synapse_type':'static_synapse',
			'num_inhibitors':5
		}

		# Update defaults.
		self.params.update(params)

		# Set NEST defaults.
		nest.SetDefaults('static_synapse', params={'weight':self.params['initial_weight']})

		nest.SetDefaults('stdp_pecevski_synapse', params={
			'eta_0':self.params['initial_stdp_rate'],
			'eta_final':self.params['final_stdp_rate'], 
			'learning_time':self.params['stdp_time'],
			'max_weight':self.params['max_weight'],
			'min_weight':self.params['min_weight'],
			'weight':self.params['initial_weight'],
			'w_baseline':self.params['weight_baseline'],
			'tau':self.params['tau'],
			'T':self.params['T']
			})

		nest.SetDefaults('srm_pecevski_alpha', params={
			'rect_exc':self.params['use_rect_psp_exc'], 
			'rect_inh':self.params['use_rect_psp_inh'],
			'e_0_exc':self.params['amplitude_exc'], 
			'e_0_inh':self.params['amplitude_inh'],
			'I_e':self.params['external_current'],
			'tau_m':self.params['tau_membrane'], # Membrane time constant (only affects current injections). Is this relevant?
			'tau_exc':self.params['tau_exc'],
			'tau_inh':self.params['tau'], 
			'tau_bias':self.params['tau'],
			'eta_bias': self.params['first_bias_rate'], # eta_bias is set to 0.02 manually after 600 s.
			'b_baseline':self.params['bias_baseline'], # This is set by a call elsewhere.
			'max_bias':self.params['max_bias'],
			'min_bias':self.params['min_bias'],
			'mu_bias':self.params['bias_alpha_mean'],
			'sigma_bias':self.params['bias_alpha_std'],
			'use_random_bias':self.params['default_random_bias'],
			'bias':self.params['initial_bias'],
			'dead_time':self.params['tau'], # Abs. refractory period.
			'dead_time_random':self.params['dead_time_random'],
			'c_1':self.params['linear_term_prob'], # Linear part of transfer function.
			'c_2':self.params['exp_term_prob'], # The coefficient of the exponential term in the transfer function.
			'c_3':self.params['exp_term_prob_scale'], # Scaling coefficient of effective potential in exponential term.
			'T':self.params['T']
			})

		# Create layer model specialisations.
		# Input population model.
		if self.params['chi_neuron_type'] == 'srm_pecevski_alpha':
			self.chi_neuron_params={"bias":self.params['bias_chi']}
		else:
			raise NotImplementedError

		# Hidden population model.
		if self.params['alpha_neuron_type'] == 'srm_pecevski_alpha':
			self.alpha_neuron_params={
				'use_random_bias':True,
				'mu_bias':self.params['bias_alpha_mean'],
				'sigma_bias':self.params['bias_alpha_std']
				}
		else:
			raise NotImplementedError

		# Output population model.
		if self.params['zeta_neuron_type'] == 'srm_pecevski_alpha':
			self.zeta_neuron_params={"bias":self.params['bias_zeta']}
		else:
			raise NotImplementedError

		# Inhibitor population model.
		if self.params['inhibitors_neuron_type'] == 'srm_pecevski_alpha':
			self.inhibitors_neuron_params={"bias":self.params['bias_inhibitors']}
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
					}
				}
		else:
			raise NotImplementedError

		# Alpha-inhibitors synapses.
		if self.params['alpha_inhibitors_synapse_type'] == 'static_synapse':
			self.alpha_inhibitors_synapse_params={'weight':self.params['weight_alpha_inhibitors']}
		else:
			raise NotImplementedError

		# Inhibitors-alpha synapses.
		if self.params['inhibitors_alpha_synapse_type'] == 'static_synapse':
			self.inhibitors_alpha_synapse_params={'weight':self.params['weight_inhibitors_alpha']}
		else:
			raise NotImplementedError

		# Alpha-zeta synapses.
		if self.params['alpha_zeta_synapse_type'] == 'static_synapse':
			self.alpha_zeta_synapse_params={'weight':self.params['weight_alpha_zeta']}
		else:
			raise NotImplementedError


	def create_network(self, num_x_vars=2, num_discrete_vals_x=2, num_discrete_vals_z=2, num_modes=2):
		"""
		Creates a SAM module with the specified architecture.
		"""

		# Set bias baseline.
		bias_baseline = helpers.determine_bias_baseline(self.params['T'], [num_discrete_vals_x for i in range(num_x_vars)])
		self.params['bias_baseline'] = bias_baseline

		# Create neuron layers.
		self.chi = nest.Create(self.params['chi_neuron_type'], n=num_discrete_vals_x * num_x_vars, params=self.chi_neuron_params)
		self.alpha = nest.Create(self.params['alpha_neuron_type'], n=num_modes * num_discrete_vals_z, params=self.alpha_neuron_params)
		self.zeta = nest.Create(self.params['zeta_neuron_type'], n=num_discrete_vals_z, params=self.zeta_neuron_params)
		self.inhibitors = nest.Create(self.params['inhibitors_neuron_type'], n=self.params['num_inhibitors'], params=self.inhibitors_neuron_params)

		# Track all neurons.
		self.all_neurons = self.chi + self.alpha + self.zeta + self.inhibitors

		# Connect layers.
		# Chi-alpha connectivity: all-to-all.
		nest.Connect(self.chi, self.alpha, conn_spec='all_to_all', syn_spec=self.chi_alpha_synapse_params)

		# Alpha-inhibitors connectivity: all-to-all.
		nest.Connect(self.alpha, self.inhibitors, conn_spec='all_to_all', syn_spec=self.alpha_inhibitors_synapse_params)

		# Inhibitors-alpha connectivity: all-to-all.
		nest.Connect(self.inhibitors, self.alpha, conn_spec='all_to_all', syn_spec=self.inhibitors_alpha_synapse_params)

		# Alpha-zeta connectivity: subpopulation-to-subpopulation-index.
		for i in range(len(self.alpha)):
			j = i // num_modes
			nest.Connect([self.alpha[i]], [self.zeta[j]], syn_spec=self.alpha_zeta_synapse_params)

		# Set initialised flag.
		self.initialised = True


	def get_local_nodes(self, neurons):
		"""
		Returns a list of (node ID, process ID) pairs for all nodes that were
		created by this process.
		"""
		node_info = nest.GetStatus(neurons)
		return [(ni['global_id'], ni['vp']) for ni in node_info if ni['local']]


	def inject_random_current(self, duration):
		"""
		Simulates the network for the given duration while a constant external current
		is presented to each population coding neuron in the network, chosen randomly 
		to be excitatory or inhibitory.
		"""

		def set_currents(neurons):
			'Sets randomly chosen excitatory/inhibitory currents for passed neurons.'
			for gid, vp in self.get_local_nodes(neurons):
				rand = self.rngs[vp].randint(2) # Random integer in {0, 1}
				current = self.params['nu_current_minus'] if rand == 0 else self.params['nu_current_plus']
				print("Inhibiting {}".format(gid) if rand == 0 else "Activating {}".format(gid))
				nest.SetStatus([gid], {'I_e':current})

		if not self.initialised:
			raise Exception("SAM module not initialised yet.")

		# Set currents in input and output layers.
		set_currents(self.chi)
		set_currents(self.zeta)

		# Create a spike reader.
		spikereader = nest.Create('spike_detector', params={'withtime':True, 'withgid':True})

		# Connect all neurons to the spike reader.
		nest.Connect(self.all_neurons, spikereader)

		# Simulate.
		nest.Simulate(duration / 2)

		# Remove external currents.
		nest.SetStatus(self.all_neurons, {'I_e':0.0})

		# Simulate.
		nest.Simulate(duration / 2)

		# Plot spikes.
		self.plot_spikes(spikereader)


	def plot_spikes(self, spikereader):
		"""
		Connects a spike reader to each neuron and plots the spike trace over a simulation
		with the given duration.
		"""
		# Get spikes and plot.
		spikes = nest.GetStatus(spikereader, keys='events')[0]
		senders = spikes['senders']
		times = spikes['times']

		pylab.figure()
		pylab.plot(times, senders, '|')
		pylab.show()


	def print_network_info(self):
		"""
		Prints network architecture info.
		"""
		print("baseline =", self.params['bias_baseline'])
		print(self.all_neurons)
		

