import logging
import nest
import nest.raster_plot
import numpy as np

from ltl import sdict
from ltl.optimizees.optimizee import Optimizee

logger = logging.getLogger("ltl-sam")
nest.set_verbosity("M_WARNING")


class SAMOptimizee(Optimizee):
    """
    Simple Association Module, an SNN consisting of WTA circuitry that performs unsupervised learning of simple
    multinomial probability distributions to learn associations between variables. Described in Pecevski et al. 2016
    """

    def __init__(self, traj, n_NEST_threads=1):
        super(SAMOptimizee, self).__init__(traj)

        self.n_NEST_threads = n_NEST_threads
        self._initialize()

        # create_individual can be called because __init__ is complete except for traj initialization
        indiv_dict = self.create_individual()
        for key, val in indiv_dict.items():
            traj.individual.f_add_parameter(key, val)

    def _initialize(self):
        # Set parameters of the NEST simulation kernel
        nest.SetKernelStatus({'print_time': False,
                              'local_num_threads': self.n_NEST_threads})

    def create_individual(self):
        jee, jei, jie, jii = np.random.randint(1, 20, 4).astype(np.float64)
        return dict(jee=jee, jei=jei, jie=jie, jii=jii)

    def bounding_func(self, individual):
        individual = {key: np.float64(value if value > 0.01 else 0.01) for key, value in individual.items()}
        return individual

    def prepare_network(self, num_x_vars=2, num_discrete_vals_x=2, num_discrete_vals_z=2, num_modes=2):
        """
        Creates a Stochastic Association Module (SAM), as described in Pecevski, 2016. A SAM is a Winner-Take-All (
        WTA) module that performs unsupervised learning of a target multinomial distribution (density estimation).
        Although in the sense of density estimation it is an unsupervised algorithm (during the learning phase
        samples are simply presented from the target distribution, with no labelling or splitting into input/output),
        the module learns the structure of variable interdependencies, so that withholding one variable and
        presenting samples from the rest to the module causes it to estimate the target conditional probability of
        the withheld variable.

        Note: I have tried to replicate the network architecture and properties described in Pecevski, 2016. However,
        NEST has some limitations that do not entirely allow this. The deviations are listed below:
         1. Neurons cannot have different PSC forms simultaneously. Specifically, the alpha layer should have alpha
         PSPs when stimulated by excitatory connections and rectangular IPSPs when stimulated by inhibitory neurons.
         NEST does not allow this. Instead the inhibitory-alpha weights are increased to make up for this.
         2. NEST deals with PSCs, not PSPs, so there is no simple one-to-one relationship.
         3. NEST does not allow the flexibility of custom STDP equations (unless new models are created).

        Network parameters:
        ****************

        (Excitatory neurons)
        Number of X populations = num_x_vars
        Number of neurons per X population = num_discrete_vals_x
        Number of alpha populations = num_discrete_vals_z
        Number of neurons per alpha population = num_modes
        Number of zeta neurons = num_discrete_vals_z
        Neuron type: SRM w/ alpha EPSP (iaf_psc_alpha)

        (Inhibitory neurons)
        Number of inhibitory neurons = 5

        (Connectivity)
        X-alpha connectivity: all-to-all
        alpha-zeta connectivity: all-to-subpopulation index
        alpha-inh connectivity: all-to-all

        (Weights & Biases)
        alpha-inh weights = w_e2i = 80
        inh biases = b_inh = -10
        population-coding neuron biases (X & zeta) = b_pp = -10
        alpha-zeta weights = w_pp = 20
        w_min = 0
        w_max = 5
        w_- (offset) = 2.5 log 0.2
        plasticity offset (see Pecevski paper; too long to reproduce here)
        inh-alpha weights = w_i2e = -7
        b_min = -30
        b_max = 5
        Initial random weights: gaussian dist, mean w_init = 3, sigma_w0 = 0.1, redrawn to within [w_min, w_max]
        Initial random biases: gaussian dist, mean b_init = 5, sigma_b0 = 0.1, redrawn to within [w_min, w_max]

        (Learning & Inference)
        Samples generated from target distribution
        Neuron v^kl inj current = i_pp+ = 30 if y^k = l
        Neuron v^kl inj current = i_pp- = -30 if y^k != l
        Inh. current = -80, inj. in alpha subpops that do not trigger the zeta neuron corresponding to y^k = l (only
        during learning)
        Scaling term for potentiation (both for w and b) = T = 0.4
        Learning process lasts 1200 s of biological time
        For weights, nita (learning rate) decreased during first 600 s from 0.002 at t = 0 to 0.0006 at t = 600 s.
        For weights, nita set to 0 (no synaptic plasticity) during second 600 s.
        For biases, nita = 0.01 during first 600 s.
        For biases, nita = 0.02 during second 600 s.

        (Model parameters)
        tau (abs. refractory period) = 15 ms
        STDP time-window = tau
        alpha-shape EPSP epsilon_o = 2.8
        alpha-shape EPSP time constant = tau_alpha = 8.5 ms
        From/to inh. neurons, PSP is rectangular with duration tau

        """

        x = nest.Create("iaf_psc_alpha", num_discrete_vals_x * num_x_vars)
        alpha = nest.Create("iaf_psc_alpha", num_discrete_vals_z * num_modes)
        z = nest.Create("iaf_psc_alpha", num_discrete_vals_z)
        inh = nest.Create("")

    def simulate(self, traj, should_plot=False, debug=False):

        jee = traj.individual.jee
        jei = traj.individual.jei
        jie = traj.individual.jie
        jii = traj.individual.jii

        if jee < 0 or jei < 0 or jie < 0 or jii < 0:
            return (np.inf,)

        logger.info("Running for %.2f, %.2f, %.2f, %.2f", jee, jei, jie, jii)

        simtime_ms = 1000.0

        # Delay distribution.
        delay_dict = dict(distribution='normal_clipped', mu=10., sigma=20., low=3., high=200.)

        # Create nodes -------------------------------------------------
        nest.ResetKernel()
        self._initialize()
        nest.SetDefaults('iaf_psc_exp',
                         {'C_m': 30.0,  # 1.0,
                          'tau_m': 30.0,
                          'E_L': 0.0,
                          'V_th': 15.0,
                          'tau_syn_ex': 3.0,
                          'tau_syn_in': 2.0,
                          'V_reset': 13.8})

        # Create excitatory and inhibitory populations
        noise_ex = nest.Create("poisson_generator")
        noise_in = nest.Create("poisson_generator")
        nest.SetStatus(noise_ex, {"rate": 80000.0})
        nest.SetStatus(noise_in, {"rate": 15000.0})

        neurons1 = nest.Create('iaf_psc_alpha', 10000)
        neuron2 = nest.Create('iaf_psc_alpha')
        nest.SetStatus(neurons1, {"I_e": 0.0})
        nest.SetStatus(neuron2, {"I_e": 0.0})

        syn_dict_ex = {"weight": 1.2}
        syn_dict_in = {"weight": -2.0}
        nest.Connect(noise_ex, neurons1, syn_spec=syn_dict_ex)
        nest.Connect(noise_in, neurons1, syn_spec=syn_dict_in)
        nest.Connect(neurons1, neuron2, syn_spec={"weight": 7.0,
                                                  "delay": 1.0})

        multimeter = nest.Create('multimeter')
        nest.SetStatus(multimeter, {'withtime': True, 'record_from': ['V_m']})

        spikedetector = nest.Create('spike_detector',
                                    params={'withgid': True, 'withtime': True})

        nest.Connect(multimeter, neuron2)
        nest.Connect(neuron2, spikedetector)

        # SIMULATE!! -----------------------------------------------------
        nest.Simulate(simtime_ms)

        dmm = nest.GetStatus(multimeter)[0]
        Vms1 = dmm["events"]["V_m"]
        ts1 = dmm["events"]["times"]

        import pylab

        pylab.figure(1)
        pylab.plot(ts1, Vms1)

        dSD = nest.GetStatus(spikedetector, keys="events")[0]
        evs1 = dSD["senders"]
        ts1 = dSD["times"]

        pylab.figure(2)
        pylab.plot(ts1, evs1, ".")

        pylab.show()

        return 0, 0


def end(self):
    logger.info("End of all experiments. Cleaning up...")
    # There's nothing to clean up though


def main():
    import yaml
    import os
    import logging.config

    from ltl import DummyTrajectory
    from ltl.paths import Paths
    from ltl import timed

    try:
        with open('bin/path.conf') as f:
            root_dir_path = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            "You have not set the root path to store your results_kris."
            " Write the path to a path.conf text file in the bin directory"
            " before running the simulation"
        )
    paths = Paths('ltl-sam', dict(run_num='test'), root_dir_path=root_dir_path)
    with open("bin/logging.yaml") as f:
        l_dict = yaml.load(f)
        log_output_file = os.path.join(paths.results_path, l_dict['handlers']['file']['filename'])
        l_dict['handlers']['file']['filename'] = log_output_file
        logging.config.dictConfig(l_dict)

    fake_traj = DummyTrajectory()
    optimizee = SAMOptimizee(fake_traj, n_NEST_threads=4)

    fake_traj.individual = sdict(optimizee.create_individual())

    with timed(logger):
        testing_error = optimizee.simulate(fake_traj, debug=True)
    logger.info("Testing error is %s", testing_error)


if __name__ == "__main__":
    main()
