import logging
import matplotlib.pyplot as plt
import nest
import nest.raster_plot
import numpy as np
import sam.helpers as helpers
import sam.tests as tests

from collections import OrderedDict
from ltl import sdict
from ltl.optimizees.optimizee import Optimizee
from sam.sam import SAMModule

logger = logging.getLogger("ltl-sam")
nest.set_verbosity("M_WARNING")


class SAMOptimizee(Optimizee):
    """
    Provides the interface between the LTL API and the SAM class. See SAMModule for details on
    the SAM neural network.
    """

    def __init__(self, traj, n_NEST_threads=1, seed=0):
        super(SAMOptimizee, self).__init__(traj)

        # Make SAM module extension available.
        nest.Install('sammodule')
        
        self.num_threads = n_NEST_threads
        self.rs = np.random.RandomState(seed=seed)

        # create_individual can be called because __init__ is complete except for traj initialization
        self.individual = self.create_individual()
        for key, val in self.individual.items():
            traj.individual.f_add_parameter(key, val)


    def create_individual(self):
        """
        Creates random parameter values within given bounds.
        Uses an RNG seeded with the main said of the SAM module.
        """ 
        param_spec = OrderedDict(sorted(SAMModule.parameter_spec().items())) # Sort for replicability
        individual = {k: np.float64(self.rs.uniform(v[0], v[1])) for k, v in param_spec.items()}
        return individual


    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping
        """
        param_spec = SAMModule.parameter_spec()
        individual = {k: np.float64(np.clip(v, a_min=param_spec[k][0], a_max=param_spec[k][1])) for k, v in individual.items()}
        return individual


    def prepare_network(self, distribution, num_discrete_vals, num_modes):
        """
        Generates a network with the specified distribution parameters, but uses
        the hyperparameters from the individual dictionary.
        """
        self.sam = SAMModule(randomise_seed=True)

        # Find the number of variables.
        num_vars = len(list(distribution.keys())[0])

        # Convert the trajectory individual to a dictionary.
        params = {k:self.individual[k] for k in SAMModule.parameter_spec().keys()}

        # Create a SAM module with the correct parameters.
        params['num_threads'] = self.num_threads

        self.sam.create_network(num_x_vars=num_vars - 1, 
            num_discrete_vals=num_discrete_vals, 
            num_modes=num_modes,
            distribution=distribution,
            params=params)

        logging.info("Creating a SAM network with overridden parameters:\n%s", helpers.get_dictionary_string(params))


    def simulate(self, traj, show_plot=False):
        """
        Simulates a SAM module training on a target distribution; i.e. performing
        density estimation as in Pecevski et al. 2016. The loss function is the
        negative of the KL divergence between target and estimated distributions.
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

        nest.ResetKernel()
        self.individual = traj.individual
        self.prepare_network(distribution=distribution, num_discrete_vals=2, num_modes=2)
        
        # Get the conditional of the module's target distribution.
        distribution = self.sam.distribution
        conditional = helpers.compute_conditional_distribution(self.sam.distribution, self.sam.num_discrete_vals)

        # Train for the learning period set in the parameters.
        t = 0
        i = 0
        set_second_rate = False
        last_set_intrinsic_rate = self.sam.params['first_bias_rate']
        skip_kld = 10   
        skip_exp_cond = 1000
        kls_joint = []
        kls_cond = []
        kls_cond_exp = []
        set_second_rate = False
        extra_time = 0

        while t <= self.sam.params['learning_time']:
            # Inject a current for some time.
            self.sam.present_random_sample() 
            self.sam.clear_currents()
            t += self.sam.params['sample_presentation_time']

            # Compute theoretical distributions and measure KLD.
            if show_plot and i % skip_kld == 0:
                implicit = self.sam.compute_implicit_distribution()
                implicit_conditional = helpers.compute_conditional_distribution(implicit, 2)
                kls_joint.append(helpers.get_KL_divergence(implicit, distribution))
                kls_cond.append(helpers.get_KL_divergence(implicit_conditional, conditional))

            # Measure experimental conditional distribution from spike activity.
            if show_plot and i % skip_exp_cond == 0:
                # Stop plasticity for testing.
                self.sam.set_intrinsic_rate(0.0)
                self.sam.set_plasticity_learning_time(0)
                experimental_conditional = tests.measure_experimental_cond_distribution(self.sam, duration=2000.0)
                kls_cond_exp.append(helpers.get_KL_divergence(experimental_conditional, conditional))

                # Restart plasticity.
                extra_time += 2000
                self.sam.set_intrinsic_rate(last_set_intrinsic_rate)
                self.sam.set_plasticity_learning_time(int(self.sam.params['stdp_time_fraction'] * self.sam.params['learning_time'] + extra_time))
                self.sam.clear_currents()

            # Set different intrinsic rate.
            if t >= self.sam.params['learning_time'] * self.sam.params['intrinsic_step_time_fraction'] and set_second_rate == False:
                set_second_rate = True
                last_set_intrinsic_rate = self.sam.params['second_bias_rate']
                self.sam.set_intrinsic_rate(last_set_intrinsic_rate)
        
            i += 1

        self.sam.set_intrinsic_rate(0.0)
        self.sam.set_plasticity_learning_time(0)

        # Plot KL divergence plot.
        if show_plot:
            # Present evidence.
            tests.run_trained_test(self.sam)

            plt.figure()
            plt.plot(np.array(range(len(kls_cond))) * skip_kld * self.sam.params['sample_presentation_time'] * 1e-3, kls_cond, label="KLd p(z|x)")
            plt.plot(np.array(range(len(kls_joint))) * skip_kld * self.sam.params['sample_presentation_time'] * 1e-3, kls_joint, label="KLd p(x,z)")
            plt.plot(np.array(range(len(kls_cond_exp))) * skip_exp_cond * self.sam.params['sample_presentation_time'] * 1e-3, kls_cond_exp, label="Exp. KLd p(z|x)")
            plt.legend(loc='upper center')
            plt.show()

        # Calculate final divergences.
        implicit = self.sam.compute_implicit_distribution()
        implicit_conditional = helpers.compute_conditional_distribution(implicit, self.sam.num_discrete_vals)
        kld_joint = helpers.get_KL_divergence(implicit, distribution)
        kld_cond = helpers.get_KL_divergence(implicit_conditional, conditional)

        logging.info("Final loss is {}".format(kld_joint))

        return (kld_joint, )


def end(self):
    logger.info("End of experiment. Cleaning up...")
    

def main():
    import yaml
    import os
    import logging.config

    from ltl import DummyTrajectory
    from ltl.paths import Paths
    from ltl import timed

    try:
        with open('../bin/path.conf') as f:
            root_dir_path = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            "You have not set the root path to store your results_kris."
            " Write the path to a path.conf text file in the bin directory"
            " before running the simulation"
        )
    paths = Paths('ltl-sam', dict(run_num='test'), root_dir_path=root_dir_path)
    with open("../bin/logging.yaml") as f:
        l_dict = yaml.load(f)
        log_output_file = os.path.join(paths.results_path, l_dict['handlers']['file']['filename'])
        l_dict['handlers']['file']['filename'] = log_output_file
        logging.config.dictConfig(l_dict)

    fake_traj = DummyTrajectory()
    optimizee = SAMOptimizee(fake_traj, n_NEST_threads=1)

    fake_traj.individual = sdict(optimizee.create_individual())

    with timed(logger):
        loss = optimizee.simulate(fake_traj, show_plot=True)
    logging.info("Final loss is {}".format(loss))


if __name__ == "__main__":
    main()
