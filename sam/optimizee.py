import logging
import matplotlib.pyplot as plt
import nest
import nest.raster_plot
import numpy as np
import os
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

    def __init__(
            self, 
            traj, 
            time_resolution=0.05, 
            num_fitness_trials=3, 
            seed=0, 
            n_NEST_threads=1, 
            plots_directory='./sam_plots'):
        super(SAMOptimizee, self).__init__(traj)

        # Make SAM module extension available.
        nest.Install('sammodule')
        nest.SetKernelStatus({
            'local_num_threads':n_NEST_threads,
            'resolution':time_resolution})
        
        self.rs = np.random.RandomState(seed=seed)
        self.num_fitness_trials = num_fitness_trials
        self.run_number = 0 # Is this NEST-process safe?
        self.save_directory = plots_directory

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
        Bounds the individual within the required bounds via coordinate clipping.
        """
        param_spec = SAMModule.parameter_spec()
        individual = {k: np.float64(np.clip(v, a_min=param_spec[k][0], a_max=param_spec[k][1])) for k, v in individual.items()}
        return individual


    def parameter_spec(self):
        """
        Returns the minima-maxima of each explorable variable.
        Note: Dictionary is an OrderedDict with items sorted by key, to 
        ensure that items are interpreted in the same way everywhere.
        """
        return OrderedDict(sorted(SAMModule.parameter_spec().items()))


    def prepare_network(self, distribution, num_discrete_vals, num_modes):
        """
        Generates a network with the specified distribution parameters, but uses
        the hyperparameters from the individual dictionary.
        """
        nest.ResetKernel()
        self.sam = SAMModule(randomise_seed=True)

        # Find the number of variables.
        num_vars = len(list(distribution.keys())[0])

        # Convert the trajectory individual to a dictionary.
        params = {k:self.individual[k] for k in SAMModule.parameter_spec().keys()}

        # Create a SAM module with the correct parameters.
        self.sam.create_network(num_x_vars=num_vars - 1, 
            num_discrete_vals=num_discrete_vals, 
            num_modes=num_modes,
            distribution=distribution,
            params=params)

        logging.info("Creating a SAM network with overridden parameters:\n%s", helpers.get_dictionary_string(params))


    def simulate(self, traj, save_plot=True):
        """
        Simulates a SAM module training on a target distribution; i.e. performing
        density estimation as in Pecevski et al. 2016. The loss function is the
        the KL divergence between target and estimated distributions.
        If save_plot == True, will create a directory for each individual that 
        contains a text file with individual params and plots for each trial.
        """
        # Use the distribution from Peceveski et al., example 1.
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

        # Prepare paths for each individual evaluation.
        individual_directory = os.path.join(self.save_directory, str(self.run_number) + "_" + helpers.get_now_string())
        text_path = os.path.join(individual_directory, 'params.txt')
        
        # Declare fitness metrics.
        kld_joint = []
        kld_conditional = []
        kld_conditional_experimental = []

        # Run a number of trials to calculate mean fitness of this individual
        for trial in range(self.num_fitness_trials):
            nest.ResetKernel()
            self.individual = traj.individual
            self.prepare_network(distribution=distribution, num_discrete_vals=2, num_modes=2)
            
            # Create directory and params file if requested.
            if save_plot:
                params_dict = OrderedDict(sorted(self.sam.params.items()))
                helpers.create_directory(individual_directory)
                helpers.save_text(helpers.get_dictionary_string(params_dict), text_path)

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
            sam_clones = []

            connections = nest.GetConnections(self.sam.chi, self.sam.alpha)

            while t <= self.sam.params['learning_time']:
                # Inject a current for some time.
                self.sam.present_random_sample() 
                self.sam.clear_currents()
                t += self.sam.params['sample_presentation_time']

                # Compute theoretical distributions and measure KLD.
                if save_plot and i % skip_kld == 0:
                    implicit = self.sam.compute_implicit_distribution()
                    implicit_conditional = helpers.compute_conditional_distribution(implicit, 2)
                    kls_joint.append(helpers.get_KL_divergence(implicit, distribution))
                    kls_cond.append(helpers.get_KL_divergence(implicit_conditional, conditional))

                # Measure experimental conditional distribution from spike activity.
                if save_plot and i % skip_exp_cond == 0:
                    # Clone module for later tests.
                    sam_clone = self.sam.clone()

                    # Stop plasticity on clone for testing.
                    sam_clone.set_intrinsic_rate(0.0)
                    sam_clone.set_plasticity_learning_time(0)

                    sam_clones.append(sam_clone)
                                    
                # Set different intrinsic rate.
                if t >= self.sam.params['learning_time'] * self.sam.params['intrinsic_step_time_fraction'] and set_second_rate == False:
                    set_second_rate = True
                    last_set_intrinsic_rate = self.sam.params['second_bias_rate']
                    self.sam.set_intrinsic_rate(last_set_intrinsic_rate)
            
                i += 1

            self.sam.set_intrinsic_rate(0.0)
            self.sam.set_plasticity_learning_time(0)

            # Measure experimental conditional on para-experiment clones.
            if save_plot:
                plot_exp_conditionals = [tests.measure_experimental_cond_distribution(s, duration=2000.0) for s in sam_clones]
                kls_cond_exp = [helpers.get_KL_divergence(p, conditional) for p in plot_exp_conditionals] 

            # Plot KL divergence plot.
            if save_plot:
                plt.figure()
                plt.plot(np.array(range(len(kls_cond))) * skip_kld * self.sam.params['sample_presentation_time'] * 1e-3, kls_cond, label="KLd p(z|x)")
                plt.plot(np.array(range(len(kls_joint))) * skip_kld * self.sam.params['sample_presentation_time'] * 1e-3, kls_joint, label="KLd p(x,z)")
                plt.plot(np.array(range(len(kls_cond_exp))) * skip_exp_cond * self.sam.params['sample_presentation_time'] * 1e-3, kls_cond_exp, label="Exp. KLd p(z|x)")
                plt.legend(loc='upper center')
                plt.savefig(os.path.join(individual_directory, str(trial) + '.png'))

            # Calculate final divergences.
            implicit = self.sam.compute_implicit_distribution()
            implicit_conditional = helpers.compute_conditional_distribution(implicit, self.sam.num_discrete_vals)
            kld_joint.append(helpers.get_KL_divergence(implicit, distribution))
            kld_conditional.append(helpers.get_KL_divergence(implicit_conditional, conditional))

            # Measure experimental conditional for the last time by averaging on 3 runs.
            last_clone = self.sam.clone()
            experimental_conditionals = [tests.measure_experimental_cond_distribution(last_clone, duration=2000.0) for i in range(3)]
            kld_conditional_experimental.append(np.sum([helpers.get_KL_divergence(p, conditional) for p in experimental_conditionals]) / len(experimental_conditionals))

        logging.info("Mean theoretical J. KLd is {} [Loss]".format(np.sum(kld_joint) / self.num_fitness_trials))
        logging.info("Mean theoretical C. KLd is {}".format(np.sum(kld_conditional) / self.num_fitness_trials))
        logging.info("Mean experimental C. KLd is {}".format(np.sum(kld_conditional_experimental) / self.num_fitness_trials))

        self.run_number += 1

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
            "You have not set the root path to store your results."
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
