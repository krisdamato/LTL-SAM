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
from sam.samgraph import SAMGraph
from sam.spinetwork import SPINetwork


logger = logging.getLogger("ltl-sam")
nest.set_verbosity("M_WARNING")
nest.Install('sammodule')


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
	    use_pecevski=False,
            plots_directory='./sam_plots'):
        super(SAMOptimizee, self).__init__(traj)
        
        self.rs = np.random.RandomState(seed=seed)
        self.num_fitness_trials = num_fitness_trials
        self.run_number = 0 # Is this NEST-process safe?
        self.save_directory = plots_directory
        self.time_resolution = time_resolution
        self.num_threads = n_NEST_threads
        self.use_pecevski = use_pecevski
        self.set_kernel_defaults()
        self.initialise_distributions()                	

        # create_individual can be called because __init__ is complete except for traj initialization
        self.individual = self.create_individual()
        for key, val in self.individual.items():
            traj.individual.f_add_parameter(key, val)


    def set_kernel_defaults(self):
        """
        Sets main NEST parameters.
        Note: this needs to be called every time the kernel is reset.
        """
        nest.SetKernelStatus({
            'local_num_threads':self.num_threads,
            'resolution':self.time_resolution})


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
        self.set_kernel_defaults()
        self.sam = SAMModule(randomise_seed=True)

        # Find the number of variables.
        num_vars = len(list(distribution.keys())[0])

        # Convert the trajectory individual to a dictionary.
        params = {k:self.individual[k] for k in SAMModule.parameter_spec().keys()}

        # Peg the delay to the time resolution.
        params['delay'] = self.time_resolution
        params['weight_chi_alpha_mean'] = 3.0

        # Create a SAM module with the correct parameters.
        self.sam.create_network(num_x_vars=num_vars - 1, 
            num_discrete_vals=num_discrete_vals, 
            num_modes=num_modes,
            distribution=distribution,
            params=params)

        logging.info("Creating a SAM network with overridden parameters:\n%s", helpers.get_dictionary_string(params))


    def initialise_distributions(self):
        """
        Creates a set of distributions, one for each trial. Each distribution
        has the same decomposition, but uses randomly generated parameters.
        """
        self.distributions = []
        
        for i in range(self.num_fitness_trials):
            if self.use_pecevski:
                p = {
            (1,1,1):0.04,
            (1,1,2):0.04,
            (1,2,1):0.21,
            (1,2,2):0.21,
            (2,1,1):0.04,
            (2,1,2):0.21,
            (2,2,1):0.21,
            (2,2,2):0.04
        }
            else:
                p = helpers.generate_distribution(num_vars=3, num_discrete_values=2, randomiser=self.rs)
            self.distributions.append(p)
            
        # Define the Markov blanket of each RV.
        self.dependencies = {
            'y3':['y1', 'y2']
        }


    def simulate(self, traj, save_plot=True):
        """
        Simulates a SAM module training on a target distribution; i.e. performing
        density estimation as in Pecevski et al. 2016. The loss function is the
        the KL divergence between target and estimated distributions.
        If save_plot == True, will create a directory for each individual that 
        contains a text file with individual params and plots for each trial.
        """        
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
            self.set_kernel_defaults()

            self.individual = traj.individual
            self.prepare_network(distribution=self.distributions[trial], num_discrete_vals=2, num_modes=2)
            
            # Create directory and params file if requested.
            if save_plot:
                params_dict = OrderedDict(sorted(self.sam.params.items()))
                helpers.create_directory(individual_directory)
                helpers.save_text(helpers.get_dictionary_string(params_dict), text_path)

            # Get the conditional of the module's target distribution.
            distribution = self.sam.distribution
            conditional = helpers.compute_conditional_distribution(self.sam.distribution, self.sam.num_discrete_vals, self.sam.output_index)

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

            while t <= self.sam.params['learning_time']:
                # Inject a current for some time.
                self.sam.present_random_sample() 
                self.sam.clear_currents()
                t += self.sam.params['sample_presentation_time']

                # Compute theoretical distributions and measure KLD.
                if save_plot and i % skip_kld == 0:
                    implicit = self.sam.compute_implicit_distribution()
                    implicit_conditional = helpers.compute_conditional_distribution(implicit, self.sam.num_discrete_vals, self.sam.output_index)
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
                plt.close()

            # Calculate final divergences.
            implicit = self.sam.compute_implicit_distribution()
            implicit_conditional = helpers.compute_conditional_distribution(implicit, self.sam.num_discrete_vals)
            kld_joint.append(helpers.get_KL_divergence(implicit, distribution))
            kld_conditional.append(helpers.get_KL_divergence(implicit_conditional, conditional))

            # Print experimental distribution.
            if save_plot:
                logging.info("Implicit conditional distribution:\n{}".format(implicit_conditional))
                fig = helpers.plot_3d_histogram(conditional, implicit_conditional, self.sam.num_discrete_vals, target_label='p*(z|x)', estimated_label='p(z|x;θ)')
                fig.savefig(os.path.join(individual_directory, str(trial) + '_conditional.png'))
                fig = helpers.plot_3d_histogram(distribution, implicit, self.sam.num_discrete_vals, target_label='p*(x,z)', estimated_label='p(x,z;θ)')
                fig.savefig(os.path.join(individual_directory, str(trial) + '_joint.png'))
                plt.close()

            # Measure experimental conditional for the last time by averaging on 3 runs.
            last_clone = self.sam.clone()
            experimental_conditionals = [tests.measure_experimental_cond_distribution(last_clone, duration=2000.0) for i in range(3)]
            kld_conditional_experimental.append(np.sum([helpers.get_KL_divergence(p, conditional) for p in experimental_conditionals]) / len(experimental_conditionals))

        logging.info("Mean theoretical J. KLd is {} [Loss]".format(np.sum(kld_joint) / self.num_fitness_trials))
        logging.info("Mean theoretical C. KLd is {}".format(np.sum(kld_conditional) / self.num_fitness_trials))
        logging.info("Mean experimental C. KLd is {}".format(np.sum(kld_conditional_experimental) / self.num_fitness_trials))

        self.run_number += 1

        return (np.sum(kld_conditional_experimental) / self.num_fitness_trials, )


    def end(self):
        logger.info("End of experiment. Cleaning up...")
    

class SAMGraphOptimizee(Optimizee):
    """
    Provides the interface between the LTL API and the SAMGraph class. See SAMModule and
    SAMGraph for details on the SAM neural network and simulated graph model.
    """

    def __init__(
            self, 
            traj, 
            time_resolution=0.05, 
            num_fitness_trials=3, 
            seed=0, 
            n_NEST_threads=1, 
            plots_directory='./sam_plots'):
        super(SAMGraphOptimizee, self).__init__(traj)
        
        self.rs = np.random.RandomState(seed=seed)
        self.num_fitness_trials = num_fitness_trials
        self.run_number = 0 # Is this NEST-process safe?
        self.save_directory = plots_directory
        self.time_resolution = time_resolution
        self.num_threads = n_NEST_threads
        self.set_kernel_defaults()

        # Set up exerimental parameters.
        #self.initialise_experiment()
        self.intitialise_distributions()

        # create_individual can be called because __init__ is complete except for traj initialization
        self.individual = self.create_individual()
        for key, val in self.individual.items():
            traj.individual.f_add_parameter(key, val)


    def set_kernel_defaults(self):
        """
        Sets main NEST parameters.
        Note: this needs to be called every time the kernel is reset.
        """
        nest.SetKernelStatus({
            'local_num_threads':self.num_threads,
            'resolution':self.time_resolution})


    def create_individual(self):
        """
        Creates random parameter values within given bounds.
        Uses an RNG seeded with the main seed of the SAM module.
        """ 
        param_spec = self.parameter_spec() # Sort for replicability
        individual = {k: np.float64(self.rs.uniform(v[0], v[1])) for k, v in param_spec.items()}
        return individual


    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping.
        """
        param_spec = self.parameter_spec()
        individual = {k: np.float64(np.clip(v, a_min=param_spec[k][0], a_max=param_spec[k][1])) for k, v in individual.items()}
        return individual


    def initialise_experiment(self):
        """
        Sets experimental parameters.
        """
        # Use the distribution from Peceveski et al., experiment 2.
        # Define the joint probability equation in order to use helpers to compute
        # the full probability array.
        joint_equation = "p(y1,y2,y3,y4) = p(y1)*p(y2)*p(y3|y1,y2)*p(y4|y2)"

        # Define distributions.
        p1 = {(1,):0.5, (2,):0.5}
        p2 = {(1,):0.5, (2,):0.5}
        p3 = {
            (1,1,1):0.87,
            (1,1,2):0.13,
            (1,2,1):0.13,
            (1,2,2):0.87,
            (2,1,1):0.13,
            (2,1,2):0.87,
            (2,2,1):0.87,
            (2,2,2):0.13
        }
        p4 = {
            (1,1):0.87,
            (1,2):0.13,
            (2,1):0.13,
            (2,2):0.87
        }

        # Compute the joint distribution from all the components.
        self.distribution = helpers.compute_joint_distribution(
            joint_equation, 
            2, 
            p1, p2, p3, p4)

        # Define the Markov blanket of each RV.
        self.dependencies = {
            'y1':['y2', 'y3'],
            'y2':['y1', 'y3', 'y4'],
            'y3':['y1', 'y2'],
            'y4':['y2']
        }

        # Define special parameters for some modules.
        self.special_params = {
            'y1':{'max_weight':4.0},
            'y2':{'max_weight':4.0},
            'y3':{'max_weight':4.0},
            'y4':{'max_weight':2.0},
        }


    def intitialise_distributions(self):
        """
        Creates a set of distributions, one for each trial. Each distribution
        has the same decomposition, but uses randomly generated parameters.
        """
        self.distributions = []
        
        # Joint equation is fixed.
        joint_equation = "p(y1,y2,y3,y4) = p(y1)*p(y2)*p(y3|y1,y2)*p(y4|y2)"
        
        # First two distributions are fixed.
        p1 = {(1,):0.5, (2,):0.5}
        p2 = {(1,):0.5, (2,):0.5}

        for i in range(self.num_fitness_trials):
            p3 = helpers.generate_distribution(num_vars=3, num_discrete_values=2, randomiser=self.rs)
            p3 = helpers.compute_conditional_distribution(joint=p3, num_discrete_values=2)
            p4 = helpers.generate_distribution(num_vars=2, num_discrete_values=2, randomiser=self.rs)
            p4 = helpers.compute_conditional_distribution(joint=p4, num_discrete_values=2)

            # Compute the joint distribution given the individual distributions.
            self.distributions.append(helpers.compute_joint_distribution(joint_equation, 
                2,
                p1, p2, p3, p4))

        # Define the Markov blanket of each RV.
        self.dependencies = {
            'y1':['y2', 'y3'],
            'y2':['y1', 'y3', 'y4'],
            'y3':['y1', 'y2'],
            'y4':['y2']
        }

        # Define special parameters for some modules.
        self.special_params = {
            'y1':{'max_weight':4.0},
            'y2':{'max_weight':4.0},
            'y3':{'max_weight':4.0},
            'y4':{'max_weight':2.0},
        }


    def parameter_spec(self):
        """
        Returns the minima-maxima of each explorable variable.
        Note: Dictionary is an OrderedDict with items sorted by key, to 
        ensure that items are interpreted in the same way everywhere.
        """
        return OrderedDict(sorted(SAMGraph.parameter_spec(len(self.dependencies)).items()))


    def prepare_network(self, distribution, dependencies, num_discrete_vals, num_modes, special_params={}):
        """
        Generates a recurrent network with the specified distribution and 
        dependency parameters, but uses the hyperparameters from the individual dictionary.
        """
        nest.ResetKernel()
        self.set_kernel_defaults()

        self.graph = SAMGraph(randomise_seed=True)

        # Convert the trajectory individual to a dictionary.
        params = {k:self.individual[k] for k in SAMGraph.parameter_spec(len(dependencies)).keys()}

        # Peg the delay to the time resolution.
        params['delay'] = self.time_resolution
        params['weight_chi_alpha_mean'] = 4.0 / 3

        # Create a SAM module with the correct parameters.
        self.graph.create_network(
            num_discrete_vals=num_discrete_vals, 
            num_modes=num_modes, 
            dependencies=dependencies, 
            distribution=distribution, 
            params=params,
            special_params=special_params)

        logging.info("Creating a recurrent SAM graph network with overridden parameters:\n%s", self.graph.parameter_string())
        logging.info("Using distribution:\n%s", helpers.get_ordered_dictionary_string(distribution))


    def simulate(self, traj, run_intermediates=False, save_plot=False):
        """
        Simulates a recurrently connected group of SAM modules, training on a target 
        distribution; i.e. performing density estimation as in Pecevski et al. 2016,
        experiment 2. The loss function is the the KL divergence between target and 
        estimated distributions. 
        If save_plot == True, this will create a directory for each individual that 
        contains a text file with individual params and plots for each trial.
        """
        # Prepare paths for each individual evaluation.
        individual_directory = os.path.join(self.save_directory, str(self.run_number) + "_" + helpers.get_now_string())
        text_path = os.path.join(individual_directory, 'params.txt')
        
        # Declare fitness metrics.
        kld_joint_experimental = []
        kld_joint_experimental_valid = []

        # Run a number of trials to calculate mean fitness of this individual
        for trial in range(self.num_fitness_trials):
            nest.ResetKernel()
            self.set_kernel_defaults()

            self.individual = traj.individual
            self.prepare_network(
                distribution=self.distributions[trial], 
                dependencies=self.dependencies, 
                num_discrete_vals=2, 
                num_modes=2,
                special_params=self.special_params)
            
            # Create directory and params file if requested.
            if save_plot:
                helpers.create_directory(individual_directory)
                helpers.save_text(self.graph.parameter_string(), text_path)

            # Get the network's target distribution.
            distribution = self.graph.distribution

            # Train for the learning period set in the parameters.
            t = 0
            i = 0
            set_second_rate = False
            last_set_intrinsic_rate = self.graph.params['first_bias_rate']
            skip_kld = 1000
            skip_kld_module = 10
            kls_joints = [[] for j in range(len(self.dependencies))]
            set_second_rate = False
            graph_clones = []
            last_klds = []

            while t <= self.graph.params['learning_time']:
                # Inject a current for some time.
                self.graph.present_random_sample() 
                self.graph.clear_currents()
                t += self.graph.params['sample_presentation_time']

                # Compute theoretical distributions and measure KLD.
                if save_plot and i % skip_kld_module == 0:
                    implicits = [s.compute_implicit_distribution() for s in self.graph.sams.values()]
                    distributions = [s.distribution for s in self.graph.sams.values()]
                    for j, (implicit, dist) in enumerate(zip(implicits, distributions)):
                        kls_joints[j].append(helpers.get_KL_divergence(implicit, dist))

                # Measure experimental joint distribution from spike activity.
                if save_plot and run_intermediates and i % skip_kld == 0:
                    # Clone network for later tests.
                    graph_clone = self.graph.clone()

                    # Stop plasticity on clone for testing.
                    graph_clone.set_intrinsic_rate(0.0)
                    graph_clone.set_plasticity_learning_time(0)

                    graph_clones.append(graph_clone)
                                    
                # Set different intrinsic rate.
                if t >= self.graph.params['learning_time'] * self.graph.params['intrinsic_step_time_fraction'] and set_second_rate == False:
                    set_second_rate = True
                    last_set_intrinsic_rate = self.graph.params['second_bias_rate']
                    self.graph.set_intrinsic_rate(last_set_intrinsic_rate)
            
                i += 1

            self.graph.set_intrinsic_rate(0.0)
            self.graph.set_plasticity_learning_time(0)

            # Measure experimental joint distribution on para-experiment clones.
            if save_plot:
                plot_exp_joints = [g.measure_experimental_joint_distribution(duration=20000.0, invalid_handling='first') for g in graph_clones]
                plot_joint_klds = [helpers.get_KL_divergence(p, distribution) for p in plot_exp_joints] 
                plot_joint_klds_valid = [helpers.get_KL_divergence(p, distribution, exclude_invalid_states=True) for p in plot_exp_joints] 

            # Plot modules' KL divergence plots and experimental KL divergence of joint distribution.
            if save_plot:
                fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 20))
                for kld, ym in zip(kls_joints, self.graph.sams):
                    ax[0].plot(np.array(range(len(kld))) * skip_kld_module * self.graph.params['sample_presentation_time'] * 1e-3, kld, label="Analytical KLD {}".format(ym))
                    ax[0].set_title('KL divergence between target and estimated marginal (module) distributions')
                ax[0].legend(loc='upper center')
                if run_intermediates:
                    ax[1].plot(np.array(range(len(plot_joint_klds))) * skip_kld * self.graph.params['sample_presentation_time'] * 1e-3, plot_joint_klds, label="Experimental KLD")
                    ax[1].plot(np.array(range(len(plot_joint_klds_valid))) * skip_kld * self.graph.params['sample_presentation_time'] * 1e-3, plot_joint_klds_valid, label="Experimental KLD (valid only)")
                    ax[1].legend(loc='upper center')
                    ax[1].set_title('KL Divergence between target and estimated joint distribution')

            # Measure experimental KL divergence of entire network by averaging on a few runs.
            last_clone = self.graph.clone()
            experimental_joint = self.graph.measure_experimental_joint_distribution(duration=20000.0, invalid_handling='first')
            this_kld = helpers.get_KL_divergence(experimental_joint, distribution)
            kld_joint_experimental.append(this_kld)
            kld_joint_experimental_valid.append(helpers.get_KL_divergence(experimental_joint, distribution, exclude_invalid_states=True))

            # Draw spiking of output neurons.
            if save_plot:
                last_clone.draw_stationary_state(duration=500, ax=ax[2])
                fig.savefig(os.path.join(individual_directory, str(trial) + '.png'))
                plt.close()

            # Draw histogram of states.
            if save_plot:
                fig = helpers.plot_histogram(distribution, experimental_joint, self.graph.num_discrete_vals, "p*(y)", "p(y;θ)", renormalise_estimated_states=True)
                fig.savefig(os.path.join(individual_directory, str(trial) + '_histogram.png'))
                plt.close()

            logging.info("This run's experimental joint KLD is {}".format(this_kld))

            # Pre-emptively end the fitness trials if the fitness is too bad.
            #if this_kld >= 0.7: break

        self.run_number += 1

        if save_plot:
            last_klds = [kls_joints[i][-1] for i in range(len(kls_joints))]
        mean_loss = np.sum(kld_joint_experimental) / len(kld_joint_experimental)
        mean_loss_valid = np.sum(kld_joint_experimental_valid) / len(kld_joint_experimental_valid)

        logging.info("[Loss] Experimental network joint KLD is {}".format(mean_loss))
        logging.info("Experimental network joint KLD (on valid states only) is {}".format(mean_loss_valid))
        if save_plot:
            logging.info("Mean analytical module joint KLD is {}".format(np.sum(last_klds) / len(last_klds)))

        return (mean_loss, )


    def end(self):
        logger.info("End of experiment. Cleaning up...")


class SPINetworkOptimizee(Optimizee):
    """
    Provides the interface between the LTL API and the SPINetwork class. See SPINetwork and
    SAMGraph for details on the SPI neural network (which is analogous to a SAMGraph in terms
    of intended operation).
    """

    def __init__(
            self, 
            traj, 
            time_resolution=0.1, 
            num_fitness_trials=3, 
            seed=0, 
            n_NEST_threads=1, 
            plots_directory='./sam_plots'):
        super(SPINetworkOptimizee, self).__init__(traj)
        
        self.rs = np.random.RandomState(seed=seed)
        self.num_fitness_trials = num_fitness_trials
        self.run_number = 0
        self.save_directory = plots_directory
        self.time_resolution = time_resolution
        self.num_threads = n_NEST_threads
        self.set_kernel_defaults()

        # Set up exerimental parameters.
        #self.initialise_experiment()
        self.intitialise_distributions()

        # create_individual can be called because __init__ is complete except for traj initialization
        self.individual = self.create_individual()
        for key, val in self.individual.items():
            traj.individual.f_add_parameter(key, val)


    def set_kernel_defaults(self):
        """
        Sets main NEST parameters.
        Note: this needs to be called every time the kernel is reset.
        """
        nest.SetKernelStatus({
            'local_num_threads':self.num_threads,
            'resolution':self.time_resolution})


    def create_individual(self):
        """
        Creates random parameter values within given bounds.
        Uses an RNG seeded with the main seed of the SAM module.
        """ 
        param_spec = self.parameter_spec() # Sort for replicability
        individual = {k: np.float64(self.rs.uniform(v[0], v[1])) for k, v in param_spec.items()}
        return individual


    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping.
        """
        param_spec = self.parameter_spec()
        individual = {k: np.float64(np.clip(v, a_min=param_spec[k][0], a_max=param_spec[k][1])) for k, v in individual.items()}
        return individual


    def initialise_experiment(self):
        """
        Sets experimental parameters.
        """
        # Use the distribution from Peceveski et al., experiment 2.
        # Define the joint probability equation in order to use helpers to compute
        # the full probability array.
        joint_equation = "p(y1,y2,y3,y4) = p(y1)*p(y2)*p(y3|y1,y2)*p(y4|y2)"

        # Define distributions.
        p1 = {(1,):0.5, (2,):0.5}
        p2 = {(1,):0.5, (2,):0.5}
        p3 = {
            (1,1,1):0.87,
            (1,1,2):0.13,
            (1,2,1):0.13,
            (1,2,2):0.87,
            (2,1,1):0.13,
            (2,1,2):0.87,
            (2,2,1):0.87,
            (2,2,2):0.13
        }
        p4 = {
            (1,1):0.87,
            (1,2):0.13,
            (2,1):0.13,
            (2,2):0.87
        }

        # Compute the joint distribution from all the components.
        self.distribution = helpers.compute_joint_distribution(
            joint_equation, 
            2, 
            p1, p2, p3, p4)

        # Define the Markov blanket of each RV.
        self.dependencies = {
            'y1':['y2', 'y3'],
            'y2':['y1', 'y3', 'y4'],
            'y3':['y1', 'y2'],
            'y4':['y2']
        }


    def intitialise_distributions(self):
        """
        Creates a set of distributions, one for each trial. Each distribution
        has the same decomposition, but uses randomly generated parameters.
        """
        self.distributions = []
        
        # Joint equation is fixed.
        joint_equation = "p(y1,y2,y3,y4) = p(y1)*p(y2)*p(y3|y1,y2)*p(y4|y2)"
        
        # First two distributions are fixed.
        p1 = {(1,):0.5, (2,):0.5}
        p2 = {(1,):0.5, (2,):0.5}

        for i in range(self.num_fitness_trials):
            p3 = helpers.generate_distribution(num_vars=3, num_discrete_values=2, randomiser=self.rs)
            p3 = helpers.compute_conditional_distribution(joint=p3, num_discrete_values=2)
            p4 = helpers.generate_distribution(num_vars=2, num_discrete_values=2, randomiser=self.rs)
            p4 = helpers.compute_conditional_distribution(joint=p4, num_discrete_values=2)

            # Compute the joint distribution given the individual distributions.
            self.distributions.append(helpers.compute_joint_distribution(joint_equation, 
                2,
                p1, p2, p3, p4))

        # Define the Markov blanket of each RV.
        self.dependencies = {
            'y1':['y2', 'y3'],
            'y2':['y1', 'y3', 'y4'],
            'y3':['y1', 'y2'],
            'y4':['y2']
        }


    def parameter_spec(self):
        """
        Returns the minima-maxima of each explorable variable.
        Note: Dictionary is an OrderedDict with items sorted by key, to 
        ensure that items are interpreted in the same way everywhere.
        """
        return OrderedDict(sorted(SPINetwork.parameter_spec(len(self.dependencies)).items()))


    def prepare_network(self, distribution, dependencies, num_discrete_vals, special_params={}):
        """
        Generates a recurrent network with the specified distribution and 
        dependency parameters, but uses the hyperparameters from the individual dictionary.
        """
        nest.ResetKernel()
        self.set_kernel_defaults()

        self.network = SPINetwork(randomise_seed=True)

        # Convert the trajectory individual to a dictionary.
        params = {k:self.individual[k] for k in SPINetwork.parameter_spec(len(dependencies)).keys()}

        # Create a SPI network with the correct parameters.
        self.network.create_network(
            num_discrete_vals=num_discrete_vals, 
            dependencies=dependencies, 
            distribution=distribution, 
            override_params=params,
            special_params=special_params)

        logging.info("Creating a recurrent SPI network with overridden parameters:\n%s", self.network.parameter_string())
        logging.info("Using distribution:\n%s", helpers.get_ordered_dictionary_string(distribution))


    def simulate(self, traj, run_intermediates=False, save_plot=False):
        """
        Simulates a recurrently connected network of neuron pools, training on a target 
        distribution; i.e. performing density estimation as in Pecevski et al. 2016,
        experiment 2. The loss function is the the KL divergence between target and 
        estimated distributions. 
        If save_plot == True, this will create a directory for each individual that 
        contains a text file with individual params and plots for each trial.
        """
        # Prepare paths for each individual evaluation.
        individual_directory = os.path.join(self.save_directory, str(self.run_number) + "_" + helpers.get_now_string())
        text_path = os.path.join(individual_directory, 'params.txt')
        
        # Declare fitness metrics.
        kld_joint_experimental = []
        kld_joint_experimental_valid = []

        # Run a number of trials to calculate mean fitness of this individual
        for trial in range(self.num_fitness_trials):
            nest.ResetKernel()
            self.set_kernel_defaults()

            self.individual = traj.individual
            self.prepare_network(
                distribution=self.distributions[trial], 
                dependencies=self.dependencies, 
                num_discrete_vals=2)
            
            # Create directory and params file if requested.
            if save_plot:
                helpers.create_directory(individual_directory)
                helpers.save_text(self.network.parameter_string(), text_path)

            # Get the network's target distribution.
            distribution = self.network.distribution

            # Train for the learning period set in the parameters.
            t = 0
            i = 0
            set_second_rate = False
            last_set_intrinsic_rate = self.network.params['bias_rate_1']
            skip_kld = 1000
            set_second_rate = False
            clones = []
            last_klds = []
            debug = False

            if debug:
                # Print connections between first and second chi pools.
                conn = nest.GetConnections(self.network.chi_pools['y2'], self.network.chi_pools['y4'])
                print("some connections:\n", conn)

                # nest.SetStatus([conn[10]], {'debug':True})

                # Attach a spike reader to all population coding layers.
                spikereader = nest.Create('spike_detector', params={'withtime':True, 'withgid':True})
                nest.Connect(self.network.all_neurons, spikereader, syn_spec={'delay':self.network.params['delay_devices']})

            while t <= self.network.params['learning_time']:
                if i % 1000 == 0: logging.info("Time: {}".format(t))

                # Inject a current for some time.
                self.network.present_random_sample() 
                self.network.clear_currents()
                t += self.network.params['sample_presentation_time']

                # Draw debug spikes.
                if debug:
                    helpers.plot_spikes(spikereader)
                    spikes = nest.GetStatus(spikereader, keys='events')[0]
                    exp_joint = self.network.get_distribution_from_spikes(spikes, t - self.network.params['sample_presentation_time'], t)
                    print(exp_joint)

                # Measure experimental joint distribution from spike activity.
                if save_plot and run_intermediates and i % skip_kld == 0:
                    # Clone network for later tests.
                    clone = self.network.clone()

                    # Stop plasticity on clone for testing.
                    clone.set_intrinsic_rate(0.0)
                    clone.set_plasticity_learning_time(0)

                    clones.append(clone)
                                    
                # Set different intrinsic rate.
                if t >= self.network.params['learning_time'] * self.network.params['bias_change_time_fraction'] and set_second_rate == False:
                    set_second_rate = True
                    last_set_intrinsic_rate = self.network.params['bias_rate_2']
                    self.network.set_intrinsic_rate(last_set_intrinsic_rate)
            
                i += 1

            self.network.set_intrinsic_rate(0.0)
            self.network.set_plasticity_learning_time(0)

            # Measure experimental joint distribution on para-experiment clones.
            if save_plot:
                plot_exp_joints = [g.measure_experimental_joint_distribution(duration=20000.0) for g in clones]
                plot_joint_klds = [helpers.get_KL_divergence(p, distribution) for p in plot_exp_joints] 
                plot_joint_klds_valid = [helpers.get_KL_divergence(p, distribution, exclude_invalid_states=True) for p in plot_exp_joints] 

            # Plot experimental KL divergence of joint distribution.
            if save_plot:
                fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 20))
                if run_intermediates:
                    ax[0].plot(np.array(range(len(plot_joint_klds))) * skip_kld * self.network.params['sample_presentation_time'] * 1e-3, plot_joint_klds, label="Experimental KLD")
                    ax[0].plot(np.array(range(len(plot_joint_klds_valid))) * skip_kld * self.network.params['sample_presentation_time'] * 1e-3, plot_joint_klds_valid, label="Experimental KLD (valid only)")
                    ax[0].legend(loc='upper center')
                    ax[0].set_title('KL Divergence between target and estimated joint distribution')

            # Measure experimental KL divergence of entire network by averaging on a few runs.
            experimental_joint = self.network.measure_experimental_joint_distribution(duration=20000.0)
            this_kld = helpers.get_KL_divergence(experimental_joint, distribution)
            this_kld_valid = helpers.get_KL_divergence(experimental_joint, distribution, exclude_invalid_states=True)
            kld_joint_experimental.append(this_kld)
            kld_joint_experimental_valid.append(this_kld_valid)

            # Draw spiking of output neurons.
            if save_plot:
                last_clone = self.network.clone()
                last_clone.draw_stationary_state(duration=500, ax=ax[1])
                fig.savefig(os.path.join(individual_directory, str(trial) + '.png'))
                plt.close()

            # Draw histogram of states.
            if save_plot:
                fig = helpers.plot_histogram(distribution, experimental_joint, self.network.num_discrete_vals, "p*(y)", "p(y;θ)", renormalise_estimated_states=True)
                fig.savefig(os.path.join(individual_directory, str(trial) + '_histogram.png'))
                plt.close()

            logging.info("This run's experimental joint KLD is {}".format(this_kld))
            logging.info("This run's experimental joint KLD (valid only) is {}".format(this_kld_valid))

            # Pre-emptively end the fitness trials if the fitness is too bad.
            #if this_kld >= 1.5: break

        self.run_number += 1

        mean_loss = np.sum(kld_joint_experimental) / len(kld_joint_experimental)
        mean_loss_valid = np.sum(kld_joint_experimental_valid) / len(kld_joint_experimental_valid)

        logging.info("Final mean experimental network joint KLD is {}".format(mean_loss))
        logging.info("Experimental network joint KLD (on valid states only) is {}".format(mean_loss_valid))

        return (mean_loss, )


    def end(self):
        logger.info("End of experiment. Cleaning up...")


class SPIConditionalNetworkOptimizee(Optimizee):
    """
    Provides the interface between the LTL API and the SPINetwork class. See SPINetwork and
    SAMModule for details on the SPI neural network (which is analogous to a SAMModule in terms
    of intended operation).
    """

    def __init__(
            self, 
            traj, 
            time_resolution=0.1, 
            num_fitness_trials=3, 
            seed=0, 
            n_NEST_threads=1, 
            plots_directory='./sam_plots'):
        super(SPIConditionalNetworkOptimizee, self).__init__(traj)
        
        self.rs = np.random.RandomState(seed=seed)
        self.num_fitness_trials = num_fitness_trials
        self.run_number = 0
        self.save_directory = plots_directory
        self.time_resolution = time_resolution
        self.num_threads = n_NEST_threads
        self.set_kernel_defaults()

        # Set up exerimental parameters.
        #self.initialise_experiment()
        self.intitialise_distributions()

        # create_individual can be called because __init__ is complete except for traj initialization
        self.individual = self.create_individual()
        for key, val in self.individual.items():
            traj.individual.f_add_parameter(key, val)


    def set_kernel_defaults(self):
        """
        Sets main NEST parameters.
        Note: this needs to be called every time the kernel is reset.
        """
        nest.SetKernelStatus({
            'local_num_threads':self.num_threads,
            'resolution':self.time_resolution})


    def create_individual(self):
        """
        Creates random parameter values within given bounds.
        Uses an RNG seeded with the main seed of the SAM module.
        """ 
        param_spec = self.parameter_spec() # Sort for replicability
        individual = {k: np.float64(self.rs.uniform(v[0], v[1])) for k, v in param_spec.items()}
        return individual


    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping.
        """
        param_spec = self.parameter_spec()
        individual = {k: np.float64(np.clip(v, a_min=param_spec[k][0], a_max=param_spec[k][1])) for k, v in individual.items()}
        return individual


    def initialise_experiment(self):
        """
        Sets experimental parameters.
        """
        # Use the distribution from Peceveski et al., experiment 1.
        # Define distribution.
        self.distribution = {
            (1,1,1):0.04,
            (1,1,2):0.04,
            (1,2,1):0.21,
            (1,2,2):0.21,
            (2,1,1):0.04,
            (2,1,2):0.21,
            (2,2,1):0.21,
            (2,2,2):0.04
        }

        # Define the Markov blanket of each RV.
        self.dependencies = {
            'y3':['y1', 'y2']
        }


    def intitialise_distributions(self):
        """
        Creates a set of distributions, one for each trial. Each distribution
        has the same decomposition, but uses randomly generated parameters.
        """
        self.distributions = []
        
        for i in range(self.num_fitness_trials):
            p = helpers.generate_distribution(num_vars=3, num_discrete_values=2, randomiser=self.rs)
            self.distributions.append(p)
            
        # Define the Markov blanket of each RV.
        self.dependencies = {
            'y3':['y1', 'y2']
        }


    def parameter_spec(self):
        """
        Returns the minima-maxima of each explorable variable.
        Note: Dictionary is an OrderedDict with items sorted by key, to 
        ensure that items are interpreted in the same way everywhere.
        """
        return OrderedDict(sorted(SPINetwork.parameter_spec(len(self.dependencies), 'conditional').items()))


    def prepare_network(self, distribution, dependencies, num_discrete_vals, special_params={}):
        """
        Generates a recurrent network with the specified distribution and 
        dependency parameters, but uses the hyperparameters from the individual dictionary.
        """
        nest.ResetKernel()
        self.set_kernel_defaults()

        self.network = SPINetwork(randomise_seed=True)

        # Convert the trajectory individual to a dictionary.
        params = {k:self.individual[k] for k in SPINetwork.parameter_spec(len(dependencies), 'conditional').keys()}

        # Create a SPI network with the correct parameters.
        self.network.create_conditional_network(
            num_discrete_vals=num_discrete_vals, 
            dependencies=dependencies, 
            distribution=distribution, 
            override_params=params)

        logging.info("Creating a recurrent SPI network with overridden parameters:\n%s", self.network.parameter_string())
        logging.info("Using distribution:\n%s", helpers.get_ordered_dictionary_string(distribution))


    def simulate(self, traj, run_intermediates=False, save_plot=False):
        """
        Simulates a recurrently connected network of neuron pools, training on a target 
        distribution; i.e. performing density estimation as in Pecevski et al. 2016,
        experiment 2. The loss function is the the KL divergence between target and 
        estimated distributions. 
        If save_plot == True, this will create a directory for each individual that 
        contains a text file with individual params and plots for each trial.
        """
        # Prepare paths for each individual evaluation.
        individual_directory = os.path.join(self.save_directory, str(self.run_number) + "_" + helpers.get_now_string())
        text_path = os.path.join(individual_directory, 'params.txt')
        
        # Declare fitness metrics.
        kld_cond_experimental = []

        # Run a number of trials to calculate mean fitness of this individual
        for trial in range(self.num_fitness_trials):
            nest.ResetKernel()
            self.set_kernel_defaults()

            self.individual = traj.individual
            self.prepare_network(
                distribution=self.distributions[trial], 
                dependencies=self.dependencies, 
                num_discrete_vals=2)
            
            # Create directory and params file if requested.
            if save_plot:
                helpers.create_directory(individual_directory)
                helpers.save_text(self.network.parameter_string(), text_path)

            # Get the network's target distribution.
            distribution = self.network.distribution
            conditional = helpers.compute_conditional_distribution(distribution, self.network.num_discrete_vals)

            # Train for the learning period set in the parameters.
            t = 0
            i = 0
            set_second_rate = False
            last_set_intrinsic_rate = self.network.params['bias_rate_1']
            skip_kld = 1000
            set_second_rate = False
            clones = []
            last_klds = []
            debug = False

            while t <= self.network.params['learning_time']:
                if i % 1000 == 0: logging.info("Time: {}".format(t))

                # Inject a current for some time.
                self.network.present_random_sample() 
                self.network.clear_currents()
                t += self.network.params['sample_presentation_time']

                # Measure experimental joint distribution from spike activity.
                if save_plot and run_intermediates and i % skip_kld == 0:
                    # Clone network for later tests.
                    clone = self.network.clone()

                    # Stop plasticity on clone for testing.
                    clone.set_intrinsic_rate(0.0)
                    clone.set_plasticity_learning_time(0)

                    clones.append(clone)
                                    
                # Set different intrinsic rate.
                if t >= self.network.params['learning_time'] * self.network.params['bias_change_time_fraction'] and set_second_rate == False:
                    set_second_rate = True
                    last_set_intrinsic_rate = self.network.params['bias_rate_2']
                    self.network.set_intrinsic_rate(last_set_intrinsic_rate)
            
                i += 1

            self.network.set_intrinsic_rate(0.0)
            self.network.set_plasticity_learning_time(0)

            # Measure experimental joint distribution on para-experiment clones.
            if save_plot:
                plot_exp_conds = [g.measure_experimental_cond_distribution(duration=5000.0) for g in clones]
                plot_cond_klds = [helpers.get_KL_divergence(p, conditional) for p in plot_exp_conds] 

            # Plot experimental KL divergence of joint distribution.
            if save_plot:
                if run_intermediates:
                    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 20))
                    ax[0].plot(np.array(range(len(plot_cond_klds))) * skip_kld * self.network.params['sample_presentation_time'] * 1e-3, plot_cond_klds, label="Experimental KLD")
                    ax[0].legend(loc='upper center')
                    ax[0].set_title('KL Divergence between target and estimated joint distribution')

            # Measure experimental KL divergence of entire network by averaging on a few runs.
            experimental_cond = self.network.measure_experimental_cond_distribution(duration=5000.0)
            this_kld = helpers.get_KL_divergence(experimental_cond, conditional)
            kld_cond_experimental.append(this_kld)

            # Draw histogram of states.
            if save_plot:
                # Attach a spike reader to all population coding layers.
                spikereader = nest.Create('spike_detector', params={'withtime':True, 'withgid':True})
                nest.Connect(self.network.all_neurons, spikereader, syn_spec={'delay':self.network.params['delay_devices']})
                self.network.present_input_evidence(sample=(1, 2), duration=1000.0)
                helpers.plot_spikes(spikereader, ax[1], title="Activity plot for input = (1, 2)")
                fig.savefig(os.path.join(individual_directory, str(trial) + '.png'))
                plt.close()

                fig = helpers.plot_histogram(conditional, experimental_cond, self.network.num_discrete_vals, "p*(z|x)", "p(z|x;θ)", renormalise_estimated_states=False)
                fig.savefig(os.path.join(individual_directory, str(trial) + '_histogram.png'))
                plt.close()

            logging.info("This run's experimental conditional KLD is {}".format(this_kld))

            # Pre-emptively end the fitness trials if the fitness is too bad.
            if this_kld >= 0.5: break

        self.run_number += 1

        mean_loss = np.sum(kld_cond_experimental) / len(kld_cond_experimental)

        logging.info("Final mean experimental conditional KLD is {}".format(mean_loss))

        return (mean_loss, )


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
    optimizee = SPINetworkOptimizee(fake_traj, n_NEST_threads=1, plots_directory='./spi_test_run')

    fake_traj.individual = sdict(optimizee.create_individual())

    with timed(logger):
        loss = optimizee.simulate(fake_traj, show_plot=True)
    logging.info("Final loss is {}".format(loss))


if __name__ == "__main__":
    main()
