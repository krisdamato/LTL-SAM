import argparse
import logging.config
import numpy as np
import os

from collections import OrderedDict
from pypet import Environment, pypetconstants
from ltl.logging_tools import create_shared_logger_data, configure_loggers
from ltl.optimizers.naturalevolutionstrategies import NaturalEvolutionStrategiesOptimizer, NaturalEvolutionStrategiesParameters
from ltl.paths import Paths
from sam.optimizee import SPINetworkOptimizee
from sam.spinetwork import SPINetwork

logger = logging.getLogger('bin.ltl-spigraph-nes')

def main(path_name, 
         resolution, 
         min_delay, 
         fixed_delay, 
         max_delay, 
         use_pecevski,
         num_trials):
    name = path_name
    try:
        with open('bin/path.conf') as f:
            root_dir_path = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            "You have not set the root path to store your results."
            " Write the path to a path.conf text file in the bin directory"
            " before running the simulation"
        )
    paths = Paths(name, dict(run_no='test'), root_dir_path=root_dir_path)

    traj_file = os.path.join(paths.output_dir_path, 'data.h5')

    # Create an environment that handles running our simulation
    # This initializes a PyPet environment
    env = Environment(trajectory=name, filename=traj_file, file_title='{} data'.format(name),
                      comment='{} data'.format(name),
                      add_time=True,
                      automatic_storing=True,
                      use_scoop=True,
                      multiproc=True,
                      wrap_mode=pypetconstants.WRAP_MODE_LOCAL,
                      log_stdout=False,  # Sends stdout to logs
                      )

    create_shared_logger_data(logger_names=['bin', 'optimizers'],
                              log_levels=['INFO', 'INFO'],
                              log_to_consoles=[True, True],
                              sim_name=name,
                              log_directory=paths.logs_path)
    configure_loggers()

    # Get the trajectory from the environment
    traj = env.trajectory

    # NOTE: Innerloop simulator
    optimizee = SPINetworkOptimizee(traj, 
                                    n_NEST_threads=1, 
                                    time_resolution=resolution, 
                                    min_delay=min_delay, 
                                    fixed_delay=fixed_delay,
                                    max_delay=max_delay,
                                    use_pecevski=use_pecevski,
                                    plots_directory=paths.output_dir_path, 
                                    num_fitness_trials=num_trials)

    # Get bounds for mu and sigma calculation.
    param_spec = OrderedDict(sorted(SPINetwork.parameter_spec(4).items()))
    names = [k for k, _ in param_spec.items()]
    mu = np.array([(v_min + v_max) / 2 for k, (v_min, v_max) in param_spec.items()])
    sigma = np.array([(v_max - v_min) / 2 for k, (v_min, v_max) in param_spec.items()])

    print("Using means: {}\nUsing stds: {}".format(dict(zip(names, mu)), dict(zip(names, sigma))))

    # NOTE: Outerloop optimizer initialization
    parameters = NaturalEvolutionStrategiesParameters(seed=0, pop_size=96,
                                            n_iteration=40, 
                                            learning_rate_sigma=0.5,
                                            learning_rate_mu=0.5,
                                            mu=mu,
                                            sigma=sigma,
                                            mirrored_sampling_enabled=True,
                                            fitness_shaping_enabled=True,
                                            stop_criterion=np.Inf
                                            )
        
    optimizer = NaturalEvolutionStrategiesOptimizer(traj, 
                                          optimizee_create_individual=optimizee.create_individual,
                                          optimizee_fitness_weights=(-1.0,),
                                          parameters=parameters,
                                          optimizee_bounding_func=optimizee.bounding_func,
                                          fitness_plot_name=path_name
                                          )

    # Add post processing
    env.add_postprocessing(optimizer.post_process)

    # Run the simulation with all parameter combinations
    env.run(optimizee.simulate)

    # NOTE: Outerloop optimizer end
    optimizer.end(traj)

    # Finally disable logging and close all log-files
    env.disable_logging()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True, help='Experiment name')
    parser.add_argument('-r', '--resolution', required=True, type=float, help='Resolution')
    parser.add_argument('-mind', '--min_delay', required=True, type=float, help='Minimum delay')
    parser.add_argument('-fd', '--fixed_delay', required=True, type=float, help='Fixed delay')
    parser.add_argument('-maxd', '--max_delay', required=True, type=float, help='Maximum delay')
    parser.add_argument('-p', '--use_pecevski', action='store_true', help='Use Pecevski distributions')
    parser.add_argument('-nt', '--num_trials', required=True, help='Number of trials')
    args = parser.parse_args()

    main(path_name=args.name, resolution=args.resolution, min_delay=args.min_delay, fixed_delay=args.fixed_delay, max_delay=args.max_delay, use_pecevski=args.use_pecevski, num_trials=args.num_trials)