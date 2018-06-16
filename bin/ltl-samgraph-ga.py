import logging.config
import numpy as np
import os
import argparse

from pypet import Environment, pypetconstants
from ltl.logging_tools import create_shared_logger_data, configure_loggers
from ltl.optimizers.evolution import GeneticAlgorithmOptimizer, GeneticAlgorithmParameters
from ltl.optimizers.crossentropy.distribution import Gaussian
from ltl.optimizers.face.optimizer import FACEOptimizer, FACEParameters
from ltl.paths import Paths
from sam.optimizee import SAMGraphOptimizee

logger = logging.getLogger('bin.ltl-samgraph-ga')


def main(path_name, 
         resolution, 
         fixed_delay,
         state_handling,
         use_pecevski):
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
                      log_stdout=False,  # Sends stdout to logs
                      multiproc=True,
                      use_scoop=True,
                      freeze_input=False,
                      wrap_mode=pypetconstants.WRAP_MODE_LOCAL
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
    optimizee = SAMGraphOptimizee(traj, 
                                    n_NEST_threads=1, 
                                    time_resolution=resolution, 
                                    fixed_delay=fixed_delay, 
                                    use_pecevski=use_pecevski, 
                                    state_handling=state_handling,
                                    plots_directory=paths.output_dir_path, 
                                    num_fitness_trials=5)

    # NOTE: Outerloop optimizer initialization
    parameters = GeneticAlgorithmParameters(seed=0, popsize=200, CXPB=0.5,
                                            MUTPB=1.0, NGEN=20, indpb=0.05,
                                            tournsize=20, matepar=0.5,
                                            mutpar=1.0, remutate=False
                                            )

    optimizer = GeneticAlgorithmOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                          optimizee_fitness_weights=(-0.1,),
                                          parameters=parameters,
                                          optimizee_bounding_func=optimizee.bounding_func,
                                          optimizee_parameter_spec=optimizee.parameter_spec,
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
    parser.add_argument('-fd', '--fixed_delay', required=True, type=float, help='Fixed delay')
    parser.add_argument('-p', '--use_pecevski', action='store_true', help='Use Pecevski distributions')
    parser.add_argument('-s', '--state_handling', required=True, help='State handling ("none", "first", "random"')
    args = parser.parse_args()

    main(path_name=args.name, resolution=args.resolution, fixed_delay=args.fixed_delay, state_handling=args.state_handling, use_pecevski=args.use_pecevski)
