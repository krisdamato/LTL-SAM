import logging.config
import os

from pypet import Environment

from ltl.logging_tools import create_shared_logger_data, configure_loggers
from ltl.optimizers.evolution import GeneticAlgorithmOptimizer, GeneticAlgorithmParameters
from ltl.paths import Paths
from ltl.recorder import Recorder
from sam.optimizee import SAMOptimizee

logger = logging.getLogger('bin.ltl-sam-ga')


def main():
    name = 'LTL-SAM-GA'
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

    # print("All output logs can be found in directory ", paths.logs_path)

    traj_file = os.path.join(paths.output_dir_path, 'data.h5')

    # Create an environment that handles running our simulation
    # This initializes a PyPet environment
    env = Environment(trajectory=name, filename=traj_file, file_title='{} data'.format(name),
                      comment='{} data'.format(name),
                      add_time=True,
                      automatic_storing=True,
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
    optimizee = SAMOptimizee(traj, n_NEST_threads=1)

    # NOTE: Outerloop optimizer initialization
    parameters = GeneticAlgorithmParameters(seed=0, popsize=30, CXPB=0.5,
                                            MUTPB=1.0, NGEN=100, indpb=0.01,
                                            tournsize=20, matepar=0.5,
                                            mutpar=1.0, remutate=False
                                            )

    optimizer = GeneticAlgorithmOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                          optimizee_fitness_weights=(-0.1,),
                                          parameters=parameters,
                                          optimizee_bounding_func=optimizee.bounding_func,
                                          optimizee_parameter_spec=optimizee.parameter_spec
                                          )

    # Add post processing
    env.add_postprocessing(optimizer.post_process)

    # Run the simulation with all parameter combinations
    env.run(optimizee.simulate)

    # NOTE: Outerloop optimizer end
    optimizer.end(traj)

    # Finally disable logging and close all log-files
    env.disable_logging()

    # Run the simulation with all parameter combinations
    env.run(optimizee.simulate)

if __name__ == '__main__':
    main()
