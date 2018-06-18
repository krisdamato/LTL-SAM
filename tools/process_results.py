import argparse
import ast
import glob
import os
import shutil

from collections import OrderedDict


def copy_log_files_to(dest):
    '''Looks for text files in this directory recursively and spits them to the destination folder.'''

    # Find all files that contain "LOG" in them
    directory = "D:\\LTL results\\New"
    filenames = glob.glob(directory + '/**/*_LOG.txt', recursive=True)
    filenames = sorted(filenames)

    for f in filenames:
        shutil.copy(f, dest)


def process_sam_results(log_dir="D:\\LTL results\\New"):
    # Set SAM HP order
    hps = ['bias_baseline', 'weight_baseline', 'T', 'relative_bias_spike_rate', 'first_bias_rate', 'second_bias_rate', 'initial_stdp_rate', 'final_stdp_rate', 'exp_term_prob', 'exp_term_prob_scale']
    hps_latex = ['$b_-$', '$w_-$', '$T$', '$R$', "$\\eta'_0$", "$\\eta'_1$", "$\\eta_0$", '$\\eta_1$', '$c_1$', '$c_2$']

    # Find all files that contain "SAM" and "LOG" in them
    directory = log_dir
    filenames = glob.glob(directory + '/**/*SAM-*_LOG.txt', recursive=True)
    filenames = sorted(filenames)

    best_dicts = process_files(filenames, hps, hps_latex)

    return filenames, best_dicts


def process_samgraph_results(log_dir="D:\\LTL results\\New"):
    # Set SAMGRAPH HP order
    hps = ['bias_baseline_1', 'bias_baseline_2', 'bias_baseline_3', 'bias_baseline_4', 'weight_baseline', 'T', 'relative_bias_spike_rate', 'first_bias_rate', 'initial_stdp_rate', 'final_stdp_rate', 'exp_term_prob', 'exp_term_prob_scale']
    hps_latex = ['$b^1_-$', '$b^2_-$', '$b^3_-$', '$b^4_-$', '$w_-$', '$T$', '$R$', "$\\eta'_0$", '$\\eta_0$', '$\\eta_1$', '$c_1$', '$c_2$']

    # Find all files that contain "LOG" in them
    directory = log_dir
    filenames = glob.glob(directory + '/**/*SAMGRAPH-*_LOG.txt', recursive=True)
    filenames = sorted(filenames)

    best_dicts = process_files(filenames, hps, hps_latex)

    return filenames, best_dicts


def process_files(filenames, hps, hps_latex):
    # Create dictionary of hp:value dictionaries
    best_dicts = {}
    
    for filename in filenames:
        with open(filename) as f:
            best_n = 100000000
            for n, line in enumerate(f):
                search_string = "generation 19" if "GA" in filename else "generation 79"
                if search_string in line:
                    best_n = n + 2 if "GA" in filename else n + 5
                if n == best_n:
                    dict_str = line[line.find("{"):line.find("}") + 1]
                    best_dict = ast.literal_eval(dict_str)
                    best_dict = OrderedDict((k, best_dict[k]) for k in hps if k in best_dict)
                    best_dicts[filename] = best_dict

    # Remove files that do not have all the variables
    remove_list = []
    for fn in best_dicts.keys():
        not_found = len([n for n in hps if n not in best_dicts[fn]]) > 0
        if not_found:
            remove_list.append(fn)

    for i in remove_list:
        best_dicts.pop(i)

    # Print table of HPs
    relevant_fns = [fn for fn in filenames if fn in best_dicts]
    for fn in relevant_fns:
        last_slash_pos = fn.rfind('\\')
        last_dot_post = fn.rfind('.')
        print("{} & ".format(fn[last_slash_pos + 1:last_dot_post]), end='')
    print('') # next line
    for var, var_latex in zip(hps, hps_latex):
        print(var_latex, "& ", end='')
    
        vars = [round(best_dicts[n][var], 4) for n in relevant_fns]
        vars_unrounded = [best_dicts[n][var] for n in relevant_fns]
        var_strings = ["{:.4f}".format(v) for v in vars]
        print(" & ".join(var_strings), "\\\\")

    return best_dicts


def run_best_sam(resolution, fixed_delay, use_pecevski, num_trials, is_nes=False):
    '''Runs the best SAM setup in the log file chosen by the user.'''

    import logging.config
    
    from sam.optimizee import SAMOptimizee
    from ltl import DummyTrajectory
    from ltl import sdict

    from pypet import Environment, pypetconstants
    from ltl.logging_tools import create_shared_logger_data, configure_loggers
    from ltl.optimizers.evolution import GeneticAlgorithmOptimizer, GeneticAlgorithmParameters
    from ltl.paths import Paths
    from sam.optimizee import SAMOptimizee, SAMGraphOptimizee

    logger = logging.getLogger('bin.ltl-sam-ga')

    name = "trial"
    root_dir_path = "plots"
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

    print("Running with resolution = {}, fixed delay = {}, use_pecevski = {}\n".format(resolution, fixed_delay, use_pecevski))
   
    fns, hps = process_sam_results('/home/krisdamato/LTL-SAM/results/')
    print('')
    for i, fn in enumerate(fns):
        print("{}: {}".format(i, fn))

    try:
        i = int(input('\nChoose log index: '))
    except ValueError:
        print("Not a number!")
        return

    if i >= len(fns):
        print("Choose within the range!")
        return

    # Get best hps in the chosen log file.
    params = hps[fns[i]]
    params['stdp_time_fraction'] = 0.5
    params['intrinsic_step_time_fraction'] = 0.5
    params['learning_time'] = 600000

    # Create the SAM optimizee.
    optimizee = SAMOptimizee(traj, 
                            use_pecevski=use_pecevski, 
                            n_NEST_threads=1, 
                            time_resolution=resolution,
                            fixed_delay=fixed_delay,
                            plots_directory='/home/krisdamato/LTL-SAM/plots/', 
                            forced_params=params,
                            plot_all=True,
                            num_fitness_trials=num_trials)

    # Run simulation with the forced params.
    optimizee.simulate(traj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--copy', action='store_true', help='Just copy log files to LTL-SAM folder')
    parser.add_argument('-rs', '--run_sam', action='store_true', help='Run best SAM in log files, asking for user input to select which log file to use.')
    parser.add_argument('-r', '--resolution', required=False, type=float, help='Resolution')
    parser.add_argument('-fd', '--fixed_delay', required=False, type=float, help='Fixed delay')
    parser.add_argument('-p', '--use_pecevski', action='store_true', help='Use Pecevski distributions')
    parser.add_argument('-in', '--is_nes', action='store_true', help='Assume NES log type')
    parser.add_argument('-nt', '--num_trials', required=False, type=int, help='Number of trials')

    args = parser.parse_args()

    if args.copy: copy_log_files_to("D:\\LTL-SAM\\results\\")
    if args.run_sam: run_best_sam(resolution=args.resolution, fixed_delay=args.fixed_delay, use_pecevski=args.use_pecevski, num_trials=args.num_trials, is_nes=args.is_nes)

    process_samgraph_results()
