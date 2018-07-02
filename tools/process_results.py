import argparse
import ast
import glob
import os
import shutil
import re

from collections import OrderedDict


def copy_log_files_to(dest):
    '''Looks for text files in this directory recursively and spits them to the destination folder.'''

    # Find all files that contain "LOG" in them
    directory = "D:\\LTL results\\New"
    filenames = glob.glob(directory + '/**/*_LOG.txt', recursive=True)
    filenames = sorted(filenames)

    for f in filenames:
        shutil.copy(f, dest)


def process_sam_results(log_dir="D:\\LTL results\\New", search_str=''):
    # Set SAM HP order
    hps = ['bias_baseline', 'weight_baseline', 'T', 'relative_bias_spike_rate', 'first_bias_rate', 'second_bias_rate', 'initial_stdp_rate', 'final_stdp_rate', 'exp_term_prob', 'exp_term_prob_scale']
    hps_latex = ['$b_-$', '$w_-$', '$T$', '$R$', "$\\eta'_0$", "$\\eta'_1$", "$\\eta_0$", '$\\eta_1$', '$c_1$', '$c_2$']

    # Find all files that contain "SAM" and "LOG" in them
    directory = log_dir
    filenames = glob.glob(directory + '/**/*SAM-*{}*_LOG.txt'.format(search_str), recursive=True)
    filenames = sorted(filenames)

    best_dicts, _, _ = process_files(filenames, hps, hps_latex)

    return filenames, best_dicts


def process_samgraph_results(log_dir="D:\\LTL results\\New", search_str=''):
    # Set SAMGRAPH HP order
    hps = ['bias_baseline_1', 'bias_baseline_2', 'bias_baseline_3', 'bias_baseline_4', 'weight_baseline', 'T', 'relative_bias_spike_rate', 'first_bias_rate', 'initial_stdp_rate', 'final_stdp_rate', 'exp_term_prob', 'exp_term_prob_scale']
    hps_latex = ['$b^1_-$', '$b^2_-$', '$b^3_-$', '$b^4_-$', '$w_-$', '$T$', '$R$', "$\\eta'_0$", '$\\eta_0$', '$\\eta_1$', '$c_1$', '$c_2$']

    # Find all files that contain "LOG" in them
    directory = log_dir
    filenames = glob.glob(directory + '/**/*SAMGRAPH-*{}*_LOG.txt'.format(search_str), recursive=True)
    filenames = sorted(filenames)

    best_dicts, _, _ = process_files(filenames, hps, hps_latex)

    return filenames, best_dicts


def process_spi_results(log_dir="D:\\LTL results\\New", search_str=''):
    # Set SPI HP order
    hps = ['bias_baseline', 'weight_baseline', 'T', 'bias_relative_spike_rate', 'bias_rate_1', 'stdp_rate_initial', 'stdp_rate_final', 'prob_exp_term', 'prob_exp_term_scale',
           'connectivity_chi_chi', 'connectivity_chi_self', 'connectivity_chi_inh', 'connectivity_inh_self', 'connectivity_inh_chi',
            'weight_chi_chi_max', 'weight_chi_self', 'weight_chi_inhibitors', 'weight_inhibitors_self', 'weight_inhibitors_chi']
    hps_latex = ["$b_-$", 
			"$w_-$", 
			"$T$",
			"$R$",
			"$\\eta'$", 
			"$\\eta_0$",
			"$\\eta_1$", 
			"$c_1$", 
			"$c_2$", 
			"$\\rho_{PP}$",
			"$\\rho_{E}$", 
			"$\\rho_{PI}$",
			"$\\rho_{I}$",
			"$\\rho_{IP}$",
			"$w_{\\textrm{MAX}}$", 
			"$w_{E}$", 
			"$w_{PI}$",
			"$w_{I}$", 
			"$w_{IP}$"]

    # Find all files that contain "SAM" and "LOG" in them
    directory = log_dir
    filenames = glob.glob(directory + '/**/*SPI-*{}*_LOG.txt'.format(search_str), recursive=True)
    filenames = sorted(filenames)

    best_dicts, _, _ = process_files(filenames, hps, hps_latex)

    return filenames, best_dicts


def process_spigraph_results(log_dir="D:\\LTL results\\New", search_str=''):
    # Set SPIGRAPH HP order
    hps = ['bias_baseline_1', 'bias_baseline_2', 'bias_baseline_3', 'bias_baseline_4', 
           'weight_baseline', 'T', 'bias_relative_spike_rate', 'bias_rate_1', 'stdp_rate_initial', 'stdp_rate_final', 'prob_exp_term', 'prob_exp_term_scale',
           'connectivity_chi_chi', 'connectivity_chi_self', 'connectivity_chi_inh', 'connectivity_inh_self', 'connectivity_inh_chi',
            'weight_chi_chi_max_1', 'weight_chi_chi_max_2', 'weight_chi_chi_max_3', 'weight_chi_chi_max_4', 
            'weight_chi_self', 'weight_chi_inhibitors', 'weight_inhibitors_self', 'weight_inhibitors_chi']
    hps_latex = ["$b^1_-$", "$b^2_-$", "$b^3_-$", "$b^4_-$" 
			"$w_-$", 
			"$T$",
			"$R$",
			"$\\eta'$", 
			"$\\eta_0$",
			"$\\eta_1$", 
			"$c_1$", 
			"$c_2$", 
			"$\\rho_{PP}$",
			"$\\rho_{E}$", 
			"$\\rho_{PI}$",
			"$\\rho_{I}$",
			"$\\rho_{IP}$",
			"$w^1_{\\textrm{MAX}}$", 
			"$w^2_{\\textrm{MAX}}$", 
			"$w^3_{\\textrm{MAX}}$", 
			"$w^4_{\\textrm{MAX}}$", 
			"$w_{E}$", 
			"$w_{PI}$",
			"$w_{I}$", 
			"$w_{IP}$"]

    # Find all files that contain "SAM" and "LOG" in them
    directory = log_dir
    filenames = glob.glob(directory + '/**/*SPIGRAPH-*{}*_LOG.txt'.format(search_str), recursive=True)
    filenames = sorted(filenames)

    best_dicts, _, _ = process_files(filenames, hps, hps_latex)

    return filenames, best_dicts


def process_files(filenames, hps, hps_latex):
    # Create dictionary of hp:value dictionaries
    best_dicts = {}
    best_fitnesses = {}
    best_gen_fitnesses = {}
    
    for filename in filenames:
        with open(filename) as f:
            best_n = 100000000
            best_fitness_ind = 10000000
            best_fitness_gen = 10000000
            if "GA" in filename:
                for n, line in enumerate(f):
                    search_string = "generation 19"
                    if search_string in line:
                        best_n = n + 2
                    if n == best_n:
                        dict_str = line[line.find("{"):line.find("}") + 1]
                        best_dict = ast.literal_eval(dict_str)
                        best_dict = OrderedDict((k, best_dict[k]) for k in hps if k in best_dict)
                        best_dicts[filename] = best_dict
                        best_fitness_ind = float(line[line.find("(") + 1:line.find(",)")])

                # Fill in fitnesses dictionary.
                best_fitnesses[filename] = best_fitness_ind
            else: # It's a NES run
                for n, line in enumerate(f):
                    best_fitness_string = "Best Fitness: "
                    avg_fitness_string = "Average Fitness: "
                    if best_fitness_string in line:
                        this_fitness = -float(line[line.find(best_fitness_string):].replace(best_fitness_string, ""))
                        if this_fitness < best_fitness_ind:
                            best_fitness_ind = this_fitness
                            best_n = n + 3
                    if avg_fitness_string in line:
                        this_fitness_gen = float(line[line.find(avg_fitness_string):].replace(avg_fitness_string, ""))
                        if this_fitness_gen < best_fitness_gen:
                            best_fitness_gen= this_fitness_gen
                
                # Go through them again to find the best individual
                f.seek(0)
                for n, line in enumerate(f):
                    if n == best_n:
                        dict_str = line[line.find("{"):line.find("}") + 1]
                        best_dict = ast.literal_eval(dict_str)
                        best_dict = OrderedDict((k, best_dict[k]) for k in hps if k in best_dict)
                        best_dicts[filename] = best_dict

                # Fill in fitnesses dictionaries.
                best_fitnesses[filename] = best_fitness_ind
                best_gen_fitnesses[filename] = best_fitness_gen

    # Remove files that do not have all the variables
    remove_list = []
    for fn in best_dicts.keys():
        not_found = len([n for n in hps if n not in best_dicts[fn]]) > 0
        if not_found:
            remove_list.append(fn)

    for i in remove_list:
        best_dicts.pop(i)
        best_fitnesses.pop(i)
        best_gen_fitnesses.pop(i)

    # Print table of HPs
    relevant_fns = [fn for fn in filenames if fn in best_dicts]
    for fn in relevant_fns:
        last_slash_pos = fn.rfind('\\')
        last_dot_post = fn.rfind('.')
        print("{}:".format(fn[last_slash_pos + 1:last_dot_post]))
        print("Best fitness: {}".format(best_fitnesses[fn]))
        if "NES" in fn:
            print("Best gen. fitness: {}".format(best_gen_fitnesses[fn]))
        print("")

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

    return best_dicts, best_fitnesses, best_gen_fitnesses


def run_best_sam(resolution, fixed_delay, use_pecevski, num_trials, seed):
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

    # Determine delay.
    delay = float(re.search('SAM-(.*)ms', fns[i]).group(1).replace('_', '.'))

    # Get best hps in the chosen log file.
    params = hps[fns[i]]
    params['stdp_time_fraction'] = 1.0
    params['intrinsic_step_time_fraction'] = 1.0
    params['learning_time'] = 300000
    params['delay'] = delay

    # Create the SAM optimizee.
    optimizee = SAMOptimizee(traj, 
                            use_pecevski=use_pecevski, 
                            n_NEST_threads=1, 
                            time_resolution=resolution,
                            fixed_delay=fixed_delay,
                            plots_directory='/home/krisdamato/LTL-SAM/plots/', 
                            forced_params=params,
                            plot_all=False,
                            seed=seed,
                            num_fitness_trials=num_trials)

    # Run simulation with the forced params.
    optimizee.simulate(traj)


def run_best_samgraph(resolution, fixed_delay, use_pecevski, num_trials, state_handling, seed):
    '''Runs the best SAM setup in the log file chosen by the user.'''

    import logging.config
    
    from sam.optimizee import SAMOptimizee
    from ltl import sdict

    from pypet import Environment, pypetconstants
    from ltl.logging_tools import create_shared_logger_data, configure_loggers
    from ltl.paths import Paths
    from sam.optimizee import SAMGraphOptimizee

    logger = logging.getLogger('bin.ltl-samgraph-ga')

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
   
    fns, hps = process_samgraph_results('/home/krisdamato/LTL-SAM/results/')
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

    # Determine delay.
    delay = float(re.search('SAMGRAPH-(.*)ms', fns[i]).group(1).replace('_', '.'))

    # Get best hps in the chosen log file.
    params = hps[fns[i]]
    params['stdp_time_fraction'] = 1.0
    params['intrinsic_step_time_fraction'] = 1.0
    params['learning_time'] = 300000
    params['delay'] = delay

    # Create the SAM optimizee.
    optimizee = SAMGraphOptimizee(traj, 
                            use_pecevski=use_pecevski, 
                            n_NEST_threads=1, 
                            time_resolution=resolution,
                            fixed_delay=fixed_delay,
                            plots_directory='/home/krisdamato/LTL-SAM/plots/', 
                            forced_params=params,
                            plot_all=False,
                            seed=seed,
                            state_handling=state_handling,
                            num_fitness_trials=num_trials)

    # Run simulation with the forced params.
    optimizee.simulate(traj)


def run_best_spi(resolution, fixed_delay, min_delay, max_delay, use_pecevski, num_trials, seed):
    '''Runs the best SPI setup in the log file chosen by the user.'''
    import logging.config
    
    from sam.optimizee import SPIConditionalNetworkOptimizee
    from ltl import sdict

    from pypet import Environment, pypetconstants
    from ltl.logging_tools import create_shared_logger_data, configure_loggers
    from ltl.paths import Paths

    logger = logging.getLogger('bin.ltl-spi-ga')

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
   
    fns, hps = process_spi_results('/home/krisdamato/LTL-SAM/results/')
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
    params['intrinsic_step_time_fraction'] = 1.0
    params['stdp_time_fraction'] = 1.0
    params['learning_time'] = 300000

    # Create the SAM optimizee.
    optimizee = SPIConditionalNetworkOptimizee(traj, 
                            use_pecevski=use_pecevski, 
                            n_NEST_threads=1, 
                            time_resolution=resolution,
                            min_delay=min_delay,
                            max_delay=max_delay,
                            fixed_delay=fixed_delay,
                            plots_directory='/home/krisdamato/LTL-SAM/plots/', 
                            forced_params=params,
                            plot_all=False,
                            seed=seed,
                            num_fitness_trials=num_trials)

    # Run simulation with the forced params.
    optimizee.simulate(traj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--copy', action='store_true', help='Just copy log files to LTL-SAM folder')
    parser.add_argument('-rs', '--run_sam', action='store_true', help='Run best SAM in log files, asking for user input to select which log file to use.')
    parser.add_argument('-rsg', '--run_sam_graph', action='store_true', help='Run best SAMGRAPH in log files, asking for user input to select which log file to use.')
    parser.add_argument('-rspi', '--run_spi', action='store_true', help='Run best SPI in log files, asking for user input to select which log file to use.')
    parser.add_argument('-rspig', '--run_spi_graph', action='store_true', help='Run best SPIGRAPH in log files, asking for user input to select which log file to use.')
    parser.add_argument('-r', '--resolution', required=False, type=float, help='Resolution')
    parser.add_argument('-fd', '--fixed_delay', required=False, type=float, help='Fixed delay')
    parser.add_argument('-mind', '--min_delay', required=False, type=float, help='Min delay')
    parser.add_argument('-maxd', '--max_delay', required=False, type=float, help='Max delay')
    parser.add_argument('-p', '--use_pecevski', action='store_true', help='Use Pecevski distributions')
    parser.add_argument('-nt', '--num_trials', required=False, type=int, help='Number of trials')
    parser.add_argument('-sd', '--seed', required=False, type=int, help='Random seed')
    parser.add_argument('-s', '--state_handling', required=False, help='State interpretation type (none, first, random)')

    args = parser.parse_args()

    if args.copy: copy_log_files_to("D:\\LTL-SAM\\results\\")
    elif args.run_sam: run_best_sam(resolution=args.resolution, fixed_delay=args.fixed_delay, use_pecevski=args.use_pecevski, num_trials=args.num_trials, seed=args.seed)
    elif args.run_sam_graph: run_best_samgraph(resolution=args.resolution, fixed_delay=args.fixed_delay, use_pecevski=args.use_pecevski, num_trials=args.num_trials, state_handling=args.state_handling, seed=args.seed)
    elif args.run_spi: run_best_spi(resolution=args.resolution, fixed_delay=args.fixed_delay, min_delay=args.min_delay, max_delay=args.max_delay, use_pecevski=args.use_pecevski, num_trials=args.num_trials, seed=args.seed)
    elif args.run_spi_graph: run_best_spigraph(resolution=args.resolution, fixed_delay=args.fixed_delay, min_delay=args.min_delay, max_delay=args.max_delay, use_pecevski=args.use_pecevski, num_trials=args.num_trials, seed=args.seed)
    else: process_spigraph_results(search_str='0_2*Random')
