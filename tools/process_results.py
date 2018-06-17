import argparse
import ast
import glob
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
                if "generation 19" in line:
                    best_n = n + 2
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


def run_best_sam(resolution, fixed_delay, use_pecevski):
    '''Runs the best SAM setup in the log file chosen by the user.'''

    from sam.optimizee import SAMOptimizee
    from ltl import DummyTrajectory
    from ltl import sdict

    print("Running with resolution = {}, fixed delay = {}, use_pecevski = {}".format(resolution, fixed_delay, use_pecevski))
   
    fns, hps = process_sam_results('/home/krisdamato/LTL-SAM/results/')
    for i, fn in enumerate(fns):
        print("{}: {}".format(i, fn))

    try:
        i = int(input('Choose log index:'))
    except ValueError:
        print("Not a number!")
        return

    if i >= len(fns):
        print("Choose within the range!")
        return

    # Get best hps in the chosen log file.
    params = hps[fns[i]]

    # Create a dummy trajectory.
    fake_traj = DummyTrajectory()
    fake_traj.individual = sdict(optimizee.create_individual())

    # Create the SAM optimizee.
    optimizee = SAMOptimizee(fake_traj, 
                            use_pecevski=use_pecevski, 
                            n_NEST_threads=1, 
                            time_resolution=resolution,
                            fixed_delay=fixed_delay,
                            plots_directory='/home/krisdamato/LTL-SAM/plots/', 
                            forced_params=params,
                            plot_all=True,
                            num_fitness_trials=10)
    
    # Run simulation with the forced params.
    optimizee.simulate(fake_traj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--copy', action='store_true', help='Just copy log files to LTL-SAM folder')
    parser.add_argument('-rs', '--run_sam', action='store_true', help='Run best SAM in log files, asking for user input to select which log file to use.')
    parser.add_argument('-r', '--resolution', required=False, type=float, help='Resolution')
    parser.add_argument('-fd', '--fixed_delay', required=False, type=float, help='Fixed delay')
    parser.add_argument('-p', '--use_pecevski', action='store_true', help='Use Pecevski distributions')
    args = parser.parse_args()

    if args.copy: copy_log_files_to("D:\\LTL-SAM\\results\\")
    if args.run_sam: run_best_sam(resolution=args.resolution, fixed_delay=args.fixed_delay, use_pecevski=args.use_pecevski)
