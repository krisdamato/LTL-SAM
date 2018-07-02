source ../NEST/bin/nest_vars.sh

python tools/process_results.py -rs -r 0.1 -fd 0.2 -nt 10 -sd 1
