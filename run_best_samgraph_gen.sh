source ../NEST/bin/nest_vars.sh

python tools/process_results.py -rsg -r 0.1 -fd 0.2 -nt 5 -sd 1 -s first
