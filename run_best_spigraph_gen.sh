source ../NEST/bin/nest_vars.sh

python tools/process_results.py -rspig -r 0.1 -mind 0.1 -fd 0.2 -maxd 0.3 -nt 5 -sd 1
