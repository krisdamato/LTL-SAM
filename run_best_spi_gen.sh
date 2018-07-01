source ../NEST/bin/nest_vars.sh

python tools/process_results.py -rspi -r 0.1 -mind 0.1 -fd 0.2 -maxd 0.3 -nt 10 -sd 1
