source ../NEST/bin/nest_vars.sh

python -m scoop -n 96 bin/ltl-sam-ga.py -n SAM-0_2ms-GA-Random -r 0.1 -fd 0.2 -nt 10
python -m scoop -n 96 bin/ltl-sam-nes.py -n SAM-0_2ms-NES-Random -r 0.1 -fd 0.2 -nt 10

python -m scoop -n 96 bin/ltl-samgraph-ga.py -n SAMGRAPH-0_2ms-GA-Random-handling_first -r 0.1 -fd 0.2 -s first -nt 5
python -m scoop -n 96 bin/ltl-samgraph-nes.py -n SAMGRAPH-0_2ms-NES-Random-handling_first -r 0.1 -fd 0.2 -s first -nt 5