source ../NEST/bin/nest_vars.sh

python -m scoop -n 96 bin/ltl-samgraph-ga.py -n SAMGRAPH-0_1ms-GA-Pecevski -r 0.1 -fd 0.1 -p
python -m scoop -n 96 bin/ltl-samgraph-ga.py -n SAMGRAPH-0_2ms-GA-Pecevski -r 0.1 -fd 0.2 -p
python -m scoop -n 96 bin/ltl-samgraph-ga.py -n SAMGRAPH-0_5ms-GA-Pecevski -r 0.1 -fd 0.5 -p
python -m scoop -n 96 bin/ltl-samgraph-ga.py -n SAMGRAPH-1_0ms-GA-Pecevski -r 0.1 -fd 1.0 -p
python -m scoop -n 96 bin/ltl-samgraph-ga.py -n SAMGRAPH-2_0ms-GA-Pecevski -r 0.1 -fd 2.0 -p