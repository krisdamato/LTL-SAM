source ../NEST/bin/nest_vars.sh

python -m scoop -n 96 bin/ltl-samgraph-ga.py -n SAMGRAPH-0_1ms-GA-Pecevski-handling_first -r 0.1 -fd 0.1 -p -s first -nt 5
python -m scoop -n 96 bin/ltl-samgraph-ga.py -n SAMGRAPH-0_2ms-GA-Pecevski-handling_first -r 0.1 -fd 0.2 -p -s first -nt 5
python -m scoop -n 96 bin/ltl-samgraph-ga.py -n SAMGRAPH-0_5ms-GA-Pecevski-handling_first -r 0.1 -fd 0.5 -p -s first -nt 5
python -m scoop -n 96 bin/ltl-samgraph-ga.py -n SAMGRAPH-1_0ms-GA-Pecevski-handling_first -r 0.1 -fd 1.0 -p -s first -nt 5
python -m scoop -n 96 bin/ltl-samgraph-ga.py -n SAMGRAPH-2_0ms-GA-Pecevski-handling_first -r 0.1 -fd 2.0 -p -s first -nt 5

python -m scoop -n 96 bin/ltl-samgraph-nes.py -n SAMGRAPH-0_1ms-NES-Pecevski-handling_first -r 0.1 -fd 0.1 -p -s first -nt 1
python -m scoop -n 96 bin/ltl-samgraph-nes.py -n SAMGRAPH-0_2ms-NES-Pecevski-handling_first -r 0.1 -fd 0.2 -p -s first -nt 1
python -m scoop -n 96 bin/ltl-samgraph-nes.py -n SAMGRAPH-0_5ms-NES-Pecevski-handling_first -r 0.1 -fd 0.5 -p -s first -nt 1
python -m scoop -n 96 bin/ltl-samgraph-nes.py -n SAMGRAPH-1_0ms-NES-Pecevski-handling_first -r 0.1 -fd 1.0 -p -s first -nt 1
python -m scoop -n 96 bin/ltl-samgraph-nes.py -n SAMGRAPH-2_0ms-NES-Pecevski-handling_first -r 0.1 -fd 2.0 -p -s first -nt 1
