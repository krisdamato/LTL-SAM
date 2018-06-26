source ../NEST/bin/nest_vars.sh

python -m scoop -n 96 bin/ltl-sam-ga.py -n SAM-0_1ms-GA-Pecevski -r 0.1 -fd 0.1 -p -nt 10
python -m scoop -n 96 bin/ltl-sam-ga.py -n SAM-0_2ms-GA-Pecevski -r 0.1 -fd 0.2 -p -nt 10
python -m scoop -n 96 bin/ltl-sam-ga.py -n SAM-0_5ms-GA-Pecevski -r 0.1 -fd 0.5 -p -nt 10
python -m scoop -n 96 bin/ltl-sam-ga.py -n SAM-1_0ms-GA-Pecevski -r 0.1 -fd 1.0 -p -nt 10
python -m scoop -n 96 bin/ltl-sam-ga.py -n SAM-2_0ms-GA-Pecevski -r 0.1 -fd 2.0 -p -nt 10

python -m scoop -n 96 bin/ltl-sam-nes.py -n SAM-0_1ms-NES-Pecevski -r 0.1 -fd 0.1 -p -nt 1
python -m scoop -n 96 bin/ltl-sam-nes.py -n SAM-0_2ms-NES-Pecevski -r 0.1 -fd 0.2 -p -nt 1
python -m scoop -n 96 bin/ltl-sam-nes.py -n SAM-0_5ms-NES-Pecevski -r 0.1 -fd 0.5 -p -nt 1
python -m scoop -n 96 bin/ltl-sam-nes.py -n SAM-1_0ms-NES-Pecevski -r 0.1 -fd 1.0 -p -nt 1
python -m scoop -n 96 bin/ltl-sam-nes.py -n SAM-2_0ms-NES-Pecevski -r 0.1 -fd 2.0 -p -nt 1

