python -m scoop -n 96 bin/ltl-spi-ga.py -n SPI-0_1ms-0_2ms-0_3ms-GA-Pecevski -r 0.1 -mind 0.1 -maxd 0.3 -fd 0.2 -nt 10
python -m scoop -n 96 bin/ltl-spi-nes.py -n SPI-0_1ms-0_2ms-0_3ms-NES-Pecevski -r 0.1 -mind 0.1 -maxd 0.3 -fd 0.2 -nt 10

python -m scoop -n 96 bin/ltl-spigraph-ga.py -n SPIGRAPH-0_1ms-0_2ms-0_3ms-GA-Pecevski -r 0.1 -mind 0.1 -maxd 0.3 -fd 0.2 -nt 5
python -m scoop -n 96 bin/ltl-spigraph-nes.py -n SPIGRAPH-0_1ms-0_2ms-0_3ms-NES-Pecevski -r 0.1 -mind 0.1 -maxd 0.3 -fd 0.2 -nt 5

