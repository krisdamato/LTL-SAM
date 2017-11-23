rm -r /home/krisdamato/nest-build

# First you have to create a python 3.5 environment using conda and activate it:
# conda create -n python35 python=3.5
source activate python35

# If nose is installed in root environment, it may try to use Python 2.7, so 
# first we reinstall it in this environment.
pip install nose
pip install numpy
pip install scipy
pip install matplotlib

# Install packages for environment. If this doesn't work,
# use pip install instead (after activating environment).
apt-get install -y build-essential cmake libltdl7-dev libreadline6-dev \
libncurses5-dev libgsl0-dev python-all-dev python-numpy python-scipy \
python-matplotlib ipython openmpi-bin libopenmpi-dev python-nose cython python3-dev

cd /home/krisdamato/
mkdir nest-build
cd nest-build
cmake -DCMAKE_INSTALL_PREFIX:PATH=/home/krisdamato/NEST \
	  /home/krisdamato/nest-simulator \
	  -Dwith-python=3 \
	  -Dwith-gsl=/usr/lib/x86_64-linux-gnu
make -j3
make -j3 install
source /home/krisdamato/NEST/bin/nest_vars.sh
make -j3 installcheck
