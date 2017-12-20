curl -O https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz
mkdir $HOME/openmpi
mv openmpi-3.0.0.tar.gz $HOME/openmpi
cd $HOME/openmpi
tar -xzvf openmpi-3.0.0.tar.gz
cd openmpi-3.0.0.tar.gz
./configure --prefix=$HOME/openmpi
make all
make install
export PATH=$PATH:$HOME/openmpi/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/openmpi/lib
