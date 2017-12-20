# Checkout SAM branch in NEST-simulator.
cd /home/krisdamato/nest-simulator
git checkout pecevski2016_srm

# Create a build directory.
rm -r /home/krisdamato/sam-build
mkdir /home/krisdamato/sam-build
cd /home/krisdamato/sam-build

# Build SAM
source /home/krisdamato/NEST/bin/nest_vars.sh
rm -r /home/krisdamato/sam
rm /home/krisdamato/NEST/lib/nest/sammodule.so /home/krisdamato/NEST/lib/nest/libsammodule.so
cp -r /home/krisdamato/nest-simulator/sam /home/krisdamato/
cmake -Dwith-nest=${NEST_INSTALL_DIR}/bin/nest-config ../sam
make
make install
