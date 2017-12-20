# Install LTL requirements and checkout genetic operators branch
cd /home/krisdamato/LTL
pip3 install --editable . --process-dependency-links 
git checkout genetic_bit_operators_rebased

# Install requirements for SAM and create results directory
cd /home/krisdamato/LTL-SAM
pip3 install mpi4py bitstring
mkdir results
echo results > /bin/path.conf

# Modify matplotlib to use different backend
sed -i 's/backend      : TkAgg/backend : Agg/g' '/usr/local/lib/python3.5/site-packages/matplotlib/mpl-data/matplotlibrc'
