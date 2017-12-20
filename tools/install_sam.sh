# Install LTL requirements and checkout genetic operators branch
cd /home/krisdamato/LTL
pip3 install --editable . --process-dependency-links 
git checkout genetic_bit_operators_rebased

# Install requirements for SAM and create results directory
cd /home/krisdamato/LTL-SAM
pip3 install mpi4py bitstring
mkdir results
echo results > /bin/path.conf
