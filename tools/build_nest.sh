cmake -DCMAKE_INSTALL_PREFIX:PATH=/home/krisdamato/NEST \
	  /home/krisdamato/nest-simulator \
	  -Dwith-python=3 \
	  -Dwith-gsl=/usr/lib/x86_64-linux-gnu \
	  -DPYTHON_EXECUTABLE=/usr/bin/python3.5 \
	  -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so.1 \
	  -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m \
	  -DPYTHON_INCLUDE_DIR2=/usr/include/x86_64-linux-gnu/python3.5m
make
make install
source /home/krisdamato/NEST/bin/nest_vars.sh
make installcheck
