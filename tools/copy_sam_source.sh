rm -r /home/krisdamato/sam
rm /home/krisdamato/NEST/lib/nest/sammodule.so /home/krisdamato/NEST/lib/nest/libsammodule.so
cp -r /home/krisdamato/nest-simulator/sam /home/krisdamato/
cmake -Dwith-nest=${NEST_INSTALL_DIR}/bin/nest-config ../sam
make
make install
