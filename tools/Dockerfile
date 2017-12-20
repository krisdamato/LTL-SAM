FROM python:3.5.4
WORKDIR /home/krisdamato
COPY ./ ./
RUN cd /tmp

# Set proxies (will need to be set again once the container is run)
ENV http_proxy="http://proxy.um.edu.mt:80/"
ENV https_proxy="http://proxy.um.edu.mt:80/"

# Install apt upgrades and necessary packages
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git bzip2

# Install CMake 3.10 (apt-get install cmake doesn't get latest)
RUN apt-get purge -y cmake && mkdir ~/temp && cd ~/temp && wget https://cmake.org/files/v3.10/cmake-3.10.0.tar.gz && tar -xzvf cmake-3.10.0.tar.gz && cd cmake-3.10.0/ && ./bootstrap && make -j4 && make install && rm -r ~/temp && cd /home/krisdamato

# Install NEST
RUN bash /home/krisdamato/LTL-SAM/tools/build_nest.sh

# Install SAM
RUN bash /home/krisdamato/LTL-SAM/tools/build_sam.sh && bash /home/krisdamato/LTL-SAM/tools/install_sam.sh
