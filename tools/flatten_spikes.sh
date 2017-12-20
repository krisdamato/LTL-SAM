docker run -it spikes bash -c "exit"
container_name=echo $(docker ps -a | grep spikes) | cut -d' ' -f1
docker export $container_name | docker import - spikes_flat:latest

