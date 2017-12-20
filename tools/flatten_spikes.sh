docker run -it krisdamato/spikes bash -c "exit"
container_name=$(echo $(docker ps -a | grep spikes) | cut -d' ' -f1)
echo $container_name
docker export $container_name | docker import - spikes_flat:latest

