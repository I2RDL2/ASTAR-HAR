echo -n "Enter name of the docker > "
read docker_name
echo "docker $docker_name is setup"
# nvidia-docker run -it --name $docker_name --network="host" \
    # -v /home/luk/hd/project/har:/root/ loklu/mt_tensorflow:tf1.2.1_py35_lib3

nvidia-docker run -it --name $docker_name -v \
/home/i2r/luk/scratch/project/har:/root/ loklu/mt_tensorflow:tf1.2.1_py35_lib3
