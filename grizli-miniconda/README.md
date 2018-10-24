(Under Development)

# docker-miniconda

Docker container running [amazonlinux](https://hub.docker.com/_/amazonlinux/) with [Grizli](https://github.com/gbrammer/grizli) running in an [Miniconda](http://conda.pydata.org/miniconda.html) environment (py35).

Usage
-----

You can download and run this image using the following commands:

    docker pull gbrammer/grizli-miniconda
    docker run -i -t gbrammer/grizli-miniconda /bin/bash

*(testing notebook implementation)*

Following the instructions for [ContinuumIO/docker-images/miniconda3](https://github.com/ContinuumIO/docker-images/tree/master/miniconda3) you can also run the image in a Jupyter Notebook server:

    # Set some working directory where products can be saved 
    # outside of the docker container
    DOCKER_WORKDIR=/tmp/ 
    
    # Avoid collision with non-docker notebooks at 8888
    NOTEBOOK_PORT=8008 
    
    # Start the container.
    docker run -v $DOCKER_WORKDIR:/workdir -i -t -p $NOTEBOOK_PORT:$NOTEBOOK_PORT gbrammer/grizli-miniconda /bin/bash -c ". /opt/conda/etc/profile.d/conda.sh && conda activate grizli-dev && cd /workdir && jupyter notebook --notebook-dir=/opt/notebooks --ip='0.0.0.0' --port=${NOTEBOOK_PORT} --no-browser --allow-root"

Then view the Jupyter notebook by opening `http://localhost:8008` in your browser. If jupyter requests a login token at the initial screen, look for the token in the shell where you initialized the docker image.



