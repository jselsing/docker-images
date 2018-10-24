(Under Development)

# docker-miniconda

Docker container running [amazonlinux](https://hub.docker.com/_/amazonlinux/)
with [Grizli](https://github.com/gbrammer/grizli) running in 
an [Miniconda](http://conda.pydata.org/miniconda.html) environment (py35).

Usage
-----

You can download and run this image using the following commands:

    docker pull gbrammer/grizli-miniconda
    docker run -i -t gbrammer/grizli-miniconda /bin/bash

*(jupyter notebook not working)*

Following the instructions for [ContinuumIO/docker-images/miniconda3](https://raw.githubusercontent.com/ContinuumIO/docker-images/miniconda3)
you can also run the image in a Jupyter Notebook server:

    docker run -i -t -p 8888:8888 gbrammer/grizli-miniconda /bin/bash -c ". /opt/conda/etc/profile.d/conda.sh; conda activate grizli-dev; conda install -y jupyter; jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser"
    
You can then view the Jupyter Notebook by opening `http://localhost:8888` 
in your browser, or `http://<DOCKER-MACHINE-IP>:8888` 
if you are using a Docker Machine VM.
