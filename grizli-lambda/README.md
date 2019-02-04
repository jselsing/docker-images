(Under Development)

# grizli-lambda

Docker container running [amazonlinux:2017.09](https://hub.docker.com/_/amazonlinux/) and a minimal [Grizli](https://github.com/gbrammer/grizli) environment (py36) sufficient to run redshift fits in AWS lambda.

Usage
-----

Download and run this image using the following commands:

    docker pull gbrammer/grizli-lambda
    docker run -v $(pwd):/workdir -i -t gbrammer/grizli-lambda /bin/bash -c "bash /tmp/package_venv.sh /venv/"
    ls -lth venv.zip

All `*.py` files in the working directory `$(pwd)` are copied to the zipped python environment and pushed to the zip file, which are then accessible to AWS Lambda functions. 


