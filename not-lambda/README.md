(Under Development)

# grizli-lambda

Docker container running [amazonlinux:2017.09](https://hub.docker.com/_/amazonlinux/) and a minimal [Grizli](https://github.com/gbrammer/grizli) environment (py36) sufficient to run redshift fits in AWS lambda.

Usage
-----

Download and run this image using the following commands:

    docker pull gbrammer/grizli-lambda
    docker run -v $(pwd):/workdir -i -t jselsing/not-lambda /bin/bash -c "add_workdir_python"
    ls -lth venv_script.zip

All `*.py` files in the working directory `$(pwd)` are copied to the zipped python environment and pushed to the `venv_script.zip` file.  The python scripts will be available to AWS Lambda after that zip file is uploaded to a lambda instance. 


