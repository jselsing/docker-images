FROM amazonlinux:2017.09

#  docker build . -t jselsing/not-lambda:latest
#  docker run --rm -it jselsing/not-lambda:latest /bin/bash

#  ## Build the full venv.zip file and copy to AWS
#  cd /tmp/ ; docker run -v $(pwd):/workdir -i -t jselsing/not-lambda /bin/bash -c "cp /tmp/venv.zip /workdir/"; cd - ; aws s3 cp /tmp/venv.zip s3://aws-grivam/tmp/venv.zip

#ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

##
## Linux tools
##

RUN yum update -y && \
    yum install -y atlas-devel \
                   atlas-sse3-devel \
                   blas-devel \
                   gcc \
                   gcc-c++ \
                   lapack-devel \
                   python36-devel \
                   python36-virtualenv \
                   findutils \
                   git \
                   vi \
                   which \
                   zip

ENV VIRTUAL_ENV /venv

### Initialize Python3.6  virtualenv
RUN /usr/bin/virtualenv-3.6 \
    --python /usr/bin/python3.6 ${VIRTUAL_ENV} \
    --always-copy \
    --no-site-packages

### For cc1plus compiler
ENV PATH "${PATH}:/usr/libexec/gcc/x86_64-amazon-linux/4.8.5/"

### Install minimal Python tools
RUN source ${VIRTUAL_ENV}/bin/activate && \
    pip install --upgrade setuptools && \
    pip install wheel && \
    pip install numpy && \
    pip install scipy && \
    pip install matplotlib && \
    pip install astropy && \
    pip install astroplan


##### Special matplotlibrc file

# Initialize matplotib directory
RUN source ${VIRTUAL_ENV}/bin/activate && \
    python -c "import matplotlib"

ADD ./Install/matplotlibrc /root/.config/matplotlib/matplotlibrc
ADD ./Install/matplotlibrc ${VIRTUAL_ENV}/


############
## Final stripped and zipped virtual environment
ADD ./Install/process.py ${VIRTUAL_ENV}/
ADD ./Install/aws_drizzler.py ${VIRTUAL_ENV}/

ADD ./Install/package_venv.sh /tmp/
RUN source ${VIRTUAL_ENV}/bin/activate && \
    cd /tmp/ && \
    bash /tmp/package_venv.sh ${VIRTUAL_ENV}

#############
## Install boto3 for testing, credentials file
RUN source ${VIRTUAL_ENV}/bin/activate && \
    pip install boto3 && \
    mkdir ~/.aws/ && \
    echo "Put AWS credentials in ~/.aws/credentials"

ADD ./Install/aws_credentials ~/.aws/credentials

ADD ./Install/add_workdir_python /usr/local/bin/add_workdir_python

# ############
# ## Working directory for passing data in/out
# RUN mkdir /workdir

#############
# REF directories
RUN mkdir /tmp/iref && \
    mkdir /tmp/uref && \
    mkdir /tmp/jref

ENV iref /tmp/iref/
ENV jref /tmp/jref/
ENV uref /tmp/uref/

CMD [ "/bin/bash" ]
