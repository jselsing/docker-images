FROM amazonlinux:2017.09

#  docker build . -t gbrammer/grizli-lambda-test:latest
#  docker run --rm -it gbrammer/grizli-lambda-test:latest /bin/bash
#  docker push gbrammer/grizli:latest
#  docker push gbrammer/grizli:2018.10a

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
    
### Install minimal Python tools
RUN source ${VIRTUAL_ENV}/bin/activate && \
    pip install --upgrade setuptools && \
    pip install numpy && \
    pip install scipy && \
    pip install matplotlib && \
    pip install astropy && \
    pip install stwcs && \
    pip install peakutils && \
    pip install extinction && \
    pip install git+https://github.com/gbrammer/drizzle.git
    
### Special matplotlibrc file

# Initialize matplotib directory
RUN source ${VIRTUAL_ENV}/bin/activate && \
    python -c "import matplotlib"

ADD ./Install/matplotlibrc /root/.config/matplotlib/matplotlibrc
ADD ./Install/matplotlibrc ${VIRTUAL_ENV}/

####
# Install grizli
####
RUN cd /tmp/ && \
    source ${VIRTUAL_ENV}/bin/activate && \
    git clone https://github.com/gbrammer/grizli.git && \
    cd grizli && \
    python setup.py install && \
    cd ../ && \
    rm -rf grizli
    
# Config files in GRIZLI_PATH
RUN source ${VIRTUAL_ENV}/bin/activate && \
    GRIZLI_PATH=`python -c "import grizli; print(grizli.GRIZLI_PATH)" | tail -1`  && \
    echo $GRIZLI_PATH && \
    cp ${VIRTUAL_ENV}/matplotlibrc ${GRIZLI_PATH}/matplotlibrc && \
    mkdir $GRIZLI_PATH/CONF/ && \
    ln -s $GRIZLI_PATH/CONF /tmp/

ADD ./Install/setup_grizli.py /tmp/
RUN source ${VIRTUAL_ENV}/bin/activate && \
    cd /tmp/ && \
    python setup_grizli.py

############
## Final stripped and zipped virtual environment
ADD ./Install/process.py ${VIRTUAL_ENV}/
ADD ./Install/strip_virtualenv.sh /tmp/
RUN source ${VIRTUAL_ENV}/bin/activate && \
    cd /tmp/ && \
    bash /tmp/strip_virtualenv.sh ${VIRTUAL_ENV}

#############
## Install boto3 for testing, credentials file
RUN source ${VIRTUAL_ENV}/bin/activate && \
    pip install boto3 && \
    mkdir ~/.aws/ && \
    echo "Put AWS credentials in ~/.aws/credentials!"

ADD ./Install/aws_credentials ~/.aws/credentials

RUN mkdir /workdir

CMD [ "/bin/bash" ]