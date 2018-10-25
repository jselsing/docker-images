FROM amazonlinux:latest

#  docker build . -t gbrammer/grizli:latest -t gbrammer/grizli:2018.10a
#  docker run --rm -it gbrammer/grizli:latest /bin/bash
#  docker push gbrammer/grizli:latest
#  docker push gbrammer/grizli:2018.10a

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

##
## Linux tools
##

RUN yum update -y && \
    yum install -y git vi which bzip2 curl wget tar make gcc gcc-c++ gcc-gfortran

##
## miniconda3 (Code taken from continuumio/miniconda3)
##
RUN cd /usr/local/share && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /etc/profile.d/conda.sh" >> /root/.bashrc && \
    . /etc/profile.d/conda.sh

##
## ESO fits tools
##
RUN cd /tmp && \
    wget http://www.eso.org/~jpritcha/jFEROS-DRS/FEROS-DRS/src/dfits.c && \
    wget http://www.eso.org/~jpritcha/jFEROS-DRS/FEROS-DRS/src/fitsort.c && \
    gcc -o dfits dfits.c && \
    gcc -o fitsort fitsort.c && \
    cp dfits fitsort /usr/local/bin

##   
## Grizli installation
##
RUN cd /usr/local/share && \
    mkdir python && \
    cd python && \
    git clone https://github.com/gbrammer/grizli.git && \
    cd grizli && \
    . /etc/profile.d/conda.sh && \
    conda env create -f environment_min.yml -n grizli-dev && \
    echo "conda activate grizli-dev" >> /root/.bashrc && \
    conda activate grizli-dev && \
    python setup.py install

## Environment

RUN mkdir /usr/local/share/grizli               && \
    mkdir /usr/local/share/grizli/CONF          && \
    mkdir /usr/local/share/grizli/templates     && \
    mkdir /usr/local/share/grizli/iref          && \
    mkdir /usr/local/share/grizli/jref          && \
    mkdir /usr/local/share/grizli/uref
    
RUN echo "export GRIZLI=\"/usr/local/share/grizli\"" >> /root/.bashrc     && \
    echo "export GRIZLICODE=\"/usr/local/share/python/grizli\"" >> /root/.bashrc && \
    echo "export iref=\"\${GRIZLI}/iref/\"" >> /root/.bashrc              && \
    echo "export jref=\"\${GRIZLI}/jref/\"" >> /root/.bashrc              && \
    echo "export uref=\"\${GRIZLI}/uref/\"" >> /root/.bashrc

ENV GRIZLI /usr/local/share/grizli
ENV iref   /usr/local/share/grizli/iref/
ENV jref   /usr/local/share/grizli/jref/

RUN . /etc/profile.d/conda.sh && \
    conda activate grizli-dev && \
    python -c "import grizli.utils; grizli.utils.fetch_default_calibs(ACS=False)"  && \
    python -c "import grizli.utils; grizli.utils.fetch_config_files()"             && \
    python -c "import grizli.utils; grizli.utils.fetch_default_calibs(ACS=True)"   && \
    python -c "import grizli.utils; grizli.utils.fetch_config_files(ACS=True)"     && \
    python -c "import grizli.utils; grizli.utils.symlink_templates(force=False)"

## Jupyter Notebooks

RUN mkdir /opt/notebooks && \
    cd /usr/local/share/python/grizli/examples/ && \
    cp Grizli-Pipeline.ipynb /opt/notebooks/ && \
    cp Fit-with-Photometry.ipynb /opt/notebooks/

## Prospector for grizli + eazy, just uses the smoothing function....

RUN . /etc/profile.d/conda.sh && \
    conda activate grizli-dev && \
    cd /usr/local/share/python && \
    git clone https://github.com/bd-j/prospector.git && \
    cd prospector && \
    python setup.py install


# ## EAZY-py (already installed with grizli)
# RUN . /etc/profile.d/conda.sh && \
#     conda activate grizli-dev && \
#     cd /usr/local/share/python && \
#     git clone --recursive git@github.com:gbrammer/eazy-py.git && \
#     cd eazy-py && \
#     ln -sf ../eazy-photoz . && \
#     python setup.py install && \
#     echo "export EAZYCODE=\"/usr/local/share/python/eazy-photoz\"" >> /root/.bashrc 
    
##
## MPI Processing
##

# RUN cd /usr/local/share/ && \
#     wget http://www.mpich.org/static/downloads/3.2.1/mpich-3.2.1.tar.gz && \
#     tar xzvf mpich-3.2.1.tar.gz && \
#     cd mpich-3.2.1 && \
#     ./configure && \
#     make && \
#     make install && \
#     cd ../ && \
#     rm -rf mpich-3.2.1*
# 
# RUN . /etc/profile.d/conda.sh && \
#     conda activate grizli-dev && \
#     conda install mpi4py

CMD [ "/bin/bash" ]