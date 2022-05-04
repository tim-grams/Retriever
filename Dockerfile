FROM continuumio/miniconda3:4.10.3p1
WORKDIR /project
COPY requirements.txt /project/requirements.txt
RUN conda install \
    xarray \
    netCDF4 \
    bottleneck \
    numpy \
    pandas \
    matplotlib \
    jupyterlab

# some packages not avaliable on the default channel e.i. tarfile
Run conda config --append channels conda-forge

#Run conda install tarfile

Run pip install -r /project/requirements.txt

COPY ./README.rst /project
COPY ./data /project/data
COPY ./models /project/models
COPY ./src /project/src
COPY ./notebooks /project/notebooks
# gives non-user root all permissions, add non-root user
#COPY . .
CMD ["jupyter-lab","--ip=0.0.0.0","--no-browser","--allow-root"]




