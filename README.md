# ImmunoPheno
[![tests](https://github.com/CamaraLab/ImmunoPheno/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/CamaraLab/ImmunoPheno/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/CamaraLab/ImmunoPheno/branch/package-dev/graph/badge.svg?token=R7GBNL9ST6)](https://codecov.io/gh/CamaraLab/ImmunoPheno)

ImmunoPheno is a Python library for the design, annotation, and analysis of multiplexed cytometry data based on 
existing single-cell CITE-seq datasets. It implements functionalities for finding antibodies for gating specific 
cell populations, designing optimal antibody panels for multiplexed cytometry experiments, normalizing cytometry 
(CODEX, spectral flow cytometry, CyTOF, Imaging Mass Cytometry, mIHC, etc.) and CITE-seq data, and identifying and 
annotating cell populations in these data.

## Installation
Until we upload the package to PyPI, the pip installation works from GitHub:
```commandline
pip install git+https://github.com/CamaraLab/ImmunoPheno.git
```
Installation on a standard desktop computer should take a few minutes.

The easiest way to run ImmunoPheno is via [Jupyter](https://jupyter.org/). Install Jupyter with
```commandline
pip install notebook
```
Then start up Jupyter from terminal / Powershell using
```commandline
jupyter notebook
```

## Docker image
We provide a Docker image that contains ImmunoPheno and its dependencies. Running the following command will launch 
a Jupyter notebook server on localhost with ImmunoPheno and its dependencies installed:
```commandline
docker run -it -p 8888:8888 -p 8050:8050 -v C:\Users\myusername\Documents\myfolder:/home/jovyan/work camaralab/python3:immunopheno
```
The ```-p``` flag controls the port number on local host. For example, writing ```-p 4264:8888``` will let you access 
the Jupyter server from 127.0.0.1:4264. The ```-v``` "bind mount" flag allows one to mount a local directory on the 
host machine to a folder inside the container so that you can read and write files on the host machine from within 
the Docker image. Here one must mount the folder on the host machine as /home/jovyan/work or 
/home/jovyan/some_other_folder as the primary user "jovyan" in the Docker image only has access to that directory. 
See the [Jupyter docker image documentation](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html) 
for more information.

## Documentation
Documentation and tutorials can be found on [ImmunoPheno's readthedocs.io website](https://immunopheno.readthedocs.io/en/latest/index.html).
