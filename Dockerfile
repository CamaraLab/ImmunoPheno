# Dockerfile for running ImmunoPheno

ARG OWNER=jupyter
ARG BASE_CONTAINER=$OWNER/scipy-notebook:python-3.10.11
FROM $BASE_CONTAINER

LABEL maintainer="Pablo Cámara <pcamara@pennmedicine.upenn.edu>"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# R pre-requisites and Graphviz
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    fonts-dejavu \
    gfortran \
    gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
    
WORKDIR /tmp

USER ${NB_UID}

# R packages including IRKernel which gets installed globally.
# r-e1071: dependency of the caret R package
RUN mamba install --yes \
    'r-base' \
    'r-biocmanager' \
    'r-caret' \
    'r-crayon' \
    'r-devtools' \
    'r-e1071' \
    'r-forecast' \
    'r-hexbin' \
    'r-htmltools' \
    'r-htmlwidgets' \
    'r-irkernel' \
    'r-nycflights13' \
    'r-randomforest' \
    'r-rcurl' \
    'r-rmarkdown' \
    'r-rodbc' \
    'r-rsqlite' \
    'r-shiny' \
    'r-tidymodels' \
    'r-tidyverse' \
    'rpy2' \
    'unixodbc' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN R -e 'BiocManager::install("SummarizedExperiment", ask=FALSE)' && \
    R -e 'BiocManager::install("SingleR", ask=FALSE)'

USER root

RUN apt-get update --yes && \
    apt-get install software-properties-common --yes && \
    add-apt-repository universe --yes && \
    apt-get install graphviz --yes && \
    apt-get update --yes

USER ${NB_UID}

# Python Packages
RUN pip install \
    'mysql-connector-python==8.0.32' \
    'gtfparse==1.3.0' \
    'pyensembl==2.2.8' \
    'umap-learn==0.5.3' \
    'plotly==5.13.1' \
    'statsmodels==0.13.5' \
    'seaborn==0.12.2' \
    'pynndescent==0.5.10' \
    'numba==0.58.0' \
    'scanpy==1.9.5' \
    'dash==2.11' \
    'dash-bootstrap-components==1.6.0' \
    "hdbscan==0.8.33" \
    "graphviz==0.20.3" \
    "pydot==2.0.0"
    
EXPOSE 8888 8050

WORKDIR "${HOME}"