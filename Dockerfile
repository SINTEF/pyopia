FROM continuumio/miniconda3

WORKDIR /src
COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends git

RUN conda env create -f environment.yml --force

