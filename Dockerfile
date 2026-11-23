# CUDA + ASTRA (conda) image for NeTT.  docker build -t nett . ; docker run --gpus all -v $PWD/data:/work/data nett nett check
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends wget git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh \
 && bash /tmp/mf.sh -b -p /opt/conda && rm /tmp/mf.sh
ENV PATH=/opt/conda/bin:$PATH
WORKDIR /work
COPY environment.yml pyproject.toml README.md ./
COPY nett ./nett
RUN conda env update -n base -f environment.yml && conda clean -afy
ENV NETT_DATA=/work/data NETT_RESULTS=/work/results
ENTRYPOINT []
