# syntax=docker/dockerfile:labs
FROM tensorflow/tensorflow:2.5.0
ARG USE_GPU=""

RUN apt-get update && apt-get install -y git wget && pip install -U pip

RUN curl -fsSL https://get.docker.com -o get-docker.sh
RUN sh ./get-docker.sh --dry-run
RUN pip install docker

RUN git clone --depth 1 --branch v1.9.0 https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow && pip install -e .

RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.gz && tar -xvzf boost_1_77_0.tar.gz && cd boost_1_77_0 && ./bootstrap.sh --with-python=python3 --with-libraries=python,system && ./b2 install

RUN apt-get install -y cmake
RUN apt-get -y install m4 autoconf libtool texlive-latex-base
RUN apt-get -y install vim
COPY githubKey /root/.ssh/known_hosts

ADD https://github.com/yodarocks1/eran.git /src/eran/
WORKDIR /src/eran
RUN if [ -z "$USE_GPU" ]; then ./install.sh; else ./install.sh --use-cuda; fi
RUN bash -c "source ./gurobi_setup_path.sh"
ENV PYTHONPATH="/src/eran/python_interface/"

RUN rm /root/.bashrc
ADD . /src/VERAPAK
RUN cd /src/VERAPAK && git remote set-url origin https://github.com/formal-verification-research/VERAPAK.git
RUN ln /src/VERAPAK/docker.bashrc /root/.bashrc
RUN ln -s /src/VERAPAK/examples /root/examples
RUN cat /src/VERAPAK/githubKey >> /root/.ssh/known_hosts

RUN mkdir /src/VERAPAK/_build && cd /src/VERAPAK/_build && cmake .. && make install -j12

WORKDIR /root


