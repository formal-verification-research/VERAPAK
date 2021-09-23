FROM tensorflow/tensorflow:2.5.0

RUN apt-get update && apt-get install -y git wget && pip install -U pip

RUN git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow && pip install -e .

RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.gz && tar -xvzf boost_1_77_0.tar.gz && cd boost_1_77_0 && ./bootstrap.sh --with-python=python3.6 --with-libraries=python,system && ./b2 install

RUN apt-get install -y cmake

ADD . /src

RUN mkdir /src/_build && cd /src/_build && cmake .. && make install -j12

WORKDIR /src


