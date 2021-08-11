FROM tensorflow/tensorflow:latest

RUN apt-get update && apt-get install -y git && pip install -U pip

RUN git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow && pip install -e .

ADD . /src

WORKDIR /src


