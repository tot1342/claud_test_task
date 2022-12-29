FROM ubuntu:20.04

RUN apt-get update -y
RUN apt-get install python3-pip -y
RUN apt-get install git -y
RUN git clone https://github.com/pytorch/serve.git
RUN DEBIAN_FRONTEND=noninteractive python3 ./serve/ts_scripts/install_dependencies.py
RUN pip3 install flask requests matplotlib opencv-python

RUN pip3 install torchserve torch-model-archiver torch-workflow-archiver
COPY app.py ./
RUN mkdir templates
COPY templates/* ./templates/
COPY fasterrcnn.pt ./
COPY model.py ./

RUN torch-model-archiver --model-name fasterrcnn --version 1.0 --model-file ./model.py --serialized-file ./fasterrcnn.pt --extra-files ./serve/examples/object_detector/index_to_name.json --handler model
RUN mkdir model_store
RUN mv fasterrcnn.mar model_store

CMD torchserve --start --ncs --model-store model_store --models fasterrcnn=fasterrcnn.mar & python3 -m flask run --host=0.0.0.0 --port 8000
