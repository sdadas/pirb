FROM nvcr.io/nvidia/pytorch:23.04-py3

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND="noninteractive"

# Install Python libraries
COPY requirements.txt requirements.txt
RUN apt update  \
    && apt install -y unzip python3-pip python-setuptools sudo default-jre \
    && pip install --upgrade pip \
    && pip install -r requirements.txt