FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND="noninteractive"

# Install Python libraries
COPY requirements.txt requirements.txt
RUN apt update  \
    && apt install -y unzip python3-pip python-setuptools sudo openjdk-21-jdk openjdk-21-jre \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && rm -f /usr/lib/libxgboost.so

# Install FlagEmbedding
RUN mkdir /tmp/stage && \
    git clone https://github.com/FlagOpen/FlagEmbedding.git /tmp/stage/FlagEmbedding && \
    cd /tmp/stage/FlagEmbedding && \
    pip install . && \
    cd - && \
    rm -rf /tmp/stage