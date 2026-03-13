FROM nvcr.io/nvidia/pytorch:25.06-py3

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND="noninteractive"

# Install Python libraries
COPY requirements.txt requirements.txt
RUN apt update \
    && apt install -y unzip python3-pip python3-setuptools sudo openjdk-21-jdk openjdk-21-jre \
    && pip install -r requirements.txt

# Install XFormers
RUN pip install xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128 --no-deps