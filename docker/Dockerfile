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
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
RUN pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu129 --no-deps