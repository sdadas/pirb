FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND="noninteractive"

# Install Python libraries
COPY requirements_flag.txt requirements.txt
RUN apt update  \
    && apt install -y unzip python3-pip python-setuptools sudo openjdk-21-jdk openjdk-21-jre \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && rm -f /usr/lib/libxgboost.so

# Install XFormers
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"
RUN pip install -v -U git+https://github.com/facebookresearch/xformers.git@v0.0.28.post2#egg=xformers