### Supported Ubuntu Version 18.04, 20.04, 22.04
ARG UBUNTU_VERSION
# NOSONAR: Running as root is acceptable for this build container as it's used for compilation only
FROM ubuntu:${UBUNTU_VERSION}

### Set TimeZone
ENV TZ Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ >/etc/timezone

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/bin:${PATH}"

RUN apt-get update && apt-get install -y sudo software-properties-common && \
add-apt-repository universe && \
add-apt-repository multiverse && \
apt-get update

RUN apt-get install -y x11-apps libx11-6 xauth libxext6 libxrender1 libxtst6 libxi6

RUN apt-get install -y gpg-agent tar
RUN add-apt-repository ppa:deadsnakes/ppa && \
apt-get update && apt-get install -y python3 python3-dev libpython3-dev

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /app
COPY dx_rt.tar.gz /app/dx_rt.tar.gz
COPY dx_app.tar.gz /app/dx_app.tar.gz
COPY dx_stream.tar.gz /app/dx_stream.tar.gz

RUN apt-get update && apt-get install -y python3-venv
RUN python3 -m venv /app/venv
RUN /app/venv/bin/pip install --upgrade pip

ENV PATH="/app/venv/bin:$PATH"

RUN mkdir -p dx_rt dx_app
RUN tar -xzvf ./dx_rt.tar.gz -C ./dx_rt && ls -al && cd dx_rt && ./install.sh --dep && ./build.sh --clean && cd python_package && python -m pip install .
RUN tar -xzvf ./dx_app.tar.gz -C ./dx_app && cd dx_app && ./install.sh --all && ./build.sh --clean
RUN tar -xzvf ./dx_stream.tar.gz -C ./ && cd dx_stream && ./install.sh && ./build.sh --install 

ENTRYPOINT [ "/usr/local/bin/dxrtd" ]
CMD ["/bin/bash"]