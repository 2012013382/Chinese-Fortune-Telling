ARG ALLUXIO_VERSION=2.9.4
FROM harbor.shopeemobile.com/aip/dataset-cache/alluxio-fuse:${ALLUXIO_VERSION} AS alluxio_fuse_archive
 
# From NGC container
FROM nvcr.io/nvidia/pytorch:23.05-py3
 
# Create a work user.
RUN apt-get update && apt-get install -y sudo
RUN adduser --gecos '' --disabled-password work && \
    echo "work ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd
    
# VScode
ENV INSTALL_DIR=/usr/lib/vscode-server
ENV EXTENSION_DIR=/home/work/.vscode-server/extensions
ENV DATA_DIR=/home/work/.vscode-server/data
ENV ENTRYPOINT="$INSTALL_DIR/bin/vscode-server --host 0.0.0.0 --port 8080 --without-connection-token --extensions-dir=$EXTENSION_DIR --data-dir=$DATA_DIR"
EXPOSE 8080
ENV PATH=$INSTALL_DIR/bin:${PATH}

# Paths Configurations
ENV PATH=${PATH}:/home/work/.local/bin
RUN export PATH
RUN chown -R work:work /usr/local/*
WORKDIR /home/work
RUN git config --global http.proxy aiproxy.shopee.io:80
