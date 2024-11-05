ARG PYTHON_VERSION=3.11

#####################################################
## BUILD STAGE
#####################################################
FROM ubuntu:22.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive

# Install essential packages. For vu
ARG PYTHON_VERSION
ENV PYTHON_VERSION=${PYTHON_VERSION}
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    autoconf automake libtool pkg-config \
    software-properties-common \
    apt-transport-https ca-certificates \
    g++ git wget \
    cmake gdb valgrind \
    locales locales-all \
    gnupg && \
    apt-get install --reinstall -y ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# deadsnakes needs to be run after based installations
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-distutils && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*



# Set up a virtual environment and ensure pip is upgraded
RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip
COPY requirements.txt /opt/venv/requirements.txt
RUN pip install -r /opt/venv/requirements.txt


# Directory in docker images that stores cutagi's code
ARG WDC=/usr/src/cutagi

# Create environement variable to pass the cfg file. NOTE: We should expolore the entry point
ENV NAME=VAR1

# Copy code from the host device to docker images.
COPY src/ ${WDC}/src
COPY include/ ${WDC}/include
COPY test/ ${WDC}/test
COPY extern/ ${WDC}/extern
COPY pytagi/ ${WDC}/pytagi
COPY scripts/ ${WDC}/scripts
COPY CMakeLists.txt ${WDC}/CMakeLists.txt
COPY Dockerfile ${WDC}/Dockerfile
COPY main.cpp ${WDC}/main.cpp
COPY requirements.txt ${WDC}/requirements.txt
COPY README.md ${WDC}/README.md
COPY data/toy_example/ ${WDC}/data/toy_example
COPY data/toy_time_series/ ${WDC}/data/toy_time_series
COPY data/toy_time_series_smoother/ ${WDC}/data/toy_time_series_smoother
COPY data/UCI/ ${WDC}/data/UCI

# Work directory for the Docker image
WORKDIR ${WDC}/

# Make compile script executable and run it
RUN chmod +x scripts/compile.sh && \
    scripts/compile.sh Release

#####################################################
## RUNTIME STAGE
#####################################################
FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive

ARG PYTHON_VERSION
ENV PYTHON_VERSION=${PYTHON_VERSION}
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    curl \
    file \
    gzip \
    apt-transport-https ca-certificates software-properties-common && \
    apt-get install --reinstall -y ca-certificates gnupg && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-distutils && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Update symbolic links
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Copy Python virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ARG WDC=/usr/src/cutagi
WORKDIR ${WDC}/
COPY --from=builder  ${WDC}/build/main ./build/
COPY --from=builder  ${WDC}/build/run_tests ./build/
COPY --from=builder  ${WDC}/scripts/docker_main.sh ./
COPY --from=builder ${WDC}/data ./data

# Copy google test binary file and update LD_LIBRARY_PATH
COPY --from=builder ${WDC}/build/lib/libgtest*.so* ./build/lib/

CMD ["/bin/bash","docker_main.sh"]