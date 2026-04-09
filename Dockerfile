##############################################################################
# Dockerfile — Quant Alpha Research Environment
# Python 3.13 + C++26 (Clang 18) + Bazel
##############################################################################
FROM ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    clang-18 \
    lld-18 \
    libeigen3-dev \
    lcov \
    git \
    curl \
    wget \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python 3.13
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.13 python3.13-dev python3.13-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.13 default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

# ---------------------------------------------------------------------------
# Bazel via Bazelisk
# ---------------------------------------------------------------------------
RUN curl -Lo /usr/local/bin/bazel \
    https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64 \
    && chmod +x /usr/local/bin/bazel

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
WORKDIR /workspace
COPY pyproject.toml ./
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -e ".[dev]"

# ---------------------------------------------------------------------------
# Copy source
# ---------------------------------------------------------------------------
COPY . .

# C++ build
RUN cmake -B build -G Ninja \
    -DCMAKE_CXX_COMPILER=clang++-18 \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel 4

# Default: run all tests
CMD ["bash", "-c", "pytest src/python/tests/ -v && cd build && ctest --output-on-failure"]
