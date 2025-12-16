FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Base OS deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl wget gnupg git pkg-config \
       build-essential cmake ninja-build \
       clang-format clang-tidy \
       libopenblas-dev libboost-all-dev libomp-dev \
       python3 python3-pip python3-venv \
       libgtest-dev \
       && rm -rf /var/lib/apt/lists/*

# Intel oneAPI MKL (for MKLConfig.cmake)
RUN set -eux; \
    mkdir -p /usr/share/keyrings; \
    curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB \
      | gpg --dearmor > /usr/share/keyrings/oneapi-archive-keyring.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
      > /etc/apt/sources.list.d/oneAPI.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends intel-oneapi-mkl-devel; \
    rm -rf /var/lib/apt/lists/*

# Python deps used by tools/* and some baselines (e.g., rabitq uses faiss; ivfpq/scann require extra pkgs; run.py uses PyYAML)
RUN python3 -m pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple \
 && python3 -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy faiss-cpu scann pyyaml

# Ensure `python` exists for scripts that invoke `python` instead of `python3`
RUN ln -s /usr/bin/python3 /usr/bin/python || true

# Environment for CMake to find MKL
ENV MKLROOT=/opt/intel/oneapi/mkl/latest \
    MKL_DIR=/opt/intel/oneapi/mkl/latest/lib/cmake/mkl \
    CMAKE_PREFIX_PATH=/opt/intel/oneapi/mkl/latest:/opt/intel/oneapi

# Workdir (the host repo will be mounted here at runtime)
WORKDIR /workspace

# Default shell
CMD ["bash"]
