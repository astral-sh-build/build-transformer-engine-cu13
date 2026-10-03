#!/bin/bash
# NCCL EP is enabled by default in Transformer Engine 2.17 and newer.
set -euxo pipefail

platform=$1
cuda_major=$2
python=/opt/python/cp310-cp310/bin/python
"${python}" -m pip install "nvidia-nccl-cu${cuda_major}>=2.30.4"
NCCL_HOME=$("${python}" -c 'import nvidia.nccl; print(next(iter(nvidia.nccl.__path__)))')
export NCCL_HOME
export CMAKE_PREFIX_PATH="${NCCL_HOME}:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${NCCL_HOME}/lib:${LIBRARY_PATH:-}"

exec bash -o pipefail build_tools/wheel_utils/build_wheels.sh \
  "${platform}" false true false false "${cuda_major}"
