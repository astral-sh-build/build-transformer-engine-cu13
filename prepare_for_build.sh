#!/bin/bash
# Script to prepare the build environment for Transformer Engine (CUDA 13 core).
#
# Example usage:
#   ./prepare_for_build.sh v2.16

set -euxo pipefail

export ROOT=`pwd`

if [ $# -ne 1 ]; then
    echo "Usage: $0 <transformer_engine_version>"
    echo "Example: $0 v2.16"
    exit 1
fi

TRANSFORMER_ENGINE_VERSION=$1

# Apply the CUDA core compatibility patch for the selected upstream release.
patch_dir="${ROOT}/build_scripts/patches/${TRANSFORMER_ENGINE_VERSION}"

if [ ! -d "${patch_dir}" ]; then
    echo "Error: no compatibility patch for ${TRANSFORMER_ENGINE_VERSION}" >&2
    exit 1
fi

for patch in "${patch_dir}"/*.patch; do
    if [ -f "${patch}" ]; then
        patch -p1 -d "${ROOT}" -i "${patch}"
    fi
done
