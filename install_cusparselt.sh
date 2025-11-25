#!/bin/bash

set -ex

# cuSPARSELt license: https://docs.nvidia.com/cuda/cusparselt/license.html
mktmp() {
    mkdir -p tmp_cusparselt
    cd tmp_cusparselt
}

parse_nv_tegra_release() {
    local release_file="/etc/nv_tegra_release"
    if [ -f "$release_file" ]; then
        if grep -q "R36" "$release_file"; then
            echo "12.1"
            return
        fi
        if grep -Eq "R35|JetPack 5\\.0" "$release_file"; then
            echo "12.0"
            return
        fi
    fi
}

CUDA_VERSION=${CUDA_VERSION:-$(parse_nv_tegra_release)}

mktmp

if [[ ${CUDA_VERSION:0:4} =~ ^12\.[1-4]$ ]]; then
    arch_path='sbsa'
    export TARGETARCH=${TARGETARCH:-$(uname -m)}
    if [ ${TARGETARCH} = 'amd64' ] || [ "${TARGETARCH}" = 'x86_64' ]; then
        arch_path='x86_64'
    fi
    CUSPARSELT_NAME="libcusparse_lt-linux-${arch_path}-0.5.2.1-archive"
    CUSPARSELT_URL="https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-${arch_path}/${CUSPARSELT_NAME}.tar.xz"
elif [[ ${CUDA_VERSION:0:4} == "11.8" ]]; then
    CUSPARSELT_NAME="libcusparse_lt-linux-x86_64-0.4.0.7-archive"
    CUSPARSELT_URL="https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/${CUSPARSELT_NAME}.tar.xz"
fi

if [ -n "${CUSPARSELT_URL}" ]; then
    curl --retry 3 -OLs "${CUSPARSELT_URL}"
fi

if [ -z "${CUSPARSELT_NAME}" ]; then
    echo "Unable to determine cuSPARSELt version from CUDA_VERSION=${CUDA_VERSION}. Please set CUDA_VERSION or install manually."
    exit 1
fi

if [ ! -f "${CUSPARSELT_NAME}.tar.xz" ]; then
    # Attempt to reuse bundled archive if present in repo root
    repo_archive="$(dirname "$0")/libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz"
    if [ ! -f "${CUSPARSELT_NAME}.tar.xz" ] && [ -f "${repo_archive}" ]; then
        cp "${repo_archive}" .
    fi
fi

tar xf ${CUSPARSELT_NAME}.tar.xz
cp -a ${CUSPARSELT_NAME}/include/* /usr/local/cuda/include/
cp -a ${CUSPARSELT_NAME}/lib/* /usr/local/cuda/lib64/
cd ..
rm -rf tmp_cusparselt
ldconfig
