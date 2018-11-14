#!/bin/bash
#
# Build the docker images.

set -euo pipefail

source scripts/shared.sh

parse_std_args "$@"

docker build -f docker/$major_version/Dockerfile.$arch \
    -t sagemaker-tensorflow-serving:$minor_version-$arch \
    -t sagemaker-tensorflow-serving:$major_version-$arch container
