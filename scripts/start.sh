#!/bin/bash
#
# Start a local docker container.

set -euo pipefail

source scripts/shared.sh

parse_std_args "$@"

if [ "$arch" == 'gpu' ]; then
    docker_command='nvidia-docker'
else
    docker_command='docker'
fi


MODEL_DIR="$(cd "test/resources/mme" > /dev/null && pwd)"
$docker_command run \
    -v "$MODEL_DIR":/opt/ml/model:ro \
    -p 8080:8080 \
    -e "SAGEMAKER_TFS_NGINX_LOGLEVEL=error" \
    -e "SAGEMAKER_BIND_TO_PORT=8080" \
    -e "SAGEMAKER_MULTI_MODEL=True" \
    -e "SAGEMAKER_SAFE_PORT_RANGE=9000-9999" \
    sagemaker-tensorflow-serving:1.15.0-cpu serve > log.txt 2>&1 &
