#!/bin/bash
#
# Build the docker images.

set -euo pipefail

source scripts/shared.sh

parse_std_args "$@"

# Dockfile.ei needs to login 520713654638 to pull the cpu image
if [ "$arch" = "ei" ]; then
    echo "pulling cpu image..."
    $(aws ecr get-login --no-include-email --registry-id 520713654638) &>/dev/null || echo 'warning: ecr login failed'
else
    echo "pulling previous image for layer cache... "
    $(aws ecr get-login --no-include-email --registry-id $aws_account) &>/dev/null || echo 'warning: ecr login failed'
    docker pull $aws_account.dkr.ecr.$aws_region.amazonaws.com/sagemaker-tensorflow-serving:$full_version-$arch &>/dev/null || echo 'warning: pull failed'
    docker logout https://$aws_account.dkr.ecr.$aws_region.amazonaws.com &>/dev/null
fi

echo "building image... "
docker build \
    --cache-from $aws_account.dkr.ecr.$aws_region.amazonaws.com/sagemaker-tensorflow-serving:$full_version-$arch \
    --build-arg TFS_VERSION=$full_version \
    --build-arg TFS_SHORT_VERSION=$short_version \
    --build-arg TENSORFLOW_MODEL_SERVER=$model \
    -f docker/Dockerfile.$arch \
    -t sagemaker-tensorflow-serving:$full_version-$arch \
    -t sagemaker-tensorflow-serving:$short_version-$arch container

# logout 520713654638 account after building
if [ "$arch" = "ei" ]; then
    docker logout https://520713654638.dkr.ecr.$aws_region.amazonaws.com &>/dev/null
fi