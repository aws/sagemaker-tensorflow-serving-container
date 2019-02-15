#!/bin/bash
#
# Build the docker images.

set -euo pipefail

source scripts/shared.sh

parse_std_args "$@"

echo "pulling previous image for layer cache... "
$(aws ecr get-login --no-include-email --registry-id $aws_account) &>/dev/null || echo 'warning: ecr login failed'
docker pull $aws_account.dkr.ecr.$aws_region.amazonaws.com/sagemaker-tensorflow-serving:$full_version-$arch &>/dev/null || echo 'warning: pull failed'
docker logout https://$aws_account.dkr.ecr.$aws_region.amazonaws.com &>/dev/null

echo "building image... "
docker build \
    --cache-from $aws_account.dkr.ecr.$aws_region.amazonaws.com/sagemaker-tensorflow-serving:$full_version-$arch \
    --build-arg TFS_VERSION=$full_version \
    --build-arg TFS_SHORT_VERSION=$short_version \
    -f docker/Dockerfile.$arch \
    -t sagemaker-tensorflow-serving:$full_version-$arch \
    -t sagemaker-tensorflow-serving:$short_version-$arch container
