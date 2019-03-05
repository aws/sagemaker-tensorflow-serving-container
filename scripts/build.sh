#!/bin/bash
#
# Build the docker images.

set -euo pipefail

source scripts/shared.sh

parse_std_args "$@"

if [ $arch = 'eia' ]; then
    get_tfs_executable
fi

echo "pulling previous image for layer cache... "
$(aws ecr get-login --no-include-email --registry-id $aws_account) &>/dev/null || echo 'warning: ecr login failed'
docker pull $aws_account.dkr.ecr.$aws_region.amazonaws.com/$repository:$full_version-$device &>/dev/null || echo 'warning: pull failed'
docker logout https://$aws_account.dkr.ecr.$aws_region.amazonaws.com &>/dev/null

echo "building image... "
docker build \
    --cache-from $aws_account.dkr.ecr.$aws_region.amazonaws.com/$repository:$full_version-$device \
    --build-arg TFS_VERSION=$full_version \
    --build-arg TFS_SHORT_VERSION=$short_version \
    -f docker/Dockerfile.$arch \
    -t $repository:$full_version-$device \
    -t $repository:$short_version-$device container
