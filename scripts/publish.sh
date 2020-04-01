#!/bin/bash
#
# Publish images to your ECR account.

set -euo pipefail

source scripts/shared.sh

parse_std_args "$@"

aws ecr get-login-password --region ${aws_region} \
    | docker login \
        --password-stdin \
        --username AWS \
        "${aws_account}.dkr.ecr.${aws_region}.amazonaws.com/${repository}"
docker tag $repository:$full_version-$device $aws_account.dkr.ecr.$aws_region.amazonaws.com/$repository:$full_version-$device
docker tag $repository:$full_version-$device $aws_account.dkr.ecr.$aws_region.amazonaws.com/$repository:$short_version-$device
docker push $aws_account.dkr.ecr.$aws_region.amazonaws.com/$repository:$full_version-$device
docker push $aws_account.dkr.ecr.$aws_region.amazonaws.com/$repository:$short_version-$device
docker logout https://$aws_account.dkr.ecr.$aws_region.amazonaws.com
