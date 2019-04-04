#!/usr/bin/env bash

REPOSITORY=$1
STALE_MACHINES=$(aws ecr describe-images --repository-name ${REPOSITORY} --query 'reverse(sort_by(imageDetails, &imagePushedAt)[?imageTags!=`string`].[imageDigest])' --output text | sed 's/sha256/imageDigest=sha256/' | sed -e '1,5d')
if [[ $STALE_MACHINES == *"imageDigest=sha256"* ]]; then
	echo "Removing the following images from ECR:" ${STALE_MACHINES}
	aws ecr batch-delete-image --repository-name ${REPOSITORY} --image-ids $STALE_MACHINES
fi
