#!/bin/bash
#
# Publish all images to your ECR account.

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

${DIR}/publish.sh --version 1.11.1 --arch cpu
${DIR}/publish.sh --version 1.11.1 --arch gpu
${DIR}/publish.sh --version 1.12.0 --arch cpu
${DIR}/publish.sh --version 1.12.0 --arch gpu
${DIR}/publish.sh --version 1.13.0 --arch cpu
${DIR}/publish.sh --version 1.13.0 --arch gpu
${DIR}/publish.sh --version 1.11 --arch eia
${DIR}/publish.sh --version 1.12 --arch eia
