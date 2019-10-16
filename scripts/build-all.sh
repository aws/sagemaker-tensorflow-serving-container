#!/bin/bash
#
# Build all the docker images.

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

${DIR}/build.sh --version 1.11.1 --arch cpu
${DIR}/build.sh --version 1.11.1 --arch gpu
${DIR}/build.sh --version 1.12.0 --arch cpu
${DIR}/build.sh --version 1.12.0 --arch gpu
${DIR}/build.sh --version 1.13.0 --arch cpu
${DIR}/build.sh --version 1.13.0 --arch gpu
${DIR}/build.sh --version 1.14.0 --arch cpu
${DIR}/build.sh --version 1.14.0 --arch gpu
${DIR}/build.sh --version 1.11 --arch eia
${DIR}/build.sh --version 1.12 --arch eia
${DIR}/build.sh --version 1.13 --arch eia
${DIR}/build.sh --version 1.14 --arch eia
