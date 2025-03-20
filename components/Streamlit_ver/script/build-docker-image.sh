#!/bin/bash -e
PRJ_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
cd ${PRJ_DIR}

export DOCKER_BUILDKIT=1

# CHANGE THIS {DOCKER_IMAGE_NAME}.
DOCKER_IMAGE_NAME=$(bash script/get-image-name.sh)

# LICENSE_MODE 支援: ['dongle', 'cloud', 'date', 'bin', 'src', 'hw-date'], 預設 src
LICENSE_MODE="${LICENSE_MODE:-src}"
LICENSE_TICK_SEC="${LICENSE_TICK_SEC:-30}"
LICENSE_MAX_RETRIES="${LICENSE_MAX_RETRIES:-5}"

echo "Building $DOCKER_IMAGE_NAME...)"

docker build \
    --network=host \
    --build-arg LICENSE_MODE=${LICENSE_MODE} \
    --build-arg LICENSE_MAX_RETRIES=${LICENSE_MAX_RETRIES} \
    --build-arg LICENSE_TICK_SEC=${LICENSE_TICK_SEC} \
    --force-rm \
    --tag $DOCKER_IMAGE_NAME \
    -f Dockerfile \
    .
