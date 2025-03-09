PRJ_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
cd ${PRJ_DIR}

DOCKER_IMAGE_NAME="stephC/llm-rag-uniapply"
VERSION=`grep -Po "(?<=['\"])[0-9a-zA-Z\.\-]+" app/version.py`
LICENSE_MODE="${LICENSE_MODE:-src}"

# LICENSE_MODE 支援: ['dongle', 'cloud', 'date', 'bin', 'src'], 預設 src
repo_and_tag="${DOCKER_IMAGE_NAME}:${VERSION}-${LICENSE_MODE}"

echo "${repo_and_tag}"
