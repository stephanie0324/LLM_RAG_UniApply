PRJ_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
cd ${PRJ_DIR}

# export $(grep -v '^#' ../../deploy/.env | xargs)
set -a  # 讓 source 自動 export 變數
source ../../deploy/.env
set +a  # 停止自動 export

REQUESTS_CA_BUNDLE=${REQUESTS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}
SSL_CERT_FILE=${REQUESTS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}

IMAGE_NAME=$(bash script/get-image-name.sh)

docker run \
    --name ${COMPOSE_PROJECT_NAME}-backend-1 \
    -p ${HOST_PORT}:7860 \
    -v ${PRJ_DIR}/app:/app \
    --rm \
    -it \
    --entrypoint bash \
    -e REQUESTS_CA_BUNDLE=${REQUESTS_CA_BUNDLE} \
    -e SSL_CERT_FILE=${SSL_CERT_FILE} \
    -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
    -v /etc/ssl/certs:/etc/ssl/certs \
    --gpus all \
    $IMAGE_NAME

