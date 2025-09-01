COMMIT=${1}
COMMIT="$(echo "${COMMIT}" | tr -d '[:space:]')"
export COMMIT
NO_TAG="ghcr.io/${2}/dinov3d"
TAG="${NO_TAG}:${COMMIT}"
TAG2="${NO_TAG}:latest"
TAG="$(echo "${TAG}" | tr -d '[:space:]')"
TAG2="$(echo "${TAG2}" | tr -d '[:space:]')"

echo $TAG
docker build --no-cache --platform linux/amd64 \
    -t $TAG \
    -t $TAG2 \
    -f ./dinov3_docker/dockerfile \
    .