NO_TAG="ghcr.io/${1}/dinov3d"
NO_TAG="$(echo "${NO_TAG}" | tr -d '[:space:]')"

docker push --all-tags $NO_TAG