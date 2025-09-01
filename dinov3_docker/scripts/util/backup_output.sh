#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="/path/to/local/dir"
DEST_S3="s3://your-backup-bucket/path"
SLEEP_SECONDS=300  # 5 minutes

while getopts "s:d:t:" opt; do
  case "$opt" in
    s)
      SRC_DIR="$OPTARG"
      ;;
    d)
      DEST_S3="$OPTARG"
      ;;
    t)
      SLEEP_SECONDS="$OPTARG"
      ;;
    \?)
      echo "Usage: $0 [-s /source/dir] [-d s3://destination/s3/path] [-t 300 ( backup interval in seconds )] " >&2
      exit 1
      ;;
  esac
done

log() {
  echo "$(date -Is) [s3-sync] $*"
}

while true; do
  log "Starting aws s3 sync: ${SRC_DIR} -> ${DEST_S3}"
  if aws s3 sync "${SRC_DIR}" "${DEST_S3}" --delete; then
    log "Sync completed successfully"
  else
    log "Sync FAILED, exiting with error"
    exit 1
  fi

  log "Sleeping for ${SLEEP_SECONDS}s"
  sleep "${SLEEP_SECONDS}"
done
