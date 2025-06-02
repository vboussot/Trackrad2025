#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Building container"

docker build $SCRIPT_DIR \
  --quiet \
  --platform=linux/amd64 \
  --tag "trackrad-algorithm" 2>&1

output_filename="$SCRIPT_DIR/submission_$(date +"%Y_%m_%dT%H_%M_%S").tar.gz"

echo "Saving container to $output_filename"
echo "This may take a while"

# Save the Docker container and gzip
docker save trackrad-algorithm | gzip -c > "$output_filename"

echo "Container saved"
