docker run --rm \
        --gpus=all \
        -v ./input:/input \
        -v ./output:/output \
        trackrad-algorithm
