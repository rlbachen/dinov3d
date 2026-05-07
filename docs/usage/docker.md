# Docker

To make the training of the model platform-agnostic, we wrap the code in a docker
image.

To run the docker image use the following command:
```
docker run --rm \
  -v /path/to/dataset:/dataset \
  -v /path/to/output:/output \
  -v /path/to/.aws:/root/.aws:ro \ # optional for s3 dataset download
  v /path/to/config:/config
  --gpus all \
  --shm-size=16g \
  ghcr.io/asagilmore/dinov3d /app/scripts/main.sh \
  -c /app/scripts/configs/aws_p5_config.yaml \ # can also be set to /config/your_config.yaml if you mounted a config folder
  -g 8 \ # num gpus
  -o /output
  -d # pass to skip download
```

The entire repo is located inside /app in the container, so any scripts there can be easily
accessed for inference.

More details on inference using docker to come XXX.
