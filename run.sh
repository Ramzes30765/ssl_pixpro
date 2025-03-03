#!/bin/bash

docker build -t clearml_agent_image .

# clearml-agent daemon --docker clearml_agent_image:dev0 --queue pixpro_queue --gpus all
# docker run -it --rm clearml_agent_image:v1 python /app/pipelines/train_pipeline.py
# export PYTHONPATH=$PYTHONPATH:/home/kitt/ssl_pixpro