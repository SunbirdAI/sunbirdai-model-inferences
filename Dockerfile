# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags
FROM runpod/base:0.7.0-focal-cuda1241

ENV RUNPOD_ENDPOINT_ID=yapuzewu3ebmzq
ENV AUDIO_CONTENT_BUCKET_NAME=sb-api-audio-content-sb-gcp-project-01

# The base image comes with many system dependencies pre-installed to help you get started quickly.
# Please refer to the base image's Dockerfile for more information before adding additional dependencies.
# IMPORTANT: The base image overrides the default huggingface cache location.


# --- Optional: System dependencies ---
COPY builder/setup.sh /setup.sh
RUN /bin/bash /setup.sh && \
    rm /setup.sh


# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN git clone https://github.com/SparkAudio/Spark-TTS
RUN python3.13 -m pip install --upgrade pip && \
    python3.13 -m pip install torch six && \
    python3.13 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# NOTE: The base image comes with multiple Python versions pre-installed.
#       It is reccommended to specify the version of Python when running your code.


# Add src files (Worker Template)
ADD src .
ADD content ./content
ADD test_input.json .

CMD python3.13 -u /handler.py
