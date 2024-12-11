#!/bin/bash

set -e # Stop script on error

# Update system
apt-get update && apt-get upgrade -y

# Install system dependencies
apt-get install -y --no-install-recommends software-properties-common curl git openssh-server

# Install Python 3.11 with retry logic for PPA
for i in {1..3}; do
    if add-apt-repository ppa:deadsnakes/ppa -y; then
        break
    elif [ "$i" -eq 3 ]; then
        echo "Failed to add PPA after multiple attempts."
        exit 1
    fi
    echo "Retrying to add PPA ($i/3)..."
    sleep 5
done

apt-get update && apt-get install -y --no-install-recommends python3.11 python3.11-dev python3.11-distutils
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install additional dependencies
apt-get install -y build-essential cmake libboost-system-dev libboost-thread-dev \
    libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev \
    liblzma-dev ffmpeg

# Install pip for Python 3.11
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py

# Clean up to reduce image size
apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*
