## Install docker amazon linux 2023 (g4dn.4xlarge) and Cuda Toolkit Setup

### Steps to install Docker
1. Update your system packages
It's good practice to update your system before installing new software to ensure all existing packages are up-to-date.

```sh
sudo yum update -y
```

2. Install the Docker package Use `yum` to install the Docker engine.

```sh
sudo yum install -y docker
```

3. Start the Docker service. Once installed, start the Docker daemon.

```sh
sudo service docker start
```

4. Enable Docker to start on boot. This ensures that Docker will start automatically whenever the system reboots.

```sh
sudo systemctl enable docker
```

5. Add your user to the docker group (optional but recommended). By default, only the root user can run Docker commands. To run Docker commands without sudo, add your user to the docker group.

```sh
sudo usermod -a -G docker ec2-user
```

6. Apply the new group permissions. For the group change to take effect, you must log out and then log back in, or run the newgrp command.

```sh
newgrp docker
```

7. Verify the installation. Run the hello-world image to confirm Docker is working correctly.

```sh
docker run hello-world
```

8. Add your user to the docker group. This command adds your current user ($USER) to the docker group using sudo, granting the required permissions.

```sh
sudo usermod -aG docker $USER
```

> Note: If you're using a different user than the default, replace `$USER` with that user's name.

9. Apply the new group membership. For the group change to take effect in your current terminal session, you can use the `newgrp` command.


### Installing cuda toolkit and drivers on amazon linux 2023 (00:1e.0 3D controller: NVIDIA Corporation TU104GL [Tesla T4] (rev a1))

To install the CUDA toolkit and drivers on Amazon Linux 2023 for a Tesla T4 GPU, you should use the package manager dnf with the NVIDIA repositories. The process involves installing the NVIDIA driver and then the CUDA toolkit. 

#### Prerequisites
Before you start, ensure your Amazon Linux 2023 instance is up-to-date and rebooted if there were any kernel updates. 

```sh
sudo dnf update -y
sudo reboot
```

1. Install DKMS and kernel headers
DKMS ensures the NVIDIA driver modules are rebuilt automatically if your kernel is updated. 

```sh
sudo dnf install -y dkms kernel-devel-$(uname -r)
```

2. Add the NVIDIA repository

You can add the NVIDIA CUDA repository to your system to get the correct drivers and toolkit. 

```sh
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo
```

3.  Install the NVIDIA driver
For Tesla GPUs, you can enable the open-dkms module and install the nvidia-open package. 

```sh
sudo dnf module enable -y nvidia-driver:open-dkms
sudo dnf install -y nvidia-open
```

4. Install the CUDA toolkit
With the driver installed, you can now install the CUDA toolkit. For the latest version: 

```sh
sudo dnf install -y cuda-toolkit
```

5. Set up the environment variables
To use the CUDA compiler, add the CUDA binary path to your PATH and LD_LIBRARY_PATH environment variables. 

```sh
echo 'export PATH=$PATH:/usr/local/cuda/bin' | sudo tee -a /etc/profile
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' | sudo tee -a /etc/profile
source /etc/profile
```

6. Reboot and verify
Reboot your instance and then verify the installation by checking the driver and CUDA versions. 

```sh
sudo reboot
```

After the reboot, connect again and run:

```sh
nvidia-smi
nvcc -V
```

> The error Found no NVIDIA driver on your system means your Docker container cannot access the GPU on your Amazon Linux 2023 instance, even though you have installed the NVIDIA drivers and CUDA toolkit. To fix this, you need to install and configure the NVIDIA Container Toolkit. 
The NVIDIA Container Toolkit is what allows Docker containers to interact with the host system's GPU and its drivers. Your previous steps correctly set up the CUDA toolkit and drivers on the host, but the Docker daemon itself needs to be configured to enable GPU access for containers. 

### How to install and configure the NVIDIA Container Toolkit

1. Add the NVIDIA Container Toolkit repository:
First, you need to add the repository that contains the toolkit packages.

```sh
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
```

2. Install the NVIDIA Container Toolkit:
Use dnf to install the toolkit and clean the cache.

```sh
sudo dnf clean expire-cache
sudo dnf install -y nvidia-container-toolkit
```

3. Configure Docker to use the NVIDIA Container Runtime:
This command modifies your Docker daemon's configuration file (`/etc/docker/daemon.json`) so that it is aware of the NVIDIA runtime.


```sh
sudo nvidia-ctk runtime configure --runtime=docker
```

4. Restart the Docker daemon:
Restart Docker to apply the new configuration.

```sh
sudo systemctl restart docker
```

5. Verify GPU access
To confirm that your GPU is now accessible within Docker, you can run a simple CUDA container provided by NVIDIA. 

```sh
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If the setup is successful, this command will print out the NVIDIA System Management Interface (nvidia-smi) output, confirming the container can see and use the GPU. 


6. Run your container with GPU support:
Now, run your container again, this time including the `--gpus all` flag. This flag tells Docker to provide access to all available GPUs inside the container.

```sh
docker run --rm --gpus all -p 8088:8088 sunbirddocker/sunbirdai-model-inferences:v2.3.11
```

> To pass the exported environment variables to your Docker container, you can use the -e flag for each variable. Since you have already exported the variables in your shell, you can simply use the variable name without a value after the -e flag. 
Here is the updated command:

```sh
docker run --rm --gpus all -p 8088:8088 \
    -e HF_TOKEN \
    -e RUNPOD_ENDPOINT_ID \
    -e AUDIO_CONTENT_BUCKET_NAME \
    -e GCP_CREDENTIALS \
    sunbirddocker/sunbirdai-model-inferences:v2.3.11
```

#### Breakdown of the command

- `docker run`: The main command to create and run a container.
- `--rm`: Automatically removes the container when it exits.
- `--gpus all`: Passes all available GPUs from the host to the container.
- `-p 8088:8088`: Maps port 8088 on the host to port 8088 in the container.
- `-e <VARIABLE_NAME>`: This is the key part for passing your environment variables. Since `HF_TOKEN`, `RUNPOD_ENDPOINT_ID`, `AUDIO_CONTENT_BUCKET_NAME`, and `GCP_CREDENTIALS` are already exported in your shell, Docker will automatically pick up their values and pass them into the container.
- `sunbirddocker/sunbirdai-model-inferences:v2.3.11`: The name and tag of the Docker image to run. 

#### Verification
If you want to confirm that the environment variables were passed correctly, you could add an echo command to your container startup or run docker inspect after it's running. For instance, to inspect a container named my-container, you would run: 

```sh
docker inspect -f '{{.Config.Env}}' my-container
```