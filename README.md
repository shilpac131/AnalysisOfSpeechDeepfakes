# AnalysisOfSpeechDeepfakes


This repository provides the two frameworks for generating spoofed speech and then detecting the same via docker containers.

The spoofed speech generation is available in english and hindi language and is obtained via the MahaTTS developed by [Dubverse.ai](https://dubverse.ai)

The spoofed speech detection is done via the audio anti-spoofing systems proposed in ['AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks'](https://arxiv.org/abs/2110.01200)



## Getting started

### Pre-requisites
[Docker](https://www.docker.com/) need to be installed on your system for both the frameworks. Docker is compatable with Linux, Mac and Windows. Please follow the steps to install docker on your system:> [Link](https://docs.docker.com/engine/install/)

Along with Docker, NVIDIA Container Toolkit needs to be installed for enabling the access of cuda in you machine with the container. The steps to install the same can be found here:> [Link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### I. Generating spoofed speech

The details steps to generate English and Hindi speech can be found here (Note: Having cuda enabled machine is highly appreciated):
  1. [Hindi Text-to-Speech Inference with Docker](https://github.com/Shiven-Patel-IIT/MahaTTS-Hindi-inference-using-docker)
  2. [English Text-to-Speech Inference with Docker](https://github.com/Shiven-Patel-IIT/Maha-TTS-inference-using-docker)

Both the repository are using [MahaTTS](https://github.com/dubverse-ai/MahaTTS/)

[Shiven Patel](https://github.com/Shiven-Patel-IIT) has created these GitHub repositories to help containerized using Docker to handle all dependencies seamlessly.

### II. Detecting spoofed speech using AASIST

The steps to run the spoofed speech detecting algorithm ([AASIST](https://arxiv.org/abs/2110.01200)):
  1. First, clone the repository locally

```bash
git clone https://github.com/shilpac131/AnalysisOfSpeechDeepfakes
cd AnalysisOfSpeechDeepfakes
```

**⚠️ Note:** Your audio file/files(s) (to be classified as fake or real) should be present in the AnalysisOfSpeechDeepfakes/audio_files/ directory before running the steps below as the docker doesn't recongnize paths from system so before buliding the docker copy the audio files.

  2. Build the Docker Image
  A Dockerfile defines the instructions to build a Docker image with all dependencies and configurations for an application. To build the docker image run

  ```bash
git build -t <image_name> .
```

  3. Run the following commnd to run the detection algorithm. It will prompt user with 2 options, to detect spoofed speech via single audio or via a folder conatining multiple audio. Choose accordingly. And give the path to the AnalysisOfSpeechDeepfakes/audio_files/ when prompted with the path to audio file/file(s)

```bash
docker run --gpus all -it -v ./mydockerdata:/app/data <image_name>
```
  The output scores will be available in the ./mydockerdata/output.txt file.

  

