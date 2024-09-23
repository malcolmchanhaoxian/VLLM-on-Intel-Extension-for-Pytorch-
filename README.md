<p align="center">
  <img src="https://github.com/user-attachments/assets/0bb03a7f-16bb-4734-ae4b-11989dd6b96d" width = "500">
</p>
</n>

---
# vLLM on Intel Extension for Pytorch with OpenAI Completions Template

## Introduction

This is a project to test Intel optimised Pytorch Extension (IPEX) on vLLM Model Serving Engine.
This project achieves following outcomes:
1. Test concurrency throughput on vLLM-CPU (IPEX). Inferencing is done directly on CPU.
2. To remove bias, there is no model quantization.
3. Demonstrate interoperability of opensource AI models and OpenAI stack within the same model serving framework
<br>
By following the below framework, we are operating a model serving engine as back-end. And we will be running a client frontend and the OpenAI completions template to access the model inferencing and chat streaming.
<p align="center">
<img src = "https://github.com/user-attachments/assets/b6d5f97e-61a1-4deb-b930-6320b6cc1a66" width = "700">
</p>


## Hardware Configuration

To run this demo , we provisioned a hardware stack featuring Intel 5th Gen Xeon Processor (codename EMR) to perform the model serving:<br>
<br>
**Dell PowerEdge R760**
- Processor: Xeon Platinum 8580
- Processor Configuration: 2S - 60c each // Total 240 logical cores
- Memory: 512GB

## Part 1 Model Serving Engine
### Installation
#### (1) vLLM installation
Follow the installation instruction from vLLM provided [link](https://docs.vllm.ai/en/latest/getting_started/cpu-installation.html).<br>
I recommend building direct from source and ensure the use of latest updates from vLLM.
<br>
#### (2) IPEX installation
It is recommended to download directly from Intel's github source page and follow the instructions [here](https://intel.github.io/intel-extension-for-pytorch/#installation) to build the installation package based on your requirements.<br><br>
For this demonstration, we used `pip` to perform IPEX installation using the latest version v2.4.0+cpu

```sh
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch
python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
```
#### Important
It is essential that you use TCMalloc for cache locality. To install, run the following:
```sh
sudo apt-get install libtcmalloc-minimal4 # install TCMalloc library
find / -name *libtcmalloc* # find the dynamic link library path
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD # prepend the library to LD_PRELOAD
```

### Model Serving
To serve a model, simply invoke using the command `vllm serve`
```sh
vllm serve <model path>
```
We also enabled multiple environment variables in serving our demo model engine; below are all applied environment variables. For full list of allowable environment variables that can be invoked, refer directly to vLLM's CPU installation [guide](https://docs.vllm.ai/en/latest/getting_started/cpu-installation.html)<br>

- We have enabled chunk prefill to chunk large prefills into smaller chunks and batch them together with decode requests.
- dtype is set as bfloat16. GPU-based inferencing generally use dtype-half but we enable bfloat16 which allow us to leverage on Intel Xeon Processors' **AI boost feature - AMX**
- The model we are using is [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) 

```sh
# we are maximizing our allowable memory cache for serving with approx. 50gb reserved. This was set at 400gb
export VLLM_CPU_KVCACHE_SPACE=400
# since we are running a 2socket server setup, it is necessary to perform thread binding to prevent numa node clashing
export VLLM_CPU_OMP_THREADS_BIND=all

vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --enable-chunked-prefill --dtype bfloat16 --host <host ip> --port <port number>
```

## Part 2 Client Front-end
We are using the base template provided by vLLM to build our front-end python script. The original code can be found [here](https://github.com/vllm-project/vllm/blob/main/examples/gradio_openai_chatbot_webserver.py).<br>
The version of OpenAI client chat script can be found in this repo and is designed to print the tokenisation output.<br> 
```sh
!python app.py
```
You can access the Gradio frontend on your local host, any public ip or on the temporary url provided by Gradio
<p align="center">
<img src = "https://github.com/user-attachments/assets/6bcc7548-151c-4305-b5fb-137b7fb59987" width = "700">
</p>

#### Results
Our preliminary tests shows an output token throughput of 10 - 12 tokens per second on average.

## Benchmarks
The below is a benchmark exercise to compare the ability of the model serving engine to handle concurrency while **only using CPU as inferencing device**. <br>
In order to execute this, we utilise [Locust](https://locust.io/) and swarm the model serving engine with concurrent users.<br>
We set a specific prompt, max output tokens and temperature paramaters to perform the swarm test. These can be modified in the [swarm](https://github.com/malcolmchanhaoxian/VLLM-on-Intel-Extension-for-Pytorch-/blob/main/locustfile.py) file:
```sh
"model": <model path>,
"prompt": "Tell me Intel history",
"max_tokens": 128,
"temperature": 0.9
```
Our final swarm tests shows that our CPU only model engine was able to sustain up to 512 users concurrently before failure.
Details available in html results file.
<p align="center">
<img src="https://github.com/user-attachments/assets/947aee74-7d74-4afc-839a-5fd0cc3a6f02" width="750">
</p>

#### Results
- Running demanding GPU workload on  CPU is very possible (e.g. AI inferencing on model size up to 13B is a possibility)
- By using the OpenAI completion templates - you can operate harmoniously with Azure’s OpenAI service with Intel’s software stack that is integrable with existing codebase and services.

## Disclaimers / Attribution
This repository is for educational purposes and is a community contribution from repository owner. It is not intended for any commercial purposes.
Credits and attribution should be directed to repository owner and all contributors. This repository is not owned by Intel or its subsidiary.
The performance numbers reported here are indicative and not approved legally by Intel.

