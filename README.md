<p align="center">
  <img src="https://github.com/user-attachments/assets/0bb03a7f-16bb-4734-ae4b-11989dd6b96d" width = "500" class="center">
</p>
</n>

---
# vLLM-on-Intel-Extension-for-Pytorch with OpenAI Completions Template

## Introduction

This is a project to test Intel optimised Pytorch Extension (IPEX) on vLLM Model Serving Engine.
This project achieves following outcomes:
1. Test concurrency throughput on vLLM-CPU (IPEX). Inferencing is done directly on CPU.
2. To remove bias, there is no model quantization.
3. Demonstrate interoperability of opensource AI models and OpenAI stack within the same model serving framework

## Hardware Configuration

To run this demo project, we provisioned a hardware stack featuring Intel 5th Gen Xeon Processor (codename EMR) to perform the model serving
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
### Model Serving
To serve a model, simply invoke using the command `vllm serve`
```sh
vllm serve <model path>
```
We also enabled multiple environment variables in serving our demo model engine; below are all applied environment variables. For full list of allowable environment variables that can be invoked, refer directly to vLLM's CPU installation [guide](https://docs.vllm.ai/en/latest/getting_started/cpu-installation.html)
```sh
# we are maximizing our allowable memory cache for serving. with approx. 50gb reserved. this was set at 400gb
export VLLM_CPU_KVCACHE_SPACE=40
# since we are running a 2socket server setup, it is necessary to perform thread binding to prevent numa node clashing
export VLLM_CPU_OMP_THREADS_BIND=all
```



## Benchmarks
The below is a benchmark exercise to compare the performance of optimised vs. non-optimised model. The inferencing was also tested on two seperate Azure VM instance SKU (Azure Dav5 and Azure Dv6). Azure Dv6 is powered by Intel 5th Gen Xeon Processor (Emerald Rapids) whereas Azure Dav5 is powered by AMD's 3rd Gen EPYC. _Take note that concurrency is not considered here_

<img src="https://github.com/user-attachments/assets/18d30158-3018-43c7-aed8-0b4bd4726a72" width="750">

- Between optimised (INT4 + AMX) vs. non-optimised (FP32), we observed up to an average of 2.7x better performance on the optimised model.
- Between Dv6 vs. Dav5 (both utilising quantised model - INT4), we observed up to an average of 1.8x better performance on Intel Dv6 vs Dav5.

## Disclaimers / Attribution
This repository is for educational purposes and is a community contribution from repository owner. It is not intended for any commercial purposes.
Credits and attribution should be directed to repository owner and all contributors.

