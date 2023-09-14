# PPL LLM Kernel CUDA

## Overview

`ppl.llm.kernel.cuda` is a part of `PPL.LLM` system.

![SYSTEM_OVERVIEW](https://github.com/openppl-public/ppl.nn.llm/blob/master/docs/system_overview.png)

**We recommend users who are new to this project to read the [Overview of system](https://github.com/openppl-public/ppl.nn.llm/blob/master/docs/system_overview.md).**

---

Primitive cuda kernel library for [ppl.nn.llm](https://github.com/openppl-public/ppl.nn.llm)

Currently, only Ampere and Hopper have been tested.

## Prerequisites

* Linux running on x86_64 or arm64 CPUs
* GCC >= 9.4.0
* [CMake](https://cmake.org/download/) >= 3.18
* [Git](https://git-scm.com/downloads) >= 2.7.0
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) >= 11.4. 11.6 recommended. (for CUDA)

## Quick Start

* Installing Prerequisites(on Debian or Ubuntu for example)

    ```bash
    apt-get install build-essential cmake git
    ```

* Cloning Source Code

    ```bash
    git clone https://github.com/openppl-public/ppl.llm.kernel.cuda.git
    ```

* Building from Source

    ```bash
    ./build.sh -DPPLNN_CUDA_ENABLE_NCCL=ON -DPPLNN_ENABLE_CUDA_JIT=OFF -DPPLNN_CUDA_ARCHITECTURES="'80;86;87'" -DPPLCOMMON_CUDA_ARCHITECTURES="'80;86;87'"
    ```

### License

This project is distributed under the [Apache License, Version 2.0](LICENSE).
