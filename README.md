# Devan Carlin

Independent researcher. LLM inference and quantization on consumer hardware.

## What I do

- Quantization — INT4 (AutoRound) on large dense and MoE models
- Inference serving — vLLM on Intel Arc (XPU), llama.cpp on NVIDIA
- Benchmarking — quality and throughput across model sizes, context lengths, and quant formats
- Upstream fixes — patching vLLM and vllm-xpu-kernels for Intel hardware

## Hardware

- 4x Intel Arc Pro B70 (Battlemage, 32 GB each)
- NVIDIA RTX 5090
- AMD Ryzen Threadripper PRO 3945WX, 256 GB RAM

## Projects

- [electric-sheep](https://github.com/devan-carlin/electric-sheep) — deployment scripts for Intel Arc B70 (vLLM) and RTX 5090 (llama.cpp) inference servers. Includes the technical deep-dive and benchmark results.

## Writing and contributions

- "vLLM Crashes on Intel INT4 MoE Models. Here's the Fix." — write-up of a crash in vLLM's Intel quantization backend and the guard that fixes it
- vLLM — fix for a copy_() shape-mismatch crash when loading INT4 checkpoints with symmetric (empty-qzeros) layers
