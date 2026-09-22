# Devan Carlin

Independent researcher. LLM inference and quantization on consumer and
workstation-class hardware.

## Scope

- **Quantization.** INT4 via AutoRound on large dense and MoE models. Quality
  measured against the BF16 baseline, not assumed.
- **Inference serving.** vLLM on Intel Arc through the XPU backend. llama.cpp on
  NVIDIA.
- **Upstream fixes.** Patches to vLLM and vllm-xpu-kernels for Intel hardware,
  contributed upstream rather than held in a private fork.
- **Tech stack.** Inference: `vLLM`, `llama.cpp`, `SGLang`, `AutoRound`,
  `oneAPI`/SYCL, Level-Zero. Image and video: `ComfyUI`.

## Hardware

- 4x Intel Arc Pro B70, 32 GB each, 128 GB total
- NVIDIA RTX 5090, 32 GB

## Projects

- [electric-sheep](https://github.com/devan-carlin/electric-sheep): serving
  stacks for Intel Arc B70 (`vLLM`) and RTX 5090 (`llama.cpp`), with the
  benchmarks and write-ups behind them.
  - Serves [Qwen3.8-Flash-Next W4A16](https://huggingface.co/devan-carlin/Qwen3.8-Flash-Next-W4A16):
    125B MoE, 6B active, 77 GB across 17 shards, 256K context, **53.4 tok/s** on
    4x B70.
  - Serves [Qwen3.8-27B INT4 (AutoRound)](https://huggingface.co/devan-carlin/Qwen3.8-27B-int4-AutoRound):
    18 GB, 256K context, **47.8 tok/s** on 4x B70, 58% faster than BF16 at measured
    quality parity.
  - Write-ups: *vLLM Supports Intel Arc. Getting a 125B MoE Onto Four of Them
    Still Took Two Patches.* and *vLLM Crashes on Intel INT4 MoE Models. Here's
    the Fix.*
- [vLLM fork, `xpu-qwen4exp` branch](https://github.com/devan-carlin/vllm/tree/xpu-qwen4exp):
  the two patches that make Qwen3.8-Flash-Next run on four discrete Arc cards.
  - Model port: 17 files. vLLM has no `qwen4_exp` architecture, so the patch adds
    the registry entry, config class, W4A16 linear kernels for the XPU backend,
    and the linear-attention forward path.
  - Worker affinity: 30 lines. vLLM gives every tensor-parallel worker (TP, one
    model split across several GPUs) the full GPU mask, so each rank initializes
    all four devices. Discrete Arc cards on desktop PCIe cannot open the
    resulting peer-memory handles. Rank N gets card N.
  - Upstream: fixed a `copy_()` shape-mismatch crash when loading INT4 checkpoints
    with symmetric (empty-qzeros) layers.
- [`actlens`](https://github.com/devan-carlin/actlens): cross-engine activation
  diff. Dumps intermediate tensors from a known-good engine (`llama.cpp`) and a
  suspect one (`vLLM`), walks layer-by-layer, and reports the first divergence.
  Localizes a bug to one layer in a single run. Found both `1 + w` gamma bugs
  behind the patches above: the HC-gamma bug (flat logprobs, one-line fix, ~1 day)
  and the PLE-norm bug.
- [intel-arc-pro-b70-inference-cookbook](https://github.com/devan-carlin/intel-arc-pro-b70-inference-cookbook):
  fork of SergiioB's cookbook, carrying the bare-metal vLLM XPU source-build
  recipe. Reproducible Arc B70 recipes: build steps, patch stack, launch flags,
  and measured results per model.
- [`vn-pipeline`](https://github.com/devan-carlin/vn-pipeline): pose-to-video
  pipeline. `SDPose` and `YOLOv8m-seg` extract skeletons, `ComfyUI` runs the
  graphs, `ffmpeg` assembles the output.
