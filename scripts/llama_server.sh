#!/usr/bin/env bash
# launcher/llama_server.sh
#
# Starts llama-server with an OpenAI-compatible API endpoint.
# shellworks connects to this via VLLM_BASE_URL=http://localhost:8080/v1
#
# First run: the model (~2.6GB) is downloaded from HuggingFace automatically.
# Subsequent runs use the cached copy in ~/.cache/huggingface/
#
# Usage:
#   chmod +x launcher/llama_server.sh
#   ./launcher/llama_server.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# HuggingFace token — required to download gated models.
# Set this in your .env or export it before running.
# ---------------------------------------------------------------------------
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "Error: HF_TOKEN is not set. Export it or add it to your .env file." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
HF_REPO="Qwen/Qwen3-4B-GGUF"
HF_FILE="Qwen3-4B-Q4_K_M.gguf"   # Q4_K_M: good balance of quality and size (~2.6GB)

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
PORT=8080                          # matches VLLM_BASE_URL=http://localhost:8080/v1
CTX_SIZE=4096                      # max tokens per conversation turn (input + output)
GPU_LAYERS=99                      # offload all layers to GPU; lower this if you OOM
HOST="127.0.0.1"                   # localhost only — change to 0.0.0.0 to expose on LAN

echo "Starting llama-server on http://${HOST}:${PORT}/v1"
echo "Model: ${HF_REPO} / ${HF_FILE}"
echo "Press Ctrl-C to stop."
echo ""

llama-server \
    --hf-repo  "${HF_REPO}" \
    --hf-file  "${HF_FILE}" \
    --host     "${HOST}" \
    --port     "${PORT}" \
    --ctx-size "${CTX_SIZE}" \
    --n-gpu-layers "${GPU_LAYERS}" \
    --no-mmap                      # safer on unified memory — avoids memory-mapped I/O