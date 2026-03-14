#!/usr/bin/env bash
# launcher/llama_server_qwen35.sh
#
# Starts llama-server for Qwen3.5-4B-Q4_K_M with an OpenAI-compatible API.
# shellworks connects via VLLM_BASE_URL=http://localhost:8080/v1
#
# Reasoning control for Qwen3.5 is via chat_template_kwargs.enable_thinking
# in the API request body — NOT via /think or /no_think prompt markers.
# Those markers are specific to Qwen3 and are NOT supported by Qwen3.5.
#
# First run: the model (~2.7GB) is downloaded from HuggingFace automatically.
# Subsequent runs use the cached copy in ~/.cache/huggingface/
#
# After starting, verify the reported model name with:
#   curl -s http://localhost:8080/v1/models | python3 -m json.tool
# Update canonical_model in configs/model_profiles/qwen3_5_4b_gguf.toml
# if the reported name differs from "Qwen3.5-4B-Q4_K_M".
#
# Usage:
#   chmod +x launcher/llama_server_qwen35.sh
#   ./launcher/llama_server_qwen35.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# HuggingFace token — required to download gated models.
# ---------------------------------------------------------------------------
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "Error: HF_TOKEN is not set. Export it or add it to your .env file." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Check that llama-server is available
# ---------------------------------------------------------------------------
if ! command -v llama-server &>/dev/null; then
    echo "Error: llama-server not found in PATH." >&2
    echo "  Build it with: ./install/build_llama_cpp.sh" >&2
    echo "  Then symlink:  sudo ln -sf ~/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
HF_REPO="unsloth/Qwen3.5-4B-GGUF"
HF_FILE="Qwen3.5-4B-Q4_K_M.gguf"   # Q4_K_M: good balance of quality and size (~2.7GB)

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
PORT=8080                            # matches VLLM_BASE_URL=http://localhost:8080/v1
CTX_SIZE=8192                        # Qwen3.5 natively supports 262k; cap for Orin Nano RAM
GPU_LAYERS=99                        # offload all layers to GPU; lower if you OOM
HOST="127.0.0.1"

echo "Starting llama-server on http://${HOST}:${PORT}/v1"
echo "Model: ${HF_REPO} / ${HF_FILE}"
echo "Press Ctrl-C to stop."
echo ""
echo "After starting, verify model name with:"
echo "  curl -s http://localhost:${PORT}/v1/models | python3 -m json.tool"
echo ""

llama-server \
    --hf-repo         "${HF_REPO}" \
    --hf-file         "${HF_FILE}" \
    --host            "${HOST}" \
    --port            "${PORT}" \
    --ctx-size        "${CTX_SIZE}" \
    --n-gpu-layers    "${GPU_LAYERS}" \
    --jinja                          \
    --no-mmap                        # safer on unified memory