sudo docker run -it --rm --pull always \
  --runtime=nvidia --network host \
  -e HF_TOKEN=$HF_TOKEN \
  -e VLLM_USE_FLASHINFER_MOE_FP4=1 \
  -e VLLM_FLASHINFER_MOE_BACKEND=throughput \
  -v $HOME/.cache/huggingface:/data/models/huggingface \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \
  bash -c "wget -q -O /tmp/nano_v3_reasoning_parser.py \
  --header=\"Authorization: Bearer \$HF_TOKEN\" \
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4/resolve/main/nano_v3_reasoning_parser.py \
&& vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 8192 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin /tmp/nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3 \
  --kv-cache-dtype fp8"
