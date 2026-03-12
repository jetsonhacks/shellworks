sudo docker run -it --rm --pull always \
  --runtime=nvidia --network host \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface:/data/models/huggingface \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \
  vllm serve nvidia/Qwen3-8B-FP8 \
  --gpu-memory-utilization 0.20 \
  --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --kv-cache-dtype fp8