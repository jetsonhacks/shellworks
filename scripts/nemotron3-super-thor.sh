sudo docker run -it --rm --pull always \
  --runtime=nvidia --network host \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface:/data/models/huggingface \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \
  bash -c "wget -q -O /tmp/super_v3_reasoning_parser.py \
  --header=\"Authorization: Bearer \$HF_TOKEN\" \
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/resolve/main/super_v3_reasoning_parser.py \
&& vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin /tmp/super_v3_reasoning_parser.py \
  --reasoning-parser super_v3 \
  --async-scheduling \
  --attention-backend TRITON_ATTN \
  --enable-chunked-prefill \
  --max-num-seqs 512 \
  --kv-cache-dtype fp8"