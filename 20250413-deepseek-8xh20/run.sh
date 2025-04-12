

# deepseek v3 0324 fp8 36 t/s
# vllm==0.8.3
# VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_MARLIN_USE_ATOMIC_ADD=1 python -m vllm.entrypoints.openai.api_server \
#     --host 0.0.0.0 --port 8000 \
#     --enable-chunked-prefill --enable-prefix-caching --trust-remote-code \
#     --max-model-len 64000 --max-seq-len-to-capture 64000 \
#     --tensor-parallel-size 8 --gpu-memory-utilization 0.98 \
#     --served-model-name deepseek-chat --model DeepSeek-V3-0324

# deepseek v3 0324 awq v0 45 t/s
# vllm==0.8.3
VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_MARLIN_USE_ATOMIC_ADD=1 python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 160000 --max-seq-len-to-capture 160000 \
    --enable-chunked-prefill --enable-prefix-caching --trust-remote-code \
    --tensor-parallel-size 8 --gpu-memory-utilization 0.98 \
    --served-model-name deepseek-chat --model DeepSeek-V3-0324-AWQ

# deepseek v3 0324 awq v1 42 t/s
# vllm==0.8.3
# VLLM_WORKER_MULTIPROC_METHOD=spawn python -m vllm.entrypoints.openai.api_server \
#     --host 0.0.0.0 --port 8000 \
#     --max-model-len 128000 --max-seq-len-to-capture 128000 \
#     --enable-chunked-prefill --enable-prefix-caching --trust-remote-code \
#     --tensor-parallel-size 8 --gpu-memory-utilization 0.9 \
#     --served-model-name deepseek-chat --model DeepSeek-V3-0324-AWQ

# vllm==0.7.2
# sglang==0.4.5
# sglang在一些时候需要vllm，但是不要把vllm升级到最新版本，看sglang的提示就好了
# 13.3 t/s
# VLLM_WORKER_MULTIPROC_METHOD=spawn python3 -m sglang.launch_server --model-path DeepSeek-V3-0324-AWQ --host 0.0.0.0 --port 8000 \
#     --tensor-parallel-size 8 --enable-flashinfer-mla --dtype float16

# 5.4 t/s
# VLLM_WORKER_MULTIPROC_METHOD=spawn python3 -m sglang.launch_server --model-path DeepSeek-V3-0324 --host 0.0.0.0 --port 8000 \
#     --tensor-parallel-size 8 --enable-flashinfer-mla --mem-fraction-static 0.9 --disable-cuda-graph --context-length 64000

# 
# curl http://localhost:12345/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "deepseek-chat",
#     "messages": [
#       {
#         "role": "system",
#         "content": "You are a helpful assistant."
#       },
#       {
#         "role": "user",
#         "content": "帮我写一个一万字的太湖旅游攻略"
#       }
#     ],
#     "stream": true,
#     "temperature": 0.7
#   }'
  
