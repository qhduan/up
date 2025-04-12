cd vllm

# python3 -m sglang.bench_one_batch_server --model deepseek-chat --base-url http://localhost:8000/ --batch-size 1 --output-len 1024 --input-len 1024

MODEL=cognitivecomputations/DeepSeek-V3-0324-AWQ
SERVED_MODEL=deepseek-chat
# SEED=227

echo "一般问答场景，输入200个token，单并发速度"
python3 benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/chat/completions \
    --model $MODEL --served-model-name $SERVED_MODEL \
    --dataset-name=random --random-input-len 200 --random-output-len 4096 \
    --seed 10 --num-prompts 20 --request-rate 1 --max-concurrency 1


echo "一般问答场景，输入200个token，2并发速度"
python3 benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/chat/completions \
    --model $MODEL --served-model-name $SERVED_MODEL \
    --dataset-name=random --random-input-len 200 --random-output-len 4096 \
    --seed 20 --num-prompts 20 --request-rate 2 --max-concurrency 2


echo "一般问答场景，输入200个token，4并发速度"
python3 benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/chat/completions \
    --model $MODEL --served-model-name $SERVED_MODEL \
    --dataset-name=random --random-input-len 200 --random-output-len 4096 \
    --seed 30 --num-prompts 20 --request-rate 4 --max-concurrency 4


echo "一般问答场景，输入200个token，8并发速度"
python3 benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/chat/completions \
    --model $MODEL --served-model-name $SERVED_MODEL \
    --dataset-name=random --random-input-len 200 --random-output-len 4096 \
    --seed 40 --num-prompts 20 --request-rate 8 --max-concurrency 8


echo "RAG场景，输入5000个token，1并发速度"
python3 benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/chat/completions \
    --model $MODEL --served-model-name $SERVED_MODEL \
    --dataset-name=random --random-input-len 5000 --random-output-len 4096 \
    --seed 50 --num-prompts 10 --request-rate 1 --max-concurrency 1


echo "RAG场景，输入5000个token，2并发速度"
python3 benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/chat/completions \
    --model $MODEL --served-model-name $SERVED_MODEL \
    --dataset-name=random --random-input-len 5000 --random-output-len 4096 \
    --seed 60 --num-prompts 10 --request-rate 2 --max-concurrency 2


echo "RAG场景，输入5000个token，4并发速度"
python3 benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/chat/completions \
    --model $MODEL --served-model-name $SERVED_MODEL \
    --dataset-name=random --random-input-len 5000 --random-output-len 4096 \
    --seed 70 --num-prompts 10 --request-rate 4 --max-concurrency 4


echo "RAG场景，输入10000个token，1并发速度"
python3 benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/chat/completions \
    --model $MODEL --served-model-name $SERVED_MODEL \
    --dataset-name=random --random-input-len 10000 --random-output-len 4096 \
    --seed 80 --num-prompts 10 --request-rate 1 --max-concurrency 1


echo "RAG场景，输入10000个token，2并发速度"
python3 benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/chat/completions \
    --model $MODEL --served-model-name $SERVED_MODEL \
    --dataset-name=random --random-input-len 10000 --random-output-len 4096 \
    --seed 90 --num-prompts 10 --request-rate 2 --max-concurrency 2


echo "RAG场景，输入10000个token，4并发速度"
python3 benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/chat/completions \
    --model $MODEL --served-model-name $SERVED_MODEL \
    --dataset-name=random --random-input-len 10000 --random-output-len 4096 \
    --seed 100 --num-prompts 10 --request-rate 4 --max-concurrency 4


echo "RAG场景，输入20000个token，1并发速度"
python3 benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/chat/completions \
    --model $MODEL --served-model-name $SERVED_MODEL \
    --dataset-name=random --random-input-len 20000 --random-output-len 4096 \
    --seed 110 --num-prompts 10 --request-rate 1 --max-concurrency 1


echo "RAG场景，输入20000个token，2并发速度"
python3 benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/chat/completions \
    --model $MODEL --served-model-name $SERVED_MODEL \
    --dataset-name=random --random-input-len 20000 --random-output-len 4096 \
    --seed 120 --num-prompts 10 --request-rate 2 --max-concurrency 2


echo "RAG场景，输入20000个token，4并发速度"
python3 benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/chat/completions \
    --model $MODEL --served-model-name $SERVED_MODEL \
    --dataset-name=random --random-input-len 20000 --random-output-len 4096 \
    --seed 130 --num-prompts 10 --request-rate 4 --max-concurrency 4

