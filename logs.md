问题1：OpenAI 客户端在发请求时拿到的 URL 没有 http:// 或 https://，所以 httpx 抛了：
Request URL is missing an 'http://' or 'https://' protocol.


openai.BadRequestError: Error code: 400

激活venv环境：source ~/venv310/bin/activate

vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 4096