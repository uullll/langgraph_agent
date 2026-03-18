问题1：OpenAI 客户端在发请求时拿到的 URL 没有 http:// 或 https://，所以 httpx 抛了：
Request URL is missing an 'http://' or 'https://' protocol.


openai.BadRequestError: Error code: 400

激活venv环境：source ~/venv310/bin/activate
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct-AWQ
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 4096

  source ~/venv310/bin/activate
source /etc/network_turbo
vllm serve Qwen/Qwen3-14B-AWQ \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes



vllm serve Qwen/Qwen3-4B \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

huggingface-cli download Qwen/Qwen3-14B-AWQ --local-dir ./Qwen3-14B-AWQ

安装
python3 -m venv vllm_env
source vllm_env/bin/activate

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install vllm
pip install transformers
pip install accelerate
pip install huggingface_hub
pip install langgraph langchain-openai
pip install langchain
pip install langchain-openai
huggingface-cli login

查看模型
ls ~/.cache/huggingface/hub
删除模型
rm -rf ~/.cache/huggingface/hub/


出现结果不生成pdf问题，尝试添加函数强制生成pdf
失败，生成pdf为乱码，是内容输出content不正确问题