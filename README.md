# LangGraph Agent

A lightweight OpenManus-style agent built with LangGraph for automated dataset analysis and PDF report generation.

## Model

Qwen/Qwen3-14B-AWQ (served via vLLM)

Start server:

```bash
vllm serve Qwen/Qwen3-14B-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```
Run:
```bash
python3 graph.py
```
Generated report:
workspace/student_analysis_report.pdf


langgraph_agent/  
├── graph.py  
├── nodes.py  
├── tools/  
├── workspace/  
└── README.md  
