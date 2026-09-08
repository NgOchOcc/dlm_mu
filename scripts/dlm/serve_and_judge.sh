#!/usr/bin/env bash
# LLM-judge: serve rồi chấm điểm. Copy từng khối chạy tay.
cd /prj/corp/airesearch/lasvegas/vol20-scratch/nluu/working/open-unlearning

# 1) Serve judge (Qwen3.5-27B) trên GPU 0, port 8000 — chạy nền
setsid bash scripts/dlm/serve_judge.sh start > logs/judge/vllm_serve_8000.log 2>&1 < /dev/null &

# 2) Đợi judge sẵn sàng
bash scripts/dlm/serve_judge.sh wait

# 3) Chấm điểm (dream + llada, mọi split)
bash scripts/dlm/run_judge.sh dream llada

# 4) Xem kết quả
python scripts/dlm/show_results.py

# 5) Dừng server
bash scripts/dlm/serve_judge.sh stop
