# Chạy LLaDA / Dream × TOFU trên AMD MI300 (ROCm)

Hướng dẫn từ đầu: tạo conda env → cài lib → chạy 1 file pipeline (finetune 300 epochs →
eval → 5 method unlearning: GA, GD, NPO, WGA, SimNPO → eval → xuất CSV).

Đường dẫn giả định:
- Repo: `/prj/corp/airesearch/lasvegas/vol11-scratch/nluu/working/open-unlearning`
- `dllm`: `/prj/corp/airesearch/lasvegas/vol11-scratch/nluu/working/Information-Gain-Sampler/dllm`

```bash
export REPO=/prj/corp/airesearch/lasvegas/vol11-scratch/nluu/working/open-unlearning
export DLLM=/prj/corp/airesearch/lasvegas/vol11-scratch/nluu/working/Information-Gain-Sampler/dllm
cd "$REPO"
```

---

## 1. Tạo conda env

```bash
conda create -n unlearn-rocm python=3.11 -y
conda activate unlearn-rocm
```

## 2. Cài thư viện

**2.1 — PyTorch bản ROCm TRƯỚC** (đừng để pip kéo wheel CUDA). Chỉnh URL theo ROCm của
node (`cat /opt/rocm/.info/version`):

```bash
pip install --index-url https://download.pytorch.org/whl/rocm6.2 \
    torch==2.7.0 torchvision torchaudio
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.hip)"
# phải in ra True và một số hiệu HIP
```

**2.2 — deps Python** (đã pin sẵn trong `requirements.txt`, không gồm torch):

```bash
pip install -r requirements.txt
python -c "import deepspeed; print('deepspeed', deepspeed.__version__)"
# Nếu deepspeed==0.18.0 build lỗi trên ROCm: DS_BUILD_OPS=0 pip install deepspeed
```

**2.3 — cài editable 2 gói** (`dllm` chứa class model LLaDA/Dream, rồi repo này):

```bash
pip install -e "$DLLM"
pip install -e "$REPO"
```

**Ghi chú:** flash-attn / bitsandbytes / vllm KHÔNG cần (đã comment trong
`requirements.txt`) — LLaDA/Dream tự fallback sdpa/eager.

## 3. HF cache (tùy chọn)

```bash
export HF_HOME=$REPO/.hf_cache      # để cache weights/data trên ổ lớn
# TOFU data + weights LLaDA/Dream đều public, không cần login
```

---

## 4. Chạy pipeline — chỉ 1 file

Toàn bộ (finetune 300ep → eval → unlearn 5 method → eval → CSV) nằm trong 1 script.
Mặc định dùng **8 GPU MI300** cho train, GPU 0 cho eval.

**LLaDA:**
```bash
nohup bash scripts/dlm/llada/run_full_pipeline.sh > run_llada.log 2>&1 &
```

**Dream:**
```bash
nohup bash scripts/dlm/dream/run_full_pipeline.sh > run_dream.log 2>&1 &
```

Theo dõi: `tail -f run_llada.log`.

Chỉnh tham số ở đầu mỗi script nếu cần: `FT_GPUS`, `UNLEARN_GPUS`, `EVAL_GPU`,
`FT_EPOCHS` (mặc định 300), `unlearn_epochs` (5), danh sách `trainers`.
LLaDA dùng `RIGHT_SHIFT=False`; Dream dùng `RIGHT_SHIFT=True` + `LOSS_WEIGHT='cart[geo_p:0.3]'`
(đã set sẵn trong từng file).

---

## 5. Kết quả

- Model finetune: `saves/finetune/tofu_<MODEL>_full/`
- Checkpoint unlearn: `saves/unlearn/tofu_<MODEL>_forget10_Diffusion<Method>/`
- Eval mỗi checkpoint: `saves/eval/<task_name>/TOFU_SUMMARY.json`
- **CSV tổng hợp: `saves/eval/tofu_results_summary.csv`** — mỗi dòng 1 (model, method),
  cột: `model_utility`, `forget_truth_ratio`, và prob/rougeL từng split.

Nếu chạy cả LLaDA và Dream, CSV gộp chung cả hai (phân biệt theo cột `model`). Chạy
lại tổng hợp thủ công bất cứ lúc nào:

```bash
python scripts/dlm/aggregate_results.py
```

> `forget_quality` bị loại (cần retain-only reference model chưa có). Muốn thêm:
> finetune 1 model chỉ trên retain90, eval nó, rồi truyền
> `eval.tofu_dlm.retain_logs_path=<TOFU_EVAL.json đó>`.

---

## Lưu ý ROCm nhanh

| Mục | MI300 | Xử lý |
|-----|-------|-------|
| `torch.cuda.*` | Chạy (alias HIP) | không cần sửa |
| flash-attn | Không cần | đừng bật `flash_attention_2` |
| DeepSpeed ZeRO-3 | Cần cho train | build ROCm (mục 2.2) |
| `NCCL_*` trong script | RCCL đọc prefix `NCCL_`, NVLS no-op | để nguyên |
| Chọn GPU | `CUDA_VISIBLE_DEVICES` chạy được | hoặc `HIP_VISIBLE_DEVICES` |
| Chạy dài | 300 epochs rất lâu | dùng `nohup`/`tmux` |
