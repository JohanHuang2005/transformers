# AutoCompressor

Modular implementation of [AutoCompressor](https://arxiv.org/abs/2305.14788) (Princeton NLP), extending Llama with learnable summary tokens that compress long contexts into a compact `soft_prompt` for downstream generation.

Reference checkpoint: [`princeton-nlp/AutoCompressor-Llama-2-7b-6k`](https://huggingface.co/princeton-nlp/AutoCompressor-Llama-2-7b-6k).

## Modular layout

| File | Role |
|------|------|
| `modular_autocompressor.py` | **Source file** — edit this file only |
| `configuration_autocompressor.py` | Generated from modular |
| `modeling_autocompressor.py` | Generated from modular |

Regenerate derived files after editing the modular source:

```bash
pip install libcst
python utils/modular_model_converter.py autocompressor
```

## Auto loading

Hub weights still ship with `model_type: "llama"` in `config.json`. `AutoConfig` remaps to `autocompressor` when `summary_length` is present (see `configuration_auto.py`), so the following works out of the box:

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_id = "princeton-nlp/AutoCompressor-Llama-2-7b-6k"
config = AutoConfig.from_pretrained(model_id)  # -> AutoCompressorConfig
model = AutoModelForCausalLM.from_pretrained(model_id)  # -> AutoCompressorForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "princeton-nlp/AutoCompressor-Llama-2-7b-6k"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = 'The first name of the current US president is "'
prompt_tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
context_tokens = tokenizer(long_context, add_special_tokens=False, return_tensors="pt").input_ids.cuda()

summary_vectors = model(context_tokens, output_soft_prompt=True).soft_prompt
generation = model.generate(prompt_tokens, soft_prompt=summary_vectors, max_new_tokens=12, do_sample=False)
```

Or use the explicit class:

```python
from transformers.models.autocompressor.modeling_autocompressor import AutoCompressorForCausalLM
```

## Requirements

- **PyTorch ≥ 2.6** when loading `.bin` checkpoints (Transformers security policy for `torch.load`).
- GPU recommended for the 7B checkpoint (~14GB weights in bf16).

## Tests

Unit tests (no GPU, tiny random config):

```bash
unset OMP_NUM_THREADS MKL_NUM_THREADS
python -m unittest tests.models.autocompressor.test_modeling_autocompressor -v
```

Full 7B smoke test (requires downloaded weights + GPU):

```bash
export HF_HOME=/path/to/hf/cache
# Avoid Hub network calls during .bin load (local cache only):
export HF_HUB_OFFLINE=1
export DISABLE_SAFETENSORS_CONVERSION=1
python scripts/verify_autocompressor_7b_official.py
```

---

## 中文说明（refactor 分支）

本目录为任务 1.2 第 3 点的 **Modular Transformers** 实现，分支：`refactor`（fork: `JohanHuang2005/transformers`）。

### 目录与官方 modular 流程

1. 只修改 `modular_autocompressor.py`
2. 运行 converter 生成 `configuration_*.py` 与 `modeling_*.py`：

```bash
pip install libcst
python utils/modular_model_converter.py autocompressor
```

### Auto 类加载

Hub 上 checkpoint 的 `config.json` 仍为 `model_type: "llama"`。已在 `AutoConfig` 中增加启发式：若存在 `summary_length` 字段，则映射为 `autocompressor`，从而支持：

- `AutoConfig.from_pretrained(...)` → `AutoCompressorConfig`
- `AutoModelForCausalLM.from_pretrained(...)` → `AutoCompressorForCausalLM`

### 测试

| 测试 | 命令 | 说明 |
|------|------|------|
| 单元测试 | `python -m unittest tests.models.autocompressor.test_modeling_autocompressor -v` | Config / Auto 映射 / 压缩 forward / generate |
| 7B 官方示例 | `python scripts/verify_autocompressor_7b_official.py` | 660→50 压缩 + Joe Biden / Donald Trump 生成 |

推荐环境变量（权重已下载到本地时，避免联网卡住）：

```bash
unset OMP_NUM_THREADS MKL_NUM_THREADS
export HF_HOME=/path/to/hf/cache
export HF_HUB_OFFLINE=1
export DISABLE_SAFETENSORS_CONVERSION=1
# 国内镜像（首次下载时可选）
export HF_ENDPOINT=https://hf-mirror.com
```
