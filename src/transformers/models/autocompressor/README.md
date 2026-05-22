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

## Usage

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_id = "princeton-nlp/AutoCompressor-Llama-2-7b-6k"

# Auto classes (Hub config.json uses model_type=llama; AutoConfig remaps when summary_length is present)
config = AutoConfig.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Or explicit classes
from transformers.models.autocompressor.modeling_autocompressor import AutoCompressorForCausalLM

model = AutoCompressorForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).eval().cuda()

prompt = 'The first name of the current US president is "'
prompt_tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
context_tokens = tokenizer(long_context, add_special_tokens=False, return_tensors="pt").input_ids.cuda()

summary_vectors = model(context_tokens, output_soft_prompt=True).soft_prompt
generation = model.generate(prompt_tokens, soft_prompt=summary_vectors, max_new_tokens=12, do_sample=False)
```

## Requirements

- **PyTorch ≥ 2.6** when loading `.bin` checkpoints (Transformers security policy for `torch.load`).
- GPU recommended for the 7B checkpoint (~14GB weights in bf16).

## Tests

```bash
pytest tests/models/autocompressor/test_modeling_autocompressor.py
```

Full 7B smoke test (requires downloaded weights + GPU):

```bash
export HF_HOME=/path/to/hf/cache
python scripts/verify_autocompressor_7b_official.py
```
