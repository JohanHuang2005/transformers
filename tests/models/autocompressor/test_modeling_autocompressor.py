# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the AutoCompressor model."""

import unittest

from transformers import AutoConfig, AutoModelForCausalLM, is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers.models.autocompressor.configuration_autocompressor import AutoCompressorConfig
    from transformers.models.autocompressor.modeling_autocompressor import AutoCompressorForCausalLM


@require_torch
class AutoCompressorModelTest(unittest.TestCase):
    def test_config_defaults(self):
        config = AutoCompressorConfig()
        self.assertEqual(config.model_type, "autocompressor")
        self.assertEqual(config.summary_length, 50)
        self.assertTrue(config.accumulate_summary)

    def test_auto_config_from_hub_cache(self):
        config = AutoConfig.from_pretrained("princeton-nlp/AutoCompressor-Llama-2-7b-6k")
        self.assertIsInstance(config, AutoCompressorConfig)
        self.assertEqual(config.model_type, "autocompressor")
        self.assertEqual(config.summary_length, 50)

    def test_auto_model_mapping(self):
        config = AutoCompressorConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=512,
            summary_length=10,
        )
        model_class = AutoModelForCausalLM._model_mapping[type(config)]
        self.assertEqual(model_class, AutoCompressorForCausalLM)

    def test_compression_forward(self):
        config = AutoCompressorConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=512,
            summary_length=10,
        )
        model = AutoCompressorForCausalLM(config)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 32))
        outputs = model(input_ids=input_ids, output_soft_prompt=True)
        self.assertIsNotNone(outputs.soft_prompt)
        self.assertEqual(outputs.soft_prompt.shape, (1, config.summary_length, config.hidden_size))

    def test_generate_with_soft_prompt(self):
        config = AutoCompressorConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=512,
            summary_length=10,
        )
        model = AutoCompressorForCausalLM(config)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        prompt_ids = torch.randint(0, config.vocab_size, (1, 4))
        with torch.no_grad():
            soft_prompt = model(input_ids=input_ids, output_soft_prompt=True).soft_prompt
            generated = model.generate(
                input_ids=prompt_ids,
                soft_prompt=soft_prompt,
                max_new_tokens=4,
                do_sample=False,
            )
        self.assertEqual(generated.shape[0], 1)
        self.assertGreater(generated.shape[1], prompt_ids.shape[1])
