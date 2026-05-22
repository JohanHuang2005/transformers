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
"""PyTorch AutoCompressor model."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.checkpoint
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.type_validators import interval
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaPreTrainedModel


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="princeton-nlp/AutoCompressor-Llama-2-7b-6k")
@strict
class AutoCompressorConfig(LlamaConfig):
    r"""
    Configuration class for the [AutoCompressor](https://huggingface.co/papers/2305.14788) architecture.

    AutoCompressor extends Llama with learnable summary tokens that compress long contexts into a compact
    `soft_prompt` representation for downstream generation.

    Args:
        summary_length (`int`, *optional*, defaults to 50):
            Number of summary vectors appended at the end of each segment during compression. Set to `0` to disable
            summary tokens.
        accumulate_summary (`bool`, *optional*, defaults to `True`):
            Whether to concatenate summary vectors from all previous segments into the soft prompt passed to the next
            segment.
        segment_gradient_checkpointing (`bool`, *optional*, defaults to `True`):
            Whether to apply gradient checkpointing on non-final segments during training.

    ```python
    >>> from transformers import AutoCompressorConfig, AutoCompressorForCausalLM

    >>> configuration = AutoCompressorConfig()
    >>> model = AutoCompressorForCausalLM(configuration)
    ```
    """

    model_type = "autocompressor"

    # Llama-2-7b defaults aligned with princeton-nlp/AutoCompressor-Llama-2-7b-6k config.json
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = interval(min=0.0, max=1.0)(default=0.02)
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    pretraining_tp: int | None = 1
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float | None = 0.0
    mlp_bias: bool = False
    head_dim: int | None = None

    # AutoCompressor-specific parameters from config.json
    summary_length: int = 50
    accumulate_summary: bool = True
    segment_gradient_checkpointing: bool = True


@dataclass
class AutoCompressorCausalLMOutputWithPast(CausalLMOutputWithPast):
    r"""
    Args:
        soft_prompt (`torch.FloatTensor` of shape `(batch_size, num_summary_tokens, hidden_size)`, *optional*):
            Compressed summary vectors produced when `output_soft_prompt=True`. Can be passed back to `forward` or
            `generate` via the `soft_prompt` argument.
    """

    soft_prompt: torch.FloatTensor | None = None


class AutoCompressorPreTrainedModel(LlamaPreTrainedModel):
    config_class = AutoCompressorConfig
    _no_split_modules = ["LlamaDecoderLayer"]


class AutoCompressorModel(LlamaModel):
    pass


class AutoCompressorForCausalLM(LlamaForCausalLM):
    def __init__(self, config: AutoCompressorConfig):
        super().__init__(config)
        self._setup_autocompressor(config)

    def _setup_autocompressor(self, config: AutoCompressorConfig) -> None:
        if config.summary_length > 0:
            embed_dim = self.get_input_embeddings().embedding_dim
            self.embed_summary = nn.Embedding(config.summary_length, embed_dim)
            eos_token_id = config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            self.embed_summary.weight.data.copy_(self.get_input_embeddings().weight[eos_token_id])

    def _get_past_key_values_length(self, past_key_values: Cache | None) -> int:
        if past_key_values is None:
            return 0
        return past_key_values.get_seq_length()

    def _get_summary_token_embeds(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.config.summary_length > 0:
            summary_token_ids = torch.arange(self.config.summary_length, device=device, dtype=torch.long)
            summary_token_ids = summary_token_ids.unsqueeze(0).expand(batch_size, -1)
            return self.embed_summary(summary_token_ids).to(dtype)
        return torch.empty(batch_size, 0, self.config.hidden_size, device=device, dtype=dtype)

    def _unpack_past_key_values(
        self, past_key_values: Cache | dict | None, soft_prompt: torch.Tensor | None
    ) -> tuple[Cache | None, torch.Tensor | None, int]:
        past_key_values_softprompt_length = 0
        if isinstance(past_key_values, dict):
            soft_prompt = past_key_values.get("soft_prompt", soft_prompt)
            past_key_values = past_key_values.get("past_key_values")
            if soft_prompt is not None:
                past_key_values_softprompt_length = soft_prompt.size(1)
        return past_key_values, soft_prompt, past_key_values_softprompt_length

    def _wrap_past_key_values(self, past_key_values: Cache | None, soft_prompt: torch.Tensor | None) -> Cache | dict | None:
        if past_key_values is None:
            return None
        return {"past_key_values": past_key_values, "soft_prompt": soft_prompt}

    def _forward_standard_causal_lm(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> AutoCompressorCausalLMOutputWithPast:
        # Do not call super().forward(): modular conversion inlines Llama and breaks MRO.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return AutoCompressorCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            soft_prompt=None,
        )

    def _forward_segment(
        self,
        soft_prompt: torch.Tensor,
        segment_embeds: torch.Tensor,
        summary_token_embeds: torch.Tensor,
        segment_attention_mask: torch.Tensor,
        past_key_values: Cache | None,
        use_cache: bool | None,
        segment_gradient_checkpointing: bool,
        past_key_values_softprompt_length: int,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[CausalLMOutputWithPast, torch.Tensor, torch.Tensor]:
        batch_size = segment_embeds.size(0)
        summary_length = summary_token_embeds.size(1)

        if past_key_values_softprompt_length > 0:
            softprompt_length = 0
            segment_embeds = torch.cat([segment_embeds, summary_token_embeds], dim=1)
            device = segment_embeds.device
            attn_dtype = segment_attention_mask.dtype
            segment_attention_mask = torch.cat(
                [
                    torch.ones(
                        batch_size, past_key_values_softprompt_length, device=device, dtype=attn_dtype
                    ),
                    segment_attention_mask,
                    torch.ones(batch_size, summary_length, device=device, dtype=attn_dtype),
                ],
                dim=1,
            )
        else:
            softprompt_length = soft_prompt.size(1)
            segment_embeds = torch.cat([soft_prompt, segment_embeds, summary_token_embeds], dim=1)
            device = segment_embeds.device
            attn_dtype = segment_attention_mask.dtype
            segment_attention_mask = torch.cat(
                [
                    torch.ones(batch_size, softprompt_length, device=device, dtype=attn_dtype),
                    segment_attention_mask,
                    torch.ones(batch_size, summary_length, device=device, dtype=attn_dtype),
                ],
                dim=1,
            )

        def decoder(
            segment_embeds,
            segment_attention_mask,
            segment_past_key_values,
            softprompt_length,
            past_key_values_softprompt_length,
            summary_length,
        ):
            forward_kwargs = dict(kwargs)
            position_ids = forward_kwargs.get("position_ids")
            if position_ids is not None and position_ids.size(1) != segment_embeds.size(1):
                # Generation passes position_ids for prompt tokens only; soft_prompt/summary extend the sequence.
                forward_kwargs.pop("position_ids", None)
            return self.model(
                inputs_embeds=segment_embeds,
                attention_mask=segment_attention_mask,
                past_key_values=segment_past_key_values,
                use_cache=use_cache,
                **forward_kwargs,
            )

        if segment_gradient_checkpointing:
            outputs = torch.utils.checkpoint.checkpoint(
                decoder,
                segment_embeds,
                segment_attention_mask,
                past_key_values,
                softprompt_length,
                past_key_values_softprompt_length,
                summary_length,
                use_reentrant=False,
            )
        else:
            outputs = decoder(
                segment_embeds,
                segment_attention_mask,
                past_key_values,
                softprompt_length,
                past_key_values_softprompt_length,
                summary_length,
            )

        total_length = outputs.last_hidden_state.size(1)
        segment_last_hidden_states = outputs.last_hidden_state[
            :, softprompt_length : total_length - summary_length
        ]
        new_soft_prompt = outputs.last_hidden_state[:, total_length - summary_length :]

        return outputs, segment_last_hidden_states, new_soft_prompt

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | dict | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        is_first_iteration: bool | None = False,
        **kwargs,
    ):
        model_kwargs = dict(kwargs)
        if isinstance(past_key_values, dict):
            # After the first step, soft_prompt lives in the cache bundle; do not prepend it again.
            model_kwargs.pop("soft_prompt", None)
            past_key_values_for_super = past_key_values.get("past_key_values")
        else:
            past_key_values_for_super = past_key_values

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values_for_super,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            is_first_iteration=is_first_iteration,
            **model_kwargs,
        )

        if isinstance(past_key_values, dict):
            model_inputs["past_key_values"] = past_key_values

        return model_inputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | dict | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        soft_prompt: torch.FloatTensor | None = None,
        output_soft_prompt: bool | None = None,
        segment_lengths: int | list[int] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> AutoCompressorCausalLMOutputWithPast:
        r"""
        Args:
            soft_prompt (`torch.FloatTensor` of shape `(batch_size, num_summary_tokens, hidden_size)`, *optional*):
                Pre-computed summary vectors to prepend to the current segment (e.g. from a prior compression pass).
            output_soft_prompt (`bool`, *optional*):
                Whether to return compressed summary vectors in `soft_prompt`.
            segment_lengths (`int` or `list[int]`, *optional*):
                Length(s) of segments used when compressing a long sequence without `past_key_values`. Defaults to the
                full input length when not provided.
        """
        if self.config.summary_length == 0:
            return self._forward_standard_causal_lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values if not isinstance(past_key_values, dict) else past_key_values["past_key_values"],
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        uses_autocompressor_path = (
            output_soft_prompt
            or soft_prompt is not None
            or segment_lengths is not None
            or isinstance(past_key_values, dict)
        )
        if not uses_autocompressor_path:
            return self._forward_standard_causal_lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("AutoCompressor does not support both `input_ids` and `inputs_embeds` at the same time.")

        return_dict = kwargs.pop("return_dict", self.config.use_return_dict)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        past_key_values, soft_prompt, past_key_values_softprompt_length = self._unpack_past_key_values(
            past_key_values, soft_prompt
        )
        past_key_values_length = self._get_past_key_values_length(past_key_values) - past_key_values_softprompt_length

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must specify either `input_ids` or `inputs_embeds`.")
            inputs_embeds = self.get_input_embeddings()(input_ids)

        batch_size = inputs_embeds.size(0)
        embed_dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        summary_token_embeds = self._get_summary_token_embeds(batch_size, device, embed_dtype)

        if past_key_values is None:
            if segment_lengths is None:
                segment_lengths = inputs_embeds.size(1)
            if isinstance(segment_lengths, int):
                segment_lengths = [segment_lengths]

            if attention_mask is None:
                attention_mask = torch.ones(batch_size, inputs_embeds.size(1), dtype=torch.long, device=device)

            inputs_embeds_list = torch.split(inputs_embeds, segment_lengths, dim=1)
            attention_mask_list = torch.split(attention_mask, segment_lengths, dim=1)
            summary_token_embeds_list = (
                (summary_token_embeds,) * (len(inputs_embeds_list) - 1)
                + (summary_token_embeds if output_soft_prompt else summary_token_embeds[:, :0],)
            )
        else:
            if attention_mask is None:
                attention_mask = torch.ones(
                    batch_size, inputs_embeds.size(1) + past_key_values_length, dtype=torch.long, device=device
                )

            if (
                use_cache
                and segment_lengths is not None
                and past_key_values_length + inputs_embeds.size(1) == segment_lengths
            ):
                output_soft_prompt = True
                inputs_embeds_list = (inputs_embeds, inputs_embeds[:, :0])
                attention_mask_list = (attention_mask, attention_mask[:, :0])
                summary_token_embeds_list = (summary_token_embeds, summary_token_embeds[:, :0])
            else:
                inputs_embeds_list = (inputs_embeds,)
                attention_mask_list = (attention_mask,)
                summary_token_embeds_list = (
                    (summary_token_embeds if output_soft_prompt else summary_token_embeds[:, :0],)
                )

        if soft_prompt is None:
            soft_prompt = inputs_embeds[:, :0]

        last_hidden_state_list = []
        output_attentions = kwargs.get("output_attentions", False)
        output_hidden_states = kwargs.get("output_hidden_states", False)
        output_attentions_list = []
        output_hidden_states_list = []
        segment_outputs = None

        for step, segment_summary_embeds in enumerate(summary_token_embeds_list):
            is_last_step = step == len(inputs_embeds_list) - 1
            segment_gradient_checkpointing = (
                self.config.segment_gradient_checkpointing
                and self.training
                and not is_last_step
            )

            segment_outputs, segment_hidden_states, new_soft_prompt = self._forward_segment(
                soft_prompt.to(embed_dtype),
                inputs_embeds_list[step],
                segment_summary_embeds,
                attention_mask_list[step],
                past_key_values,
                use_cache,
                segment_gradient_checkpointing,
                past_key_values_softprompt_length,
                position_ids=position_ids,
                **kwargs,
            )

            last_hidden_state_list.append(segment_hidden_states)

            if self.config.accumulate_summary:
                soft_prompt = torch.cat([soft_prompt, new_soft_prompt], dim=1)
            elif new_soft_prompt.size(1) > 0:
                soft_prompt = new_soft_prompt

            if output_attentions:
                output_attentions_list.append(segment_outputs.attentions)
            if output_hidden_states:
                output_hidden_states_list.append(segment_outputs.hidden_states)

            past_key_values = None
            past_key_values_softprompt_length = 0

        past_key_values = segment_outputs.past_key_values if segment_outputs is not None else None

        last_hidden_states = torch.cat(last_hidden_state_list, dim=1)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(last_hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        output = AutoCompressorCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=self._wrap_past_key_values(past_key_values, soft_prompt),
            hidden_states=output_hidden_states_list if output_hidden_states and output_hidden_states_list else None,
            attentions=output_attentions_list if output_attentions and output_attentions_list else None,
            soft_prompt=soft_prompt,
        )

        if return_dict:
            return output
        return tuple(v for v in [output.loss, output.logits, output.past_key_values, output.hidden_states, output.attentions, output.soft_prompt] if v is not None)


__all__ = [
    "AutoCompressorConfig",
    "AutoCompressorCausalLMOutputWithPast",
    "AutoCompressorPreTrainedModel",
    "AutoCompressorModel",
    "AutoCompressorForCausalLM",
]
