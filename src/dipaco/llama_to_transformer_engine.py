# """
# https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/te_llama
# """

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaDecoderLayer
import transformers.models.llama.modeling_llama as modeling_llama
from contextlib import contextmanager
import transformer_engine as te
from transformer_engine.pytorch.fp8 import fp8_model_init


@contextmanager
def replace_decoder(te_decoder_cls):
    """
    Replace `LlamaDecoderLayer` with custom `TELlamaDecoderLayer`.
    """
    original_llama_decoder_cls = modeling_llama.LlamaDecoderLayer
    modeling_llama.LlamaDecoderLayer = te_decoder_cls
    try:
        yield
    finally:
        modeling_llama.LlamaDecoderLayer = original_llama_decoder_cls


class TELlamaDecoderLayer(te.pytorch.TransformerLayer):
    """
    Wrapper class over TE's `TransformerLayer`. This makes the wrapper very
    similar to HF's `LlamaDecoderLayer` and easier to replace it in the code.

    Args:
        config: LlamaConfig
        args: positional args (for compatibility with `LlamaDecoderLayer`)
        kwargs: keyword args (for compatibility with `LlamaDecoderLayer`)
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            bias=False,
            layernorm_epsilon=config.rms_norm_eps,
            hidden_dropout=0,
            attention_dropout=0,
            fuse_qkv_params=False,
            normalization="RMSNorm",
            activation="swiglu",
            attn_input_format="bshd",
            num_gqa_groups=config.num_key_value_heads,
        )

    def forward(self, hidden_states, *args, attention_mask, **kwargs):
        """
        Custom forward to make sure we only pass relevant arguments to the
        forward pass of the `TransformerLayer`. Also, make sure the output
        format matches the output of the HF's `LlamaDecoderLayer`.
        """
        return (
            super().forward(
                hidden_states, attention_mask=attention_mask),
        )


class TELlamaForCausalLM:
    """
    Causal LM created with `LlamaModel`. The underlying `LlamaDecoderLayer`
    class is monkey-patched with `TELlamaDecoderLayer` class before
    initializing the causal LM with `LlamaForCausalLM`.

    Args:
        config: LlamaConfig
    """

    def __new__(cls, config: LlamaConfig):
        with replace_decoder(te_decoder_cls=TELlamaDecoderLayer):
            llama_for_causal_lm = LlamaForCausalLM(config)
        return llama_for_causal_lm

def replace_linear_layer_with_te_linear(module, parent_name=""):
    for name, child in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name

        if isinstance(child, torch.nn.Linear):
            if full_name.endswith("lm_head"):  
                print(f"Skipping {full_name} because vocab size is not divisible by 16")
                continue

            setattr(module, name, te.pytorch.Linear(
                child.in_features, 
                child.out_features, 
                bias=child.bias is not None
            ))
        else:
            replace_linear_layer_with_te_linear(child, full_name)
