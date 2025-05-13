import torch
import torch.distributed as dist
from transformers import LlamaConfig, LlamaForCausalLM


class LlamaWrapper(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        attn_implementation: str = "flash_attention_2",
        vocab_size=128002,  # matches "meta-llama/Meta-Llama-3-8B" (128000 + 2)
    ):
        super().__init__()
        config = LlamaConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=4 * hidden_size,
            vocab_size=vocab_size,
            torch_dtype=torch.bfloat16,
        )
        config._attn_implementation = attn_implementation
        self._model = LlamaForCausalLM(config).to(dtype=torch.bfloat16)

    def forward(self, input_ids: torch.Tensor):
        outputs = self._model(input_ids=input_ids)
        return outputs.logits


# additional parameter of intermediate_size
# That will allow us to fix hidden_size and manipulate intermediate_size for different model sizes
class LlamaWrapperWithMLPSize(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        attn_implementation: str = "flash_attention_2",
        vocab_size=128002,  # matches "meta-llama/Meta-Llama-3-8B" (128000 + 2)
    ):
        super().__init__()
        config = LlamaConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            torch_dtype=torch.bfloat16,
        )
        config._attn_implementation = attn_implementation
        self._model = LlamaForCausalLM(config).to(dtype=torch.bfloat16)

    def forward(self, input_ids: torch.Tensor):
        outputs = self._model(input_ids=input_ids)
        return outputs.logits
