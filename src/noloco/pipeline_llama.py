import torch
import torch.distributed as dist
from transformers import LlamaConfig, LlamaForCausalLM

from noloco.dipaco_pipeline import DiPaCoMicroBatching


class LLamaLayerWrapper(torch.nn.Module):
    def __init__(self, layer, rotary_emb):
        super().__init__()
        self.decoder_layer = layer
        self.rotary_emb = rotary_emb

    def forward(self, hidden_states):
        cache_position = torch.arange(
            hidden_states.shape[1], device=hidden_states.device
        )
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        layer_outputs = self.decoder_layer(
            hidden_states,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        return layer_outputs[0]


class DiPaCoLlama(torch.nn.Module):
    def __init__(self, config, dp_group, pp_group, send_group, recv_group, transformer_engine=False):
        super().__init__()
        _model = LlamaForCausalLM(config)
        self.pp_rank = dist.get_rank(group=pp_group)
        self.pp_world_size = dist.get_world_size(group=pp_group)
        self.dp_group = dp_group
        self.pp_group = pp_group
        self.send_group = send_group
        self.recv_group = recv_group
        self.hidden_size = config.hidden_size

        num_hidden_layers = config.num_hidden_layers
        assert num_hidden_layers % self.pp_world_size == 0
        layers_per_worker = num_hidden_layers // self.pp_world_size
        self.rotary_emb = _model.model.rotary_emb
        self.input_embeddings: torch.nn.Module | None = None
        self.has_loss = False

        layers = []
        if self.pp_rank == 0:
            self.input_embeddings = _model.get_input_embeddings()
        for i in range(layers_per_worker):
            index = i + layers_per_worker * self.pp_rank
            layers.append(
                LLamaLayerWrapper(_model.model.layers[index], self.rotary_emb)
            )
        if self.pp_rank == self.pp_world_size - 1:
            layers.append(_model.model.norm)
            layers.append(_model.get_output_embeddings())
            self.has_loss = True
        self._model = torch.nn.Sequential(*layers)
        self._step = 0
        self.to(dtype=torch.bfloat16)
        self.transformer_engine = transformer_engine

    def forward(self, input_ids, labels, train=True):
        if dist.get_rank(self.pp_group) == 0:
            assert self.input_embeddings is not None
            inputs = self.input_embeddings(input_ids)
            output = DiPaCoMicroBatching.apply(
                self._model,
                inputs,
                labels,
                self.dp_group,
                self.pp_group,
                self.send_group,
                self.recv_group,
                self._step,
                train,
                self.transformer_engine,
            )
        else:
            inputs = torch.zeros(
                input_ids.shape[0],
                input_ids.shape[1],
                self.hidden_size,
                dtype=torch.bfloat16,
                device=input_ids.device,
                requires_grad=True,
            )
            output = DiPaCoMicroBatching.apply(
                self._model,
                inputs,
                labels,
                self.dp_group,
                self.pp_group,
                self.send_group,
                self.recv_group,
                self._step,
                train,
                self.transformer_engine,
            )
        if train:
            self._step += input_ids.shape[0]
        return output


class DiPaCoLlamaBuilder:
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        attn_implementation: str = "flash_attention_2",
        vocab_size=128002,  # matches "meta-llama/Meta-Llama-3-8B" (128000 + 2)
    ):
        self.config = LlamaConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            torch_dtype=torch.bfloat16,
        )
        self.config._attn_implementation = attn_implementation

    def build(self, dp_group, pp_group, send_group, recv_group, transformer_engine=False):
        return DiPaCoLlama(
            self.config,
            dp_group,
            pp_group,
            send_group,
            recv_group,
            transformer_engine=transformer_engine,
        )
