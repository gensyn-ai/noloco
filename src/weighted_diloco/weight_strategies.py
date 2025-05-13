import logging

import torch
import torch.distributed as dist

from .btm_utils import logits_to_log_probs
from .data_loaders import ShardedDataset

_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.INFO)


class UniformStrategy:
    def __init__(self):
        self._weight = 1.0 / (dist.get_world_size())

    def __call__(self):
        return self._weight


class UniformStrategyBuilder:
    def __init__(self):
        pass

    def build(self, *args, **kwargs):
        return UniformStrategy()


class DomainLikelihoodWeightingStrategy:
    def __init__(
        self,
        path: str,
        model: torch.nn.Module,
        batch_size: int,
        number_of_batches: int,
        device: str,
        temperature: float,
    ):
        self.dataset = ShardedDataset(path, shard=False)
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=1
        )
        self.iterator = iter(self.data_loader)
        self.model = model
        self.number_of_batches = number_of_batches
        self.device = device
        self.temperature = temperature

    def __call__(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        log_probs = torch.tensor(0.0, device=self.device)
        tokens = torch.tensor(0.0, device=self.device)

        for _ in range(self.number_of_batches):
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.data_loader)
                batch = next(self.iterator)

            input_ids = batch["input_ids"].to(device=self.device)
            labels = batch["input_ids"].to(device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids)
                log_probs += logits_to_log_probs(logits, labels).sum()
                tokens += labels.shape[0] * labels.shape[1]

        log_expert_likelihoods = torch.zeros(world_size, device=self.device)
        with torch.no_grad():
            log_expert_likelihoods[rank] = log_probs / tokens
            dist.all_reduce(log_expert_likelihoods)

        weights = torch.nn.functional.softmax(
            log_expert_likelihoods / self.temperature, dim=0
        )

        _LOG.info(f"[{rank}]: {weights[rank].item()}.")
        return weights[rank].item()


class DomainLikelihoodWeightingStrategyBuilder:
    def __init__(self, path, batch_size, number_of_batches, temperature=1.0):
        self.path = path
        self.batch_size = batch_size
        self.number_of_batches = number_of_batches
        self.temperature = temperature

    def build(self, model, device):
        return DomainLikelihoodWeightingStrategy(
            path=self.path,
            model=model,
            batch_size=self.batch_size,
            number_of_batches=self.number_of_batches,
            device=device,
            temperature=self.temperature,
        )
