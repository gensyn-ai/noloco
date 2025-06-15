import torch


def logits_to_log_probs(logits: torch.Tensor, labels: torch.Tensor):
    """Compute log probabilities from logits.

    :param logits: tensor with shape [batch size, ..., vocabulary size].
    :param labels: integer tensor with shape [batch size, ...]
    """
    vocab_size = logits.shape[-1]
    return -torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size), labels.view(-1), reduction="sum"
    )
