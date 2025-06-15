import numpy as np
import torch
import torch.distributed as dist


def create_random_permutation(worldsize, seed):
    rng = np.random.default_rng(seed + 1)
    return [int(x) for x in rng.permutation(worldsize)]


def find_group(groups):
    rank = dist.get_rank()
    for group in groups:
        if rank in group:
            return group
    assert False


def batched(ranks, size):
    assert len(ranks) % size == 0
    n = len(ranks) // size
    out = [[] for _ in range(n)]
    for i, x in enumerate(ranks):
        out[i % n].append(x)
    return out


def update_gradients(
    group: tuple[int] | list[int], p: torch.nn.parameter.Parameter, outer_lr: float
):
    root = group[0]
    rank = dist.get_rank()
    assert rank in group
    mean = torch.zeros_like(p)

    if rank == root:
        tmp = torch.zeros_like(p)
        with torch.no_grad():
            mean.add_(p)
            for irank in group[1:]:
                dist.recv(tmp, src=irank)
                mean.add_(tmp)
            mean.mul_(1.0 / len(group))
            for irank in group[1:]:
                dist.send(mean, dst=irank)
    else:
        dist.send(p, dst=root)
        dist.recv(mean, src=root)

    with torch.no_grad():
        g_ = 2.0 * (p - mean)
        p.grad.add_(outer_lr * g_)


class SparseOptimizer:
    def __init__(
        self,
        model,
        dp_group,
        local_group_size: int,
        number_of_local_steps: int,
        outer_lr: float,
        **kwargs,
    ):
        self.model = model
        self.dp_group = dp_group
        self.local_group_size = local_group_size
        assert dist.get_world_size(group=self.dp_group) % self.local_group_size == 0
        self.number_of_local_steps = number_of_local_steps
        self.outer_lr = outer_lr
        self.local_optimizer = torch.optim.Adam(params=model.parameters(), **kwargs)
        self._step = 0

    def step(self):
        self._step += 1

        if self._step % self.number_of_local_steps == 0:
            world_size = dist.get_world_size(group=self.dp_group)
            dp_ranks = [
                dist.get_global_rank(group=self.dp_group, group_rank=group_rank)
                for group_rank in create_random_permutation(world_size, self._step)
            ]
            groups = list(batched(dp_ranks, self.local_group_size))
            group = find_group(groups)

            for p in self.model.parameters():
                if p.requires_grad and p.grad is not None:
                    update_gradients(group, p, self.outer_lr)
        self.local_optimizer.step()

    def zero_grad(self):
        self.local_optimizer.zero_grad()


class SparseOptimizerBuilder:
    def __init__(self, number_of_local_steps, local_group_size, lr, outer_lr=0.7):
        self.number_of_local_steps = number_of_local_steps
        self.local_group_size = local_group_size
        self.lr = lr
        self.outer_lr = outer_lr

    def build(self, model, dp_group, *args):
        return SparseOptimizer(
            model,
            dp_group,
            local_group_size=self.local_group_size,
            number_of_local_steps=self.number_of_local_steps,
            outer_lr=self.outer_lr,
            lr=self.lr,
        )
