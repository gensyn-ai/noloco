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


def outer_step(group, model, params, momentum, outer_lr, outer_momentum):
    root = group[0]
    rank = dist.get_rank()
    assert rank in group

    for p, p_old, m in zip(model.parameters(), params, momentum):
        p_old_cpu = p_old.clone().detach().cpu()
        mean = torch.zeros_like(p).to(device="cpu")
        mean_g = (p - p_old).cpu()

        if rank == root:
            tmp = torch.zeros_like(p).to(device="cpu")
            with torch.no_grad():
                mean.add_(p_old_cpu)
                for irank in group[1:]:
                    dist.recv(tmp, src=irank)
                    mean.add_(tmp)
                    dist.recv(tmp, src=irank)
                    mean_g.add_(tmp)
                mean.mul_(1.0 / len(group))
                mean_g.mul_(1.0 / len(group))
                for irank in group[1:]:
                    dist.send(mean, dst=irank)
                    dist.send(mean_g, dst=irank)
        else:
            dist.send(p_old_cpu, dst=root)
            dist.send((p - p_old).cpu(), dst=root)
            dist.recv(mean, src=root)
            dist.recv(mean_g, src=root)
        with torch.no_grad():
            m_ = (
                (outer_momentum - 1.0) * m
                + outer_lr * mean_g.to(device=m.device)
                + outer_lr * (mean.to(device=m.device) - p_old)
            )
            m.add_(m_)
            p.mul_(0.0)
            p_old.add_(m)
            p.add_(p_old)


class SparseOptimizerC:
    def __init__(
        self,
        model,
        dp_group,
        local_group_size: int,
        number_of_local_steps: int,
        outer_lr: float,
        outer_momentum: float,
        **kwargs,
    ):
        self.model = model
        self.dp_group = dp_group
        self.local_group_size = local_group_size
        assert dist.get_world_size(group=self.dp_group) % self.local_group_size == 0
        self.number_of_local_steps = number_of_local_steps
        self.outer_lr = outer_lr
        self.outer_momentum = outer_momentum
        self.local_optimizer = torch.optim.Adam(params=model.parameters(), **kwargs)
        # Prior outer step model weights.
        self.step_params = [p.clone().detach() for p in model.parameters()]

        # Global momentum for outer optimizer.
        self.momentum = [
            torch.zeros_like(p, requires_grad=False) for p in model.parameters()
        ]
        self._step = 0
        self._group = None

    def step(self):
        self.local_optimizer.step()
        self._step += 1

        if self._step % self.number_of_local_steps == 0:
            world_size = dist.get_world_size(group=self.dp_group)
            dp_ranks = [
                dist.get_global_rank(group=self.dp_group, group_rank=group_rank)
                for group_rank in create_random_permutation(world_size, self._step)
            ]
            groups = list(batched(dp_ranks, self.local_group_size))
            group = find_group(groups)
            outer_step(
                group,
                self.model,
                self.step_params,
                self.momentum,
                self.outer_lr,
                self.outer_momentum,
            )

    def zero_grad(self):
        self.local_optimizer.zero_grad()


class SparseOptimizerCBuilder:
    def __init__(
        self,
        number_of_local_steps,
        local_group_size,
        lr,
        outer_momentum=0.9,
        outer_lr=0.7,
    ):
        self.number_of_local_steps = number_of_local_steps
        self.local_group_size = local_group_size
        self.lr = lr
        self.outer_lr = outer_lr
        self.outer_momentum = outer_momentum

    def build(self, model, dp_group, *args):
        return SparseOptimizerC(
            model,
            dp_group,
            local_group_size=self.local_group_size,
            number_of_local_steps=self.number_of_local_steps,
            outer_lr=self.outer_lr,
            outer_momentum=self.outer_momentum,
            lr=self.lr,
        )
