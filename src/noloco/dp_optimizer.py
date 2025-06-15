import torch
import torch.distributed as dist


class DataParallelOptimizer:
    def __init__(
        self,
        model,
        dp_group,
        **kwargs,
    ):
        self.model = model
        self.dp_group = dp_group
        self.local_optimizer = torch.optim.Adam(params=model.parameters(), **kwargs)

    def step(self):
        requests = []
        w = 1.0 / dist.get_world_size(group=self.dp_group)
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                with torch.no_grad():
                    p.grad.mul_(w)
                requests.append(
                    dist.all_reduce(p.grad, async_op=True, group=self.dp_group)
                )
        for request in requests:
            request.wait()
        self.local_optimizer.step()

    def zero_grad(self):
        self.local_optimizer.zero_grad()


class DataParallelOptimizerBuilder:
    def __init__(
        self,
        lr,
    ):
        self.lr = lr

    def build(self, model, dp_group, *args):
        return DataParallelOptimizer(model, dp_group, lr=self.lr)
