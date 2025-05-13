import torch
import torch.distributed as dist


class DataParallelOptimizer:
    def __init__(
        self,
        params,
        weight_strategy,
        **kwargs,
    ):
        self.params = params
        self.weight_strategy = weight_strategy
        self.local_optimizer = torch.optim.Adam(params=params, **kwargs)

    def step(self):
        requests = []
        w = self.weight_strategy()
        for p in self.params:
            if p.requires_grad and p.grad is not None:
                with torch.no_grad():
                    p.grad *= w
                requests.append(dist.all_reduce(p.grad, async_op=True))
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

    def build(self, model, weight_strategy, device):
        return DataParallelOptimizer(model.parameters(), weight_strategy, lr=self.lr)
