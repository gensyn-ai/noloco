import torch


class AdamOptimizer:
    def __init__(
        self,
        params,
        lr: float,
    ):
        self.params = params
        self.local_optimizer = torch.optim.Adam(
            params=params,
            lr=lr,
        )

    def step(self):
        self.local_optimizer.step()

    def zero_grad(self):
        self.local_optimizer.zero_grad()


class AdamOptimizerBuilder:
    def __init__(
        self,
        lr: float,
    ):
        self.lr = lr

    def build(self, model, *args, **kwargs):
        return AdamOptimizer(
            params=model.parameters(),
            lr=self.lr,
        )
