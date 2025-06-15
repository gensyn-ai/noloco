import torch
import torch.distributed as dist


class DiLoCoOptimizer:
    def __init__(
        self,
        params,
        number_of_local_steps: int,
        lr: float,
        weight_strategy,
        device,
        outer_lr: float,
        outer_momentum: float,
    ):
        self.params = params
        self.local_optimizer = torch.optim.Adam(
            params=params,
            lr=lr,
        )
        self.number_of_local_steps = number_of_local_steps
        self.weight_strategy = weight_strategy
        self._step: int = 0
        self.device = device

        # Prior outer step model weights.
        self.global_param = [p.clone().detach() for p in params]

        # Global gradient.
        self.global_param_grad = [
            torch.zeros_like(p, requires_grad=False) for p in params
        ]

        # Global momentum for outer optimizer.
        self.global_momentum = [
            torch.zeros_like(p, requires_grad=False) for p in params
        ]
        self.outer_lr = outer_lr
        self.outer_momentum = outer_momentum

    def step(self):
        self._step += 1
        self.local_optimizer.step()

        if self._step % self.number_of_local_steps == 0:
            # Perform the global optimizer step.
            # Average model weights.
            requests = []
            optimizer_requests = []
            with torch.no_grad():
                w = self.weight_strategy()
                for grad_gp, gp, p, gm in zip(
                    self.global_param_grad,
                    self.global_param,
                    self.params,
                    self.global_momentum,
                ):
                    grad_gp += w * (gp - p) - grad_gp
                    requests.append(
                        (dist.all_reduce(grad_gp, async_op=True), p, gp, grad_gp, gm)
                    )

            # Average optimizer states.
            state_dict = self.local_optimizer.state_dict()
            optimizer_states = state_dict["state"]
            if optimizer_states:
                for _, value in optimizer_states.items():
                    exp_avg = value["exp_avg"]
                    exp_avg *= w
                    exp_avg_sq = value["exp_avg_sq"]
                    exp_avg_sq *= w
                    optimizer_requests.append(dist.all_reduce(exp_avg, async_op=True))
                    optimizer_requests.append(
                        dist.all_reduce(exp_avg_sq, async_op=True)
                    )

            with torch.no_grad():
                for request, p, gp, grad_gp, gm in requests:
                    request.wait()
                    gm += (self.outer_momentum - 1.0) * gm - self.outer_lr * grad_gp
                    p += gp + gm - p
                    gp += p - gp

            for request in optimizer_requests:
                request.wait()
            self.local_optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        self.local_optimizer.zero_grad()


class DiLoCoOptimizerBuilder:
    def __init__(
        self,
        number_of_local_steps: int,
        lr: float,
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
    ):
        self.number_of_local_steps = number_of_local_steps
        self.lr = lr
        self.outer_lr = outer_lr
        self.outer_momentum = outer_momentum

    def build(self, model, weight_strategy, device):
        return DiLoCoOptimizer(
            params=model.parameters(),
            number_of_local_steps=self.number_of_local_steps,
            lr=self.lr,
            weight_strategy=weight_strategy,
            device=device,
            outer_lr=self.outer_lr,
            outer_momentum=self.outer_momentum,
        )
