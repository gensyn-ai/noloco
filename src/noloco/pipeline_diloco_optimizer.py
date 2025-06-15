import torch
import torch.distributed as dist


class PipelineDiLoCoOptimizer:
    def __init__(
        self,
        model,
        number_of_local_steps: int,
        lr: float,
        device,
        outer_lr: float,
        outer_momentum: float,
        dp_group,
    ):
        self.model = model
        self.local_optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=lr,
        )
        self.number_of_local_steps = number_of_local_steps
        self._step: int = 0
        self.device = device

        # Prior outer step model weights.
        self.global_param = [torch.zeros_like(p, requires_grad=False) for p in model.parameters()]
        for gp, p in zip(self.global_param,  model.parameters()):
            gp.add_(p)

        # Global gradient.
        self.global_param_grad = [
            torch.zeros_like(p, requires_grad=False) for p in model.parameters()
        ]

        # Global momentum for outer optimizer.
        self.global_momentum = [
            torch.zeros_like(p, requires_grad=False) for p in model.parameters()
        ]
        self.outer_lr = outer_lr
        self.outer_momentum = outer_momentum
        self.dp_group = dp_group
        self.world_size = dist.get_world_size(group=dp_group)

    def step(self):
        self._step += 1
        self.local_optimizer.step()

        if self._step % self.number_of_local_steps == 0:
            # Perform the global optimizer step.
            # Average model weights.
            requests = []
            optimizer_requests = []
            with torch.no_grad():
                w = 1.0 / self.world_size
                for grad_gp, gp, p, gm in zip(
                    self.global_param_grad,
                    self.global_param,
                    self.model.parameters(),
                    self.global_momentum,
                ):
                    grad_gp.mul_(0.0)
                    grad_gp.add_(w * (gp - p))
                    requests.append(
                        (
                            dist.all_reduce(
                                grad_gp, async_op=True, group=self.dp_group
                            ),
                            p,
                            gp,
                            grad_gp,
                            gm,
                        )
                    )

            # Average optimizer states.
            #state_dict = self.local_optimizer.state_dict()
            #optimizer_states = state_dict["state"]
            #if optimizer_states:
            #    for _, value in optimizer_states.items():
            #        exp_avg = value["exp_avg"]
            #        exp_avg.mul_(w)
            #        exp_avg_sq = value["exp_avg_sq"]
            #        exp_avg_sq.mul_(w)
            #        optimizer_requests.append(
            #            dist.all_reduce(exp_avg, async_op=True, group=self.dp_group)
            #        )
            #        optimizer_requests.append(
            #            dist.all_reduce(exp_avg_sq, async_op=True, group=self.dp_group)
            #        )

            with torch.no_grad():
                for request, p, gp, grad_gp, gm in requests:
                    # Nesterov momentum optimizer.
                    request.wait()
                    gm.add_((self.outer_momentum - 1.0) * gm - self.outer_lr * grad_gp)
                    gp.add_(gm)
                    p.mul_(0.0)
                    p.add_(gp)

            for request in optimizer_requests:
                request.wait()
            #self.local_optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        self.local_optimizer.zero_grad()


class PipelineDiLoCoOptimizerBuilder:
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

    def build(self, model, dp_group, device):
        return PipelineDiLoCoOptimizer(
            model=model,
            number_of_local_steps=self.number_of_local_steps,
            lr=self.lr,
            device=device,
            outer_lr=self.outer_lr,
            outer_momentum=self.outer_momentum,
            dp_group=dp_group,
        )
