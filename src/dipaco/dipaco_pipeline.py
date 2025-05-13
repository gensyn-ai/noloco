from typing import Callable

import numpy as np
import torch
import torch.distributed as dist


def loss_fn(labels, logits, reduction="mean"):
    vocab_size = logits.shape[-1]
    return torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        reduction=reduction,
    ).unsqueeze(dim=0)


def create_random_permutation(worldsize, seed):
    rng = np.random.default_rng(seed + 1)
    return [int(x) for x in rng.permutation(worldsize)]


def from_tensor_to_list(batch: torch.Tensor):
    return [x.unsqueeze(dim=0) for x in batch.unbind(dim=0)]


def _get_global_rank(group, group_rank):
    if group is None:
        return group_rank
    else:
        return torch.distributed.get_global_rank(group=group, group_rank=group_rank)


def forward_maybe_receive_inputs(
    t: torch.Tensor, label: torch.Tensor, dp_group, pp_group, recv_group, seed, train
):
    dp_rank = torch.distributed.get_rank(group=dp_group)
    pp_rank = torch.distributed.get_rank(group=pp_group)
    dp_worlds_size = torch.distributed.get_world_size(group=dp_group)
    pp_worlds_size = torch.distributed.get_world_size(group=pp_group)

    # Generate the sending stages rank permutation
    seed_ = seed * pp_worlds_size + pp_rank - 1
    permutation = create_random_permutation(dp_worlds_size, seed_)

    recv_global_rank = None
    if pp_rank > 0:
        recv_local_rank = permutation.index(dp_rank)
        recv_global_rank = (
            recv_group[recv_local_rank]
            if train
            else _get_global_rank(group=pp_group, group_rank=pp_rank - 1)
        )
        device = t.device
        t_cpu = t.cpu()
        label_cpu = label.cpu()
        torch.distributed.recv(t_cpu, src=recv_global_rank)
        torch.distributed.recv(label_cpu, src=recv_global_rank)
        t = t_cpu.to(device=device)
        label = label_cpu.to(device=device)
    return t, label, recv_global_rank


def forward_maybe_send_outputs(
    t: torch.Tensor, label: torch.Tensor, dp_group, pp_group, send_group, seed, train
):
    dp_rank = torch.distributed.get_rank(group=dp_group)
    pp_rank = torch.distributed.get_rank(group=pp_group)
    dp_worlds_size = torch.distributed.get_world_size(group=dp_group)
    pp_worlds_size = torch.distributed.get_world_size(group=pp_group)

    seed_ = seed * pp_worlds_size + pp_rank
    permutation = create_random_permutation(dp_worlds_size, seed_)

    if pp_rank < pp_worlds_size - 1:
        send_local_rank = permutation[dp_rank]
        send_global_rank = (
            send_group[send_local_rank]
            if train
            else _get_global_rank(group=pp_group, group_rank=pp_rank + 1)
        )
        t_cpu = t.clone().detach().cpu()
        label_cpu = label.clone().detach().cpu()
        return (
            torch.distributed.isend(
                t_cpu,
                dst=send_global_rank,
            ),
            torch.distributed.isend(
                label_cpu,
                dst=send_global_rank,
            ),
            send_global_rank,
        )
    return None, None, None


def backward_maybe_receive_inputs(t: torch.Tensor, recv_rank):
    if recv_rank is not None:
        device = t.device
        t_cpu = t.cpu()
        torch.distributed.recv(
            t_cpu,
            src=recv_rank,
        )
        t = t_cpu.to(device=device)
    return t


def backward_maybe_send_inputs(t: torch.Tensor, send_rank):
    if send_rank is not None:
        t_cpu = t.clone().detach().cpu()
        return torch.distributed.isend(
            t_cpu,
            dst=send_rank,
        )
    return None


class DiPaCoMicroBatching(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        run_function: Callable[[torch.Tensor], torch.Tensor],
        batch: torch.Tensor,
        labels: torch.Tensor,
        dp_group: dist.ProcessGroup,
        pp_group: dist.ProcessGroup,
        send_group: list[int] | None,
        recv_group: list[int] | None,
        step: int,
        train: bool,
        use_fp8: bool = False, #warning - slower than bfloat16 unless the correct docker run flags are set
    ):
        batch_list = from_tensor_to_list(batch)
        label_list = from_tensor_to_list(labels)
        outputs = []
        ctx.orig_batch = batch
        ctx.label_list = []
        ctx.inputs = []
        ctx.recv_ranks = []
        ctx.send_ranks = []
        ctx.run_function = run_function
        ctx.dp_group = dp_group
        ctx.pp_group = pp_group
        ctx.send_group = send_group
        ctx.recv_group = recv_group
        requests = []

        has_loss = (
            dist.get_rank(group=pp_group) == dist.get_world_size(group=pp_group) - 1
        )
        ctx.has_loss = has_loss
        ctx.use_fp8 = use_fp8
        
        for index, micro_batch in enumerate(batch_list):
            label = label_list[index]
            x, label, recv_rank = forward_maybe_receive_inputs(
                micro_batch, label, dp_group, pp_group, recv_group, step + index, train
            )
            ctx.inputs.append(x)
            ctx.label_list.append(label)
            ctx.recv_ranks.append(recv_rank)
            with torch.no_grad():
                if has_loss:
                    y = loss_fn(label, run_function(x)) / len(label_list)
                else:
                    y = run_function(x)

                request, label_request, send_rank = forward_maybe_send_outputs(
                    y, label, dp_group, pp_group, send_group, step + index, train
                )
                requests.append(request)
                requests.append(label_request)
            outputs.append(y)
            ctx.send_ranks.append(send_rank)
        out = torch.cat(outputs, dim=0)
        for request in requests:
            if request is not None:
                request.wait()
        return out

    @staticmethod
    def backward(ctx, grad_outputs):
        requests = []
        graph_outputs = [x.unsqueeze(dim=0) for x in grad_outputs.unbind(dim=0)]

        for tensor_input, label, grad_output, recv_rank, send_rank in zip(
            ctx.inputs, ctx.label_list, graph_outputs, ctx.send_ranks, ctx.recv_ranks
        ):
            grad_output = backward_maybe_receive_inputs(grad_output, recv_rank)
            tensor_input.requires_grad = True
            with torch.enable_grad():
                if ctx.has_loss:
                    output = loss_fn(label, ctx.run_function(tensor_input)) / len(
                        ctx.label_list
                    )
                else:
                    output = ctx.run_function(tensor_input)#
            output.backward(gradient=grad_output)
            requests.append(backward_maybe_send_inputs(tensor_input.grad, send_rank))
        for request in requests:
            if request is not None:
                request.wait()

        return (
            None,
            torch.zeros_like(ctx.orig_batch),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )
