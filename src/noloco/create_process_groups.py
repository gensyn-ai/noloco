"""Module for distributed training related utility functions."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch.distributed as dist


def print_group(pg):
    global_ranks = [
        dist.get_global_rank(group=pg, group_rank=group_rank)
        for group_rank in range(dist.get_world_size(group=pg))
    ]
    print(f"[{dist.get_rank()}] {global_ranks=}")


def _group_sizes(world_sizes):
    group_size = 1
    group_sizes = [1] * len(world_sizes)
    for dim in range(len(world_sizes) - 1, -1, -1):
        group_sizes[dim] = group_size
        group_size *= world_sizes[dim]
    return group_sizes


def _sub_ranks(global_rank: int, group_sizes: Sequence[int]) -> Tuple[int, ...]:
    sub_ranks = [0] * len(group_sizes)
    for dim, group_size in enumerate(group_sizes):
        sub_ranks[dim] = global_rank // group_size
        global_rank -= sub_ranks[dim] * group_size
    return tuple(sub_ranks)


def _get_group(global_rank, dim, group_sizes, world_sizes, all_output_process_groups):
    if global_rank < 0:
        return None
    if global_rank >= dist.get_world_size():
        return None
    sub_ranks = _sub_ranks(global_rank, group_sizes)
    indices = _remainders(sub_ranks, world_sizes)
    output_process_groups = all_output_process_groups[dim][indices[dim]]
    return output_process_groups


def _loop_over_ranks(skip_dim, world_sizes, group_sizes, dim=0):
    if dim >= len(world_sizes):
        yield 0
    else:
        if dim == skip_dim:
            yield from _loop_over_ranks(skip_dim, world_sizes, group_sizes, dim + 1)
        else:
            for rank in range(world_sizes[dim]):
                for remainder in _loop_over_ranks(
                    skip_dim, world_sizes, group_sizes, dim + 1
                ):
                    yield rank * group_sizes[dim] + remainder


def _remainders(sub_ranks, world_sizes):
    remainders = [0] * len(world_sizes)
    for i in range(len(world_sizes)):
        group_size = 1
        for j in range(len(world_sizes) - 1, -1, -1):
            if i == j:
                continue
            remainders[i] += sub_ranks[j] * group_size
            group_size *= world_sizes[j]
    return remainders


def _loop_over_dimensions(
    share_parameters,
    world_sizes,
    group_sizes,
    use_shared,
    dim=0,
):
    if dim >= len(world_sizes):
        yield 0
    else:
        if share_parameters[dim] != use_shared:
            yield from _loop_over_dimensions(
                share_parameters, world_sizes, group_sizes, use_shared, dim + 1
            )
        else:
            for rank in range(world_sizes[dim]):
                for remainder in _loop_over_dimensions(
                    share_parameters, world_sizes, group_sizes, use_shared, dim + 1
                ):
                    yield rank * group_sizes[dim] + remainder


def create_process_groups(
    world_sizes: Tuple[int, int],
    pg_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[
    dist.ProcessGroup,
    dist.ProcessGroup,
    dist.ProcessGroup | None,
    dist.ProcessGroup | None,
]:
    """Create process groups for variable number of sub world sizes.

    Arguments:
        world_sizes: world sizes for each process group.
        share_parameters: if the process group shares parameters with other groups.
        pg_overrides: optional process group arguments.

    Returns:
        A tuple containing list of process groups for each world_size input and optimizer process
        group.
    """
    pg_overrides = pg_overrides or {}
    group_sizes = _group_sizes(world_sizes)
    global_rank = dist.get_rank()

    # Construct the process groups for specified world sizes.
    all_output_process_groups: List[Any] = [None] * len(world_sizes)
    all_output_ranks: List[Any] = [None] * len(world_sizes)
    for dim in range(len(world_sizes)):
        all_output_ranks[dim] = [
            [
                sub_rank * group_sizes[dim] + global_rank_remainder
                for sub_rank in range(world_sizes[dim])
            ]
            for global_rank_remainder in _loop_over_ranks(dim, world_sizes, group_sizes)
        ]

        all_output_process_groups[dim] = [
            dist.new_group(
                ranks=[
                    sub_rank * group_sizes[dim] + global_rank_remainder
                    for sub_rank in range(world_sizes[dim])
                ],
                **pg_overrides,
            )
            for global_rank_remainder in _loop_over_ranks(dim, world_sizes, group_sizes)
        ]

    dp_group = _get_group(
        global_rank, 0, group_sizes, world_sizes, all_output_process_groups
    )
    pp_group = _get_group(
        global_rank, 1, group_sizes, world_sizes, all_output_process_groups
    )

    dp_rank = dist.get_rank(group=dp_group)
    pp_rank = dist.get_rank(group=pp_group)

    # if dist.get_rank() == 0:
    #    for pg in all_output_ranks[0]:
    #        print(f"[{dist.get_rank()}]  {pg}")
    # dist.barrier()
    # print(f"[{dist.get_rank()}] {dp_rank=}  {pp_rank=}")
    # dist.barrier()

    if pp_rank > 0:
        prev_global_rank = dist.get_global_rank(pp_group, pp_rank - 1)
        recv_ranks = _get_group(
            prev_global_rank, 0, group_sizes, world_sizes, all_output_ranks
        )
        # print(f"[{dist.get_rank()}] {prev_global_rank=} {recv_ranks=}")
    else:
        recv_ranks = None

    if pp_rank < world_sizes[1] - 1:
        next_global_rank = dist.get_global_rank(pp_group, pp_rank + 1)
        send_ranks = _get_group(
            next_global_rank, 0, group_sizes, world_sizes, all_output_ranks
        )
        # print(f"[{dist.get_rank()}] {next_global_rank=} {send_ranks=}")

    else:
        send_ranks = None

    return dp_group, pp_group, send_ranks, recv_ranks
