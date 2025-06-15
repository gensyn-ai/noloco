import logging
import os
import time

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from noloco.create_process_groups import create_process_groups


def instantiate_and_set(params, dp_group):
    dataset = instantiate(params)
    dataset.pg = dp_group
    return dataset


def eval_weight_variance(model, device, dp_group, pp_group):
    dp_world_size = dist.get_world_size(group=dp_group)

    v = torch.tensor(0.0, device=device, requires_grad=False)
    weights = torch.tensor(0.0, device=device, requires_grad=False)
    for p in model.parameters():
        with torch.no_grad():
            mean = torch.zeros_like(p)
            mean += p
            dist.all_reduce(mean, group=dp_group)
            mean /= dp_world_size
            v += torch.pow(mean - p, 2.0).sum()
            weights += p.numel() * dp_world_size

    dist.all_reduce(v)
    dist.all_reduce(weights, group=pp_group)
    return torch.sqrt(v / weights)


def get_eval_ppl(model, dev_data_loader, device, dp_group, pp_group, eval_steps=-1):
    model.eval()
    dev_loss = torch.tensor(0.0, device=device, requires_grad=False)
    samples = torch.tensor(0.0, device=device, requires_grad=False)

    pp_rank = dist.get_rank(group=pp_group)
    pp_world_size = dist.get_world_size(group=pp_group)

    for step, batch in enumerate(dev_data_loader):
        input_ids = batch["input_ids"].to(device=device)
        labels = batch["label"].to(device=device)
        with torch.no_grad():
            samples += labels.shape[0]
            loss = model(input_ids, labels, train=False)
            if pp_rank == pp_world_size - 1:
                dev_loss += loss.sum() * labels.shape[0]

        if eval_steps > 0 and step >= eval_steps:
            break
    if pp_rank == pp_world_size - 1:
        dist.all_reduce(dev_loss, group=dp_group)
        dist.all_reduce(samples, group=dp_group)
    model.train()
    return torch.exp(dev_loss / samples)


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = (
        f"cuda:{local_rank % torch.cuda.device_count()}"
        if torch.cuda.is_available()
        else "cpu"
    )
    #torch.set_default_device(device)
    #torch.cuda.set_device(local_rank % torch.cuda.device_count())

    log_dir = os.path.join(cfg.log_dir, cfg.exp_name)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    backend = "gloo"  # "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(
        backend=backend,
        device_id=torch.device(device),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dp_world_size = cfg.dp_world_size
    assert world_size % dp_world_size == 0

    pp_world_size = world_size // dp_world_size

    # Create process groups.
    (dp_group, pp_group, send_group, recv_group) = create_process_groups(
        (dp_world_size, pp_world_size)
    )

    dp_rank = dist.get_rank(group=dp_group)
    pp_rank = dist.get_rank(group=pp_group)
    has_loss = pp_rank == pp_world_size - 1

    if rank == 0:
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
    dist.barrier()
    file_handler = logging.FileHandler(os.path.join(log_dir, f"training_{rank}.log"))
    logger.addHandler(file_handler)

    if rank == 0:
        logger.info(OmegaConf.to_yaml(cfg))
        logger.info(f"Using communication backend: {backend}.")
    dist.barrier()

    tb_root_path = os.path.join(cfg.log_dir, "tensorboard")
    tb_log_path = os.path.join(tb_root_path, cfg.exp_name)
    checkpoint_path = os.path.join(log_dir, "checkpoints")
    if rank == 0:
        logger.info("Creating folders ...")
        if not os.path.isdir(tb_root_path):
            os.makedirs(tb_root_path, exist_ok=True)
        if not os.path.isdir(tb_log_path):
            os.makedirs(tb_log_path, exist_ok=True)
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
    dist.barrier()
    writer = SummaryWriter(log_dir=tb_log_path) if dp_rank == 0 and has_loss else None

    train_dataset = instantiate(cfg.train_data_loader)
    train_dataset.pg = dp_group
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, num_workers=1, drop_last=True
    )
    validation_data_loaders = {
        key: torch.utils.data.DataLoader(
            instantiate_and_set(value, dp_group),
            batch_size=cfg.batch_size,
            num_workers=0,
        )
        for key, value in cfg.validation_data_loaders.items()
    }
    model_builder = instantiate(cfg.model)
    model = model_builder.build(dp_group, pp_group, send_group, recv_group, cfg.transformer_engine)
    model.to(device=device)
    for p in model.parameters():
        with torch.no_grad():
            p.data.mul_(1.0 / dist.get_world_size(group=dp_group))
            dist.all_reduce(p.data, group=dp_group)

    optimizer_builder = instantiate(cfg.optimizer)
    optimizer = optimizer_builder.build(model, dp_group, device)

    scheduler_builder = instantiate(cfg.scheduler)
    scheduler = scheduler_builder.build(optimizer.local_optimizer)

    train_loss = torch.tensor(0.0, device=device, requires_grad=False)

    num_steps = cfg.num_steps
    eval_steps = cfg.eval_steps
    evalulation_interval = cfg.evalulation_interval

    step = 0
    time_spent = 0.0
    model.train()
    file_handler.flush()

    while step < num_steps:
        for batch in train_data_loader:
            tick = time.monotonic()
            input_ids = batch["input_ids"].to(device=device)
            labels = batch["label"].to(device=device)
            optimizer.zero_grad()
            loss = model(input_ids, labels)
            loss.backward(torch.ones_like(loss))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            tock = time.monotonic()

            time_spent += tock - tick
            if has_loss:
                with torch.no_grad():
                    train_loss += loss.sum()

            if step % evalulation_interval == 0:
                dev_losses = {
                    key: get_eval_ppl(
                        model,
                        validation_data_loader,
                        device,
                        dp_group,
                        pp_group,
                        eval_steps,
                    )
                    for key, validation_data_loader in validation_data_loaders.items()
                }
                weight_std = eval_weight_variance(
                    model, device, dp_group, pp_group
                ).item()

                if has_loss:
                    dist.all_reduce(train_loss, group=dp_group)
                if writer is not None:
                    train_loss_log = (
                        train_loss.item() / dp_world_size / evalulation_interval
                    )
                    lr = optimizer.local_optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"[{step}/{num_steps}] "
                        f"train_loss: {train_loss_log:.2f} "
                        f"time per step: {(time_spent / evalulation_interval):.2f} "
                        f"lr: {lr:.6f} "
                        + f"std: {weight_std:.6f} "
                        + " ".join(
                            [
                                f"{key}_ppl: {(dev_loss.item()):.1f}"
                                for key, dev_loss in dev_losses.items()
                            ]
                        )
                    )
                    file_handler.flush()

                    writer.add_scalar("Loss/train", train_loss_log, step)
                    for key, dev_loss in dev_losses.items():
                        writer.add_scalar(f"Loss/{key}_ppl", dev_loss.item(), step)
                    writer.add_scalar(
                        "Time/step", time_spent / evalulation_interval, step
                    )
                    writer.add_scalar("Model/std", weight_std, step)

                time_spent = 0.0
                train_loss.mul_(0.0)

            if step >= num_steps:
                break

    # torch.save(model.state_dict(), os.path.join(checkpoint_path, f"model_{rank}.bin"))

    torch.distributed.destroy_process_group()
    logging.info("Exited successfully.")
    file_handler.flush()


if __name__ == "__main__":
    main()
