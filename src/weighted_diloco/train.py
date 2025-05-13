import logging
import os
import time

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter


def loss_fn(labels, logits, reduction="mean"):
    vocab_size = logits.shape[-1]
    return torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        reduction=reduction,
    )


def get_eval_ppl(model, dev_data_loader, device, eval_steps=-1):
    model.eval()
    dev_loss = torch.tensor(0.0, device=device)
    tokens = torch.tensor(0.0, device=device)
    for step, batch in enumerate(dev_data_loader):
        input_ids = batch["input_ids"].to(device=device)
        labels = batch["label"].to(device=device)
        with torch.no_grad():
            tokens += labels.shape[0] * labels.shape[1]
            dev_loss += loss_fn(labels, model(input_ids), reduction="sum")

        if eval_steps > 0 and step >= eval_steps:
            break

    dist.reduce(dev_loss, dst=0)
    dist.reduce(tokens, dst=0)
    model.train()
    return torch.exp(dev_loss / tokens)


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    torch.cuda.set_device(local_rank)

    log_dir = os.path.join(cfg.log_dir, cfg.exp_name)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(
        backend=backend,
        device_id=torch.device(device),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
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
            os.mkdir(tb_root_path)
        if not os.path.isdir(tb_log_path):
            os.mkdir(tb_log_path)
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)

    writer = SummaryWriter(log_dir=tb_log_path) if rank == 0 else None

    train_data_loader = torch.utils.data.DataLoader(
        instantiate(cfg.train_data_loader), batch_size=cfg.batch_size, num_workers=1
    )
    validation_data_loaders = {
        key: torch.utils.data.DataLoader(
            instantiate(value), batch_size=cfg.batch_size, num_workers=1
        )
        for key, value in cfg.validation_data_loaders.items()
    }
    model = instantiate(cfg.model).to(device=device)
    with torch.no_grad():
        for p in model.parameters():
            p *= 1.0 / dist.get_world_size()
            dist.all_reduce(p)

    weight_strategy_builder = instantiate(cfg.weight_strategy)
    weight_strategy = weight_strategy_builder.build(model, device)

    optimizer_builder = instantiate(cfg.optimizer)
    optimizer = optimizer_builder.build(model, weight_strategy, device)

    scheduler_builder = instantiate(cfg.scheduler)
    scheduler = scheduler_builder.build(optimizer.local_optimizer)

    train_loss = torch.tensor(0.0, device=device)

    num_steps = cfg.num_steps
    eval_steps = cfg.eval_steps
    evalulation_interval = cfg.evalulation_interval
    gradient_accumulation = cfg.gradient_accumulation

    step = 0
    accumulation_step = 0

    time_spent = 0.0
    model.train()

    file_handler.flush()

    while step < num_steps:
        for batch in train_data_loader:
            tick = time.monotonic()
            input_ids = batch["input_ids"].to(device=device)
            labels = batch["label"].to(device=device)
            if accumulation_step == 0:
                optimizer.zero_grad()
            loss = loss_fn(labels, model(input_ids))
            loss.backward()
            accumulation_step += 1
            if accumulation_step == gradient_accumulation:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad /= gradient_accumulation
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                accumulation_step = 0
                step += 1
            tock = time.monotonic()

            time_spent += tock - tick
            with torch.no_grad():
                train_loss += loss

            if accumulation_step == 0 and step % evalulation_interval == 0:
                dev_losses = {
                    key: get_eval_ppl(model, validation_data_loader, device, eval_steps)
                    for key, validation_data_loader in validation_data_loaders.items()
                }
                dist.reduce(train_loss, dst=0)
                if writer is not None:
                    train_loss_log = (
                        train_loss.item()
                        / world_size
                        / evalulation_interval
                        / gradient_accumulation
                    )
                    lr = optimizer.local_optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"[{step}/{num_steps}] "
                        f"train_loss: {train_loss_log:.2f} "
                        f"time per step: {(time_spent / evalulation_interval):.2f} "
                        f"lr: {lr} "
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
                    torch.save(
                        model, os.path.join(checkpoint_path, f"model_{step}.bin")
                    )

                time_spent = 0.0
                train_loss = torch.tensor(0.0, device=device)

            if step >= num_steps:
                break

    torch.distributed.destroy_process_group()
    logging.info("Exited successfully.")
    file_handler.flush()


if __name__ == "__main__":
    main()
