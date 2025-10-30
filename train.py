import time
from argparse import ArgumentParser
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from grpo import (
    classification_policy_adapter,
    rollout_tensor_with_mc_dropout,
    update_policy,
)
from optimizer import MemoryEfficientAdamW
from vision_task import VisionClassificationDataset, classification_reward
from vit_model import VisionTransformer, VisionTransformerConfig


def build_dataloader(config: dict, split: str) -> DataLoader:
    data_cfg = config["data"]
    dataset = VisionClassificationDataset(
        data_dir=data_cfg[f"{split}_dir"],
        image_size=data_cfg["image_size"],
        augment=split == "train" and data_cfg.get("augment", True),
        mean=data_cfg.get("mean", (0.485, 0.456, 0.406)),
        std=data_cfg.get("std", (0.229, 0.224, 0.225)),
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=split == "train",
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=data_cfg.get("pin_memory", False),
        collate_fn=VisionClassificationDataset.collate_fn,
        drop_last=split == "train",
    )
    return loader


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cpu":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def build_model(config: dict, device: torch.device) -> VisionTransformer:
    model_cfg = config["model"].copy()
    model_cfg.pop("device", None)
    dtype_str = model_cfg.pop("dtype", "float32")
    checkpoint_path = model_cfg.pop("checkpoint_path", None)
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)
    vit_config = VisionTransformerConfig(**model_cfg)
    model = VisionTransformer(vit_config)
    model.to(device=device, dtype=dtype)
    if checkpoint_path:
        state_dict = torch.load(Path(checkpoint_path), map_location=device)
        model.load_state_dict(state_dict)
    return model


def evaluate(
    model: VisionTransformer,
    dataloader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    for batch in dataloader:
        inputs = batch.inputs.to(device)
        targets = batch.targets.to(device)
        with autocast_context(device, dtype):
            logits = model(inputs)
        predictions = logits.argmax(dim=-1)
        correct += (predictions == targets).sum().item()
        total += targets.numel()
    if was_training:
        model.train()
    return correct / max(total, 1)


def main(config_path: str) -> None:
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle)

    device = torch.device(config["model"].get("device", "cpu"))
    dtype_str = config["model"].get("dtype", "float32")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_str, torch.float32)
    torch.random.manual_seed(config["training"]["random_seed"])

    train_loader = build_dataloader(config, split="train")
    eval_loader = build_dataloader(config, split="eval")

    model = build_model(config, device=device)
    model.train()
    optimizer = MemoryEfficientAdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=tuple(config["training"]["betas"]),
        enabled=config["training"].get("memory_efficient_adamw", True),
    )

    start_time = time.time()
    training_cfg = config["training"]
    num_mc_samples = training_cfg["num_mc_samples"]
    num_train_steps = training_cfg["num_train_steps"]
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=f"{training_cfg['log_dir']}/{current_time}")
    ckpt_dir = Path(training_cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_iter = iter(train_loader)
    for step in range(1, num_train_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        episodes = rollout_tensor_with_mc_dropout(
            model=model,
            batch=batch,
            num_samples=num_mc_samples,
            reward_function=classification_reward,
            device=device,
            dtype=dtype,
        )
        results = update_policy(
            model=model,
            optimizer=optimizer,
            episodes=episodes,
            micro_batch_size=training_cfg["micro_batch_size"],
            pad_token_id=None,
            max_grad_norm=training_cfg["max_grad_norm"],
            device=device,
            dtype=dtype,
            policy_adapter=classification_policy_adapter,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time

        rewards = np.array([episode.reward for episode in episodes], dtype=np.float32)
        accuracies = np.array(
            [episode.reward_info.get("accuracy", 0.0) for episode in episodes],
            dtype=np.float32,
        )
        mean_reward = float(rewards.mean())
        std_reward = float(rewards.std())
        mean_accuracy = float(accuracies.mean())
        grad_norm = results["grad_norm"]
        entropy = results["entropy"]
        lr = optimizer.param_groups[0]["lr"]
        loss = results["loss"]

        print(
            "\r"
            f"Step {step}, mean_reward: {mean_reward:.3f}, accuracy: {mean_accuracy:.3f}, "
            f"grad_norm: {grad_norm:.3f}, loss: {loss:.4f}, duration: {duration:.2f}s",
            end="",
            flush=True,
        )

        if step % training_cfg["eval_interval"] == 0:
            eval_accuracy = evaluate(model, eval_loader, device=device, dtype=dtype)
            print(f"\rEval accuracy: {eval_accuracy:.3f}" + " " * 80)
            tb_writer.add_scalar("accuracy/eval", eval_accuracy, step)

        tb_writer.add_scalar("loss", loss, step)
        tb_writer.add_scalar("reward/mean", mean_reward, step)
        tb_writer.add_scalar("reward/std", std_reward, step)
        tb_writer.add_scalar("accuracy/train", mean_accuracy, step)
        tb_writer.add_scalar("grad_norm", grad_norm, step)
        tb_writer.add_scalar("entropy", entropy, step)
        tb_writer.add_scalar("learning_rate", lr, step)
        tb_writer.add_scalar("iteration_duration", duration, step)

        if step % training_cfg["ckpt_save_interval"] == 0:
            ckpt_path = ckpt_dir / f"ckpt_{step:06d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"\nSaved checkpoint to {ckpt_path}")

    tb_writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)

