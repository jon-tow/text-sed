import argparse
import logging
import os
import random
from typing import *

import datasets
import numpy as np
import omegaconf as oc
import torch
import torch.distributed as dist
import tqdm
import transformers
import wandb

from text_sed import diffusion, layers, slurm, utils


logger = logging.getLogger(__name__)


def train(
    config: oc.DictConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    tokenizer: transformers.PreTrainedTokenizer,
    step_state: Optional[int] = 0,
    device: Optional[Union[torch.device, str]] = "cuda:0",
):
    # Initialize datasets
    logger.info("📦 Loading dataset...") 
    text_datasets = {
        "train": datasets.load_dataset(
            config.data.name,
            name=config.data.subset_name,
            use_auth_token=config.data.use_auth_token,
            split=config.data.train_name,
        ),
        "valid": datasets.load_dataset(
            config.data.name,
            name=config.data.subset_name,
            use_auth_token=config.data.use_auth_token,
            split=config.data.valid_name,
        ),
    }

    logger.info("📦 Loading dataloaders...") 
    # Initialize data loaders
    dataloaders = {
        "train": utils.text_dataloader(
            dataset=text_datasets["train"],
            tokenizer=tokenizer,
            per_gpu_batch_size=config.train.batch_size,
            max_seq_len=config.model.seq_len,
            num_workers=config.data.num_preprocess_workers,
            use_infinite_sampler=True,
        ),
        "valid": utils.text_dataloader(
            dataset=text_datasets["valid"],
            tokenizer=tokenizer,
            per_gpu_batch_size=config.valid.batch_size,
            max_seq_len=config.model.seq_len,
            num_workers=config.data.num_preprocess_workers,
            use_infinite_sampler=True,
        ),
    }
    train_iter = iter(dataloaders["train"])
    valid_iter = iter(dataloaders["valid"])

    model.train()
    for step in tqdm.trange(
        step_state,
        config.train.max_steps,
        initial=step_state,
        disable=not utils.is_main_process(),
    ):
        step += 1

        # TODO: The `BatchSampler` + `DataLoader` prepends an extra dimension to
        # the data. This is a hack to remove it.
        inputs = next(train_iter)["input_ids"].to(device)[0]
        with torch.amp.autocast(
            device_type="cuda", dtype=utils.get_dtype(config.train.dtype)
        ):
            loss, stats = model(inputs)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.optimizer.max_grad_norm
        )
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        # Log training stats
        if step % config.train.log_every == 0:
            wandb.log({f"train/{k}": v for k, v in stats.items()}, step=step)
            info = f"🎛 Step: {step}/{config.train.max_steps} "
            info += f"| Loss: {loss:.5f} | LR: {lr_scheduler.get_last_lr()[0]:.6f}"
            logger.info(info)

        # Evaluate and log the validation stats
        if step % config.train.eval_every == 0:
            model.eval()
            # TODO: The `BatchSampler` + `DataLoader` prepends an extra dimension to
            # the data. This is a hack to remove it.
            valid_inputs = next(valid_iter)["input_ids"].to(device)[0]
            with torch.no_grad():
                _, valid_stats = model(valid_inputs)
            wandb.log({f"valid/{k}": v for k, v in valid_stats.items()}, step=step)
            model.train()
            # Save latest checkpoint
            if utils.is_main_process():
                logger.info(f"💾 Saving latest checkpoint")
                checkpoint = {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                    "config": config,
                }
                path = os.path.join(config.output_dir, f"latest.pth")
                torch.save(checkpoint, path)

        # Generate samples
        is_sample_step = step % config.train.sample_every == 0 and step != 0
        if is_sample_step and utils.is_main_process():
            model.eval()
            # TODO: Add if-statement to unwrap DDP model
            samples = model.module.sample(
                shape=(config.train.num_samples, config.model.max_gen_len, config.model.word_embed_dim),
                num_steps=config.model.num_gen_steps,
                sampler=diffusion.get_sampler(config.model.sampler),
                # use_self_cond=config.model.use_self_cond,
                time_delta=config.model.time_delta,
                device=inputs.device,
            )
            samples = tokenizer.batch_decode(samples, skip_special_tokens=True)
            sample_log = "💬 Generating samples:\n"
            for sample in samples:
                sample_log += f"➜ {sample}\n"
            logger.info(sample_log)
            model.train()

        # Save checkpoints
        is_save_step = step % config.train.save_every == 0 and step != 0
        if is_save_step and utils.is_main_process():
            logger.info(f"💾 Saving checkpoint for step {step}")
            checkpoint = {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "config": config,
            }
            path = os.path.join(config.output_dir, f"step_{step}.pth")
            torch.save(checkpoint, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--name", type=str)
    parser.add_argument("--global_rank", type=int)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--master_addr", type=int, default=-1)
    parser.add_argument("--master_port", type=int, default=-1)
    args = parser.parse_args()

    slurm.init_distributed_mode(args)
    slurm.init_signal_handler()

    config = oc.OmegaConf.load(args.config)
    if args.name is not None:
        oc.OmegaConf.update(config, "name", args.name)

    os.makedirs(config.output_dir, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    logger.info(f"🖥 Device Count: {dist.get_world_size()}")

    # Set up logging
    utils.init_logger(logger, config.output_dir)
    if utils.is_main_process():
        logger.info(f"Config: {config}")
    if config.logging.use_wandb and utils.is_main_process():
        wandb.finish()  # Clear out any previous runs.
        wandb.init(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            name=config.name,
            config=utils.flatten_dict(oc.OmegaConf.to_container(config)),
            id=config.logging.wandb_id,
        )

    # Seed RNGs for ~reproducibility
    seeds = torch.randint(
        -(2**63),
        2**63 - 1,
        [dist.get_world_size() if dist.is_initialized() else 1],
        generator=torch.Generator().manual_seed(config.seed),
    )
    torch.manual_seed(seeds[utils.get_rank()])
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Initialize tokenizer - turn off HuggingFace parallelism warnings
    logger.info("⏳ Loading tokenizer...")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model.embed_model_name,
        use_fast=config.data.use_fast_tokenizer,
        use_auth_token=config.data.use_auth_token,
    )

    # Initialize model and optimizer
    embed_mat, embed_dim = layers.auto_extract_embed_mat(config.model.embed_model_name)
    inner_model = layers.TransformerEncoder(
        word_embed_dim=config.model.word_embed_dim,
        model_dim=config.model.model_dim,
        head_dim=config.model.head_dim,
        num_heads=config.model.num_heads,
        use_self_cond=config.model.use_self_cond,
        dropout=config.model.dropout,
    )
    diff = diffusion.TextSed(
        model=inner_model,
        word_embed_dim=config.model.word_embed_dim,
        embed_mat=embed_mat,
        noise_schedule=diffusion.get_noise_schedule(config.model.noise_schedule),
    )
    optimizer = torch.optim.AdamW(
        utils.get_grouped_params(diff, config.optimizer.weight_decay),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        betas=tuple(config.optimizer.betas),
        eps=config.optimizer.eps,
    )
    lr_scheduler = transformers.get_scheduler(
        name=config.optimizer.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.optimizer.warmup_steps,
        num_training_steps=config.train.max_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.train.use_amp)

    logger.info(f"👾 Parameter count: ~{format(utils.param_count(inner_model), ',')}")

    # Load checkpoints if resuming training
    if config.train.checkpoint_path is not None:
        logger.info(f"⏳ Loading checkpoint from {config.train.checkpoint_path}")
        checkpoint = torch.load(config.train.checkpoint_path)
        diff.load_state_dict(checkpoint["model"], strict=True)
        # Move model to GPU if available before loading optimizer state
        if torch.cuda.is_available():
            diff.cuda()
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        step_state = checkpoint["step"]
    else:
        step_state = 0
        if torch.cuda.is_available():
            diff.cuda()

    if dist.is_initialized():
        diff = torch.nn.parallel.DistributedDataParallel(
            diff,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        dist.barrier()

    logger.info("🏁 Starting training...") 
    train(config, diff, optimizer, lr_scheduler, scaler, tokenizer, step_state)
