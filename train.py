"""
Basic Usage:
    torchrun --nproc_per_node=N train.py
"""
import argparse
import os
from typing import *

import datasets
import omegaconf as oc
import torch
import torch.distributed as dist
import tqdm
import transformers
import wandb

from text_sed import diffusion, layers, slurm, utils


def train(
    config: oc.DictConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    tokenizer: transformers.PreTrainedTokenizer,
    step_state: Optional[int] = 0,
    run: Optional["wandb.Run"] = None,
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
    # valid_iter = iter(dataloaders["valid"])

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
        if step % config.train.log_every == 0 and utils.is_main_process():
            run.log({f"train/{k}": v for k, v in stats.items()}, step=step)
            info = f"🎛 Step: {step}/{config.train.max_steps} "
            info += f"| Loss: {loss:.5f} | LR: {lr_scheduler.get_last_lr()[0]:.6f}"
            logger.info(info)

        # Evaluate and log the validation stats
        if step % config.train.eval_every == 0 and utils.is_main_process():
            logger.info(
                "📊 Evaluating... WARNING: Evaluation is slow! Run evaluations on checkpoints instead."
            )
            # model.eval()
            # TODO: The `BatchSampler` + `DataLoader` prepends an extra dimension to
            # the data. This is a hack to remove it.
            # valid_inputs = next(valid_iter)["input_ids"].to(device)[0]
            # with torch.no_grad():
            #     _, valid_stats = model(valid_inputs)
            # run.log({f"valid/{k}": v for k, v in valid_stats.items()}, step=step)
            # model.train()
            # Save latest checkpoint
            # if utils.is_main_process():
            #     logger.info(f"💾 Saving latest checkpoint")
            #     checkpoint = {
            #         "model": model.module.state_dict(),
            #         "optimizer": optimizer.state_dict(),
            #         "lr_scheduler": lr_scheduler.state_dict(),
            #         "scaler": scaler.state_dict(),
            #         "step": step,
            #         "config": config,
            #     }
            #     path = os.path.join(config.output_dir, f"latest.pth")
            #     torch.save(checkpoint, path)

        # Generate samples
        is_sample_step = step % config.train.sample_every == 0 and step != 0
        if is_sample_step and utils.is_main_process():
            logger.info("💬 Generating samples...")
            model.eval()
            # TODO: Add if-statement to unwrap DDP model
            samples = model.module.generate(
                shape=(
                    config.train.num_samples,
                    config.model.seq_len,
                    config.model.bottleneck_dim if config.model.bottleneck_dim else embed_dim,
                ),
                num_steps=config.model.num_gen_steps,
                sampler=diffusion.get_sampler(config.model.sampler),
                time_delta=config.model.time_delta,
                device=inputs.device,
            )
            samples = tokenizer.batch_decode(samples, skip_special_tokens=True)
            sample_log = "🎙 Samples: \n"
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
    else:
        # Add timestamp to checkpoint dir name
        # TODO: There's probably a better way to do this...
        oc.OmegaConf.update(config, "output_dir", f"{config.output_dir}-{utils.get_timestamp()}")

    os.makedirs(config.output_dir, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # Set up logging
    logger = utils.init_logger(config.output_dir)
    logger.info(f"🖥 Device Count: {dist.get_world_size()}")
    logger.info(f"🎚 Config: {config}")
    run = None
    if utils.is_main_process():
        wandb.finish()  # Clear out any previous runs.
        wandb_id = wandb.util.generate_id() if config.logging.wandb_id is None else \
            config.logging.wandb_id
        run = wandb.init(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            name=f"{config.name}-{wandb_id}",
            config=utils.flatten_dict(oc.OmegaConf.to_container(config)),
            id=wandb_id,
            # group=config.logging.wandb_group,
            # job_type="train",
        )

    utils.set_seed(config.seed, use_device_specific_seeds=True)

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
    inner_model = layers.MaskConditionalTransformer(
        embed_dim=config.model.bottleneck_dim if config.model.bottleneck_dim else embed_dim,
        model_dim=config.model.model_dim,
        max_seq_len=config.model.seq_len,
        head_dim=config.model.head_dim,
        num_heads=config.model.num_heads,
    )
    diff = diffusion.TextSed(
        model=inner_model,
        embed_mat=embed_mat,
        noise_schedule=diffusion.get_noise_schedule(config.model.noise_schedule),
        bottleneck_dim=config.model.bottleneck_dim,
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

    logger.info(f"🏘 Inner Model: {inner_model}")
    logger.info(f"👾 Parameter count: ~{format(utils.param_count(diff), ',')}")

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
    try:
        train(config, diff, optimizer, lr_scheduler, scaler, tokenizer, step_state, run=run)
    except Exception as e:
        logger.info(f"🛑 Training interrupted.\n{e}")
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except KeyboardInterrupt:
            os.system("kill -9 $(ps aux | grep train.py | grep -v grep | awk '{print $2}')") 
