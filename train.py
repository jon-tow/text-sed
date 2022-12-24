"""
Basic Usage:
torchrun --nproc_per_node=<N> train.py
"""
import argparse
import copy
import os
import time
from typing import *

import omegaconf as oc
import torch
import torch.distributed as dist
import tqdm
import transformers

import datasets
import wandb
from text_sed import diffusion, layers, slurm, utils


def train(
    config: oc.DictConfig,
    model: torch.nn.Module,
    model_ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    tokenizer: transformers.PreTrainedTokenizer,
    step_state: Optional[int] = 0,
    run: Optional["wandb.Run"] = None,
    device: Optional[Union[torch.device, str]] = "cuda:0",
):
    # Initialize datasets
    logger.info("📦 Loading dataset...")
    text_datasets = {"train": datasets.load_dataset(**config.data.train_kwargs)}
    if config.data.valid_kwargs:
        text_datasets["valid"] = datasets.load_dataset(**config.data.valid_kwargs)

    # Initialize data loaders
    logger.info("📦 Loading dataloaders...")
    dataloaders = {
        "train": utils.text_dataloader(
            dataset=text_datasets["train"],
            tokenizer=tokenizer,
            per_gpu_batch_size=config.train.batch_size,
            max_seq_len=config.model.seq_len,
            num_workers=config.data.num_preprocess_workers,
            use_infinite_sampler=True,
        ),
    }
    if config.data.valid_kwargs:
        dataloaders["valid"] = utils.text_dataloader(
            dataset=text_datasets["valid"],
            tokenizer=tokenizer,
            per_gpu_batch_size=config.valid.batch_size,
            max_seq_len=config.model.seq_len,
            num_workers=config.data.num_preprocess_workers,
            use_infinite_sampler=True,
        )

    # Initialize data iterators
    train_iter = iter(dataloaders["train"])
    if config.data.valid_kwargs:
        valid_iter = iter(dataloaders["valid"])

    logger.info("⏳ Begin model training...")
    model.train()
    for step in tqdm.trange(
        step_state,
        config.train.max_steps,
        initial=step_state,
        disable=not utils.is_main_process(),
    ):
        step += 1

        batch = next(train_iter)
        # TODO: The `BatchSampler` + `DataLoader` prepends an extra dimension to the data. 
        input_ids = batch["input_ids"][0].to(device)
        attention_mask = batch["attention_mask"][0].to(device)
        with torch.amp.autocast(
            device_type="cuda", dtype=utils.get_dtype(config.train.dtype)
        ):
            loss, stats = model(input_ids, attention_mask=attention_mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.optimizer.max_grad_norm
        )
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        if step % config.model.ema_every == 0:
            utils.ema_update(model, model_ema, config.model.ema_decay)
        optimizer.zero_grad(set_to_none=True)

        # Log training stats
        if step % config.train.log_every == 0 and utils.is_main_process():
            # Log learning across all param groups
            stats[f"learning_rate"] = lr_scheduler.get_last_lr()[0]
            run.log({f"train/{k}": v for k, v in stats.items()}, step=step)
            info = f"🎛 Step: {step}/{config.train.max_steps} "
            info += f"𑗔 Loss: {loss:.5f} "
            info += f"𑗔 MSE Loss: {stats['loss_mse']:.5f} "
            info += f"𑗔 Recon Loss: {stats['loss_recon']:.5f} "
            info += f"𑗔 LR: {stats['learning_rate']:.6f}"
            logger.info(info)

        # Evaluate and log the validation stats
        is_eval_step = step % config.train.eval_every == 0 and step > 0 and config.data.valid_kwargs
        if is_eval_step and utils.is_main_process():
            logger.info(
                "📊 Evaluating... "
                "WARNING: Evaluation is slow! Run evaluations on checkpoints instead."
            )
            model.eval()
            # TODO: The `BatchSampler` + `DataLoader` prepends an extra dimension to the data.
            valid_inputs = next(valid_iter)["input_ids"].to(device)[0]
            with torch.no_grad():
                _, valid_stats = model(valid_inputs)
            run.log({f"valid/{k}": v for k, v in valid_stats.items()}, step=step)
            model.train()

        # Generate samples
        is_sample_step = step % config.train.sample_every == 0
        if is_sample_step and utils.is_main_process():
            logger.info("💬 Generating samples...")
            model_ema.eval()
            shape = (
                config.train.num_samples,
                config.model.seq_len,
                config.model.bottleneck_dim
                if config.model.bottleneck_dim
                else embed_dim,
            )
            start_time = time.perf_counter()
            samples = model_ema.module.generate(
                shape=shape,
                num_steps=config.model.num_gen_steps,
                sampler=diffusion.get_sampler(config.model.sampler),
                time_delta=config.model.time_delta,
                guide_scale=config.model.guide_scale,
                use_clamp=False,
                device=input_ids.device,
            )
            end_time = time.perf_counter()
            sample_log = "💬 Generating tokens..."
            for sample in samples:
                sample_log += f"\n➜ {sample}"
            samples = tokenizer.batch_decode(samples, skip_secial_tokens=True)
            sample_log += "\n"
            sample_log += "💬 Decoding tokens..."
            for sample in samples:
                sample_log += f"\n➜ {sample}"
            logger.info(sample_log)
            logger.info(
                f"🕒 Generation took {end_time - start_time:.2f} seconds."
            )
            model_ema.train()

        # Save checkpoints
        is_save_step = step % config.train.save_every == 0 and step != 0
        if is_save_step and utils.is_main_process():
            logger.info(f"💾 Saving checkpoint for step {step}")
            checkpoint = {
                "model": model.module.state_dict(),
                "model_ema": model_ema.module.state_dict(),
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
        oc.OmegaConf.update(
            config, "output_dir", f"{config.output_dir}-{utils.get_timestamp()}"
        )

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
        wandb_id = (
            wandb.util.generate_id()
            if config.logging.wandb_id is None
            else config.logging.wandb_id
        )
        run = wandb.init(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            name=f"{config.name}-{wandb_id}",
            config=utils.flatten_dict(oc.OmegaConf.to_container(config)),
            id=wandb_id,
        )

    utils.set_seed(config.seed, use_device_specific_seeds=True)

    # Initialize tokenizer
    logger.info("⏳ Loading tokenizer...")
    # Turn turn off HuggingFace parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model.embed_model_name,
        use_fast=config.data.use_fast_tokenizer,
        use_auth_token=config.data.use_auth_token,
    )

    # Initialize model and optimizer
    embed_mat, embed_dim = layers.auto_extract_embed_mat(
        config.model.embed_model_name)
    inner_model = layers.MaskConditionalTransformer(
        embed_dim=config.model.bottleneck_dim
        if config.model.bottleneck_dim
        else embed_dim,
        model_dim=config.model.model_dim,
        max_seq_len=config.model.seq_len,
        head_dim=config.model.head_dim,
        num_heads=config.model.num_heads,
        use_abs_pos=config.model.use_abs_pos,
        use_rotary=config.model.use_rotary,
    )
    model = diffusion.TextSed(
        model=inner_model,
        embed_mat=embed_mat,
        noise_schedule=diffusion.get_noise_schedule(
            config.model.noise_schedule),
        bottleneck_dim=config.model.bottleneck_dim,
        max_num_spans=config.model.max_num_spans,
    )
    optimizer = torch.optim.AdamW(
        utils.get_grouped_params(
            model, config.optimizer.weight_decay,
            exlcuded_modules=(
                torch.nn.LayerNorm,
                torch.nn.Embedding,
            )
        ),
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
    logger.info(f"👾 Parameter Count: ~{format(utils.param_count(model), ',')}")

    if torch.cuda.is_available():
        model.cuda()
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        dist.barrier()
    # Init the model EMA after DDP to avoid state dict key mismatches during updates
    model_ema = copy.deepcopy(model)

    # Load checkpoints if resuming training
    if config.train.checkpoint_path is not None:
        logger.info(f"⏳ Loading checkpoint from {config.train.checkpoint_path}")
        checkpoint = torch.load(config.train.checkpoint_path)
        model.module.load_state_dict(checkpoint["model"], strict=True)
        model_ema.module.load_state_dict(checkpoint["model_ema"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        step_state = checkpoint["step"]
    else:
        step_state = 0

    logger.info("🏁 Starting training...")
    train(
        config,
        model,
        model_ema,
        optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        tokenizer=tokenizer,
        step_state=step_state,
        run=run,
    )
