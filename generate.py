import argparse
import logging
import os
import time
from typing import *

import omegaconf as oc
import torch
import transformers

from text_sed import diffusion, layers, utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate(
    config: oc.DictConfig,
    model: torch.nn.Module,
    shape: Tuple[int, int, int],
    tokenizer: transformers.PreTrainedTokenizer,
    device: Optional[Union[torch.device, str]] = "cuda:0",
):
    model.eval()
    start_time = time.perf_counter()
    samples = model.generate(
        shape=shape,
        num_steps=config.model.num_gen_steps,
        sampler=diffusion.get_sampler(config.model.sampler),
        time_delta=config.model.time_delta,
        guide_scale=config.model.guide_scale,
        use_clamp=False,
        device=device,
    )
    samples = tokenizer.batch_decode(samples, skip_special_tokens=True)
    sample_log = "💬 Generating samples..."
    for sample in samples:
        sample_log += f"\n➜ {sample}"
    logger.info(sample_log)
    end_time = time.perf_counter()
    logger.info(f"🕒 Generation took {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--time_delta", type=float, default=None)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    config = oc.OmegaConf.load(args.config)
    if args.checkpoint_path is not None:
        oc.OmegaConf.update(config.train, "checkpoint_path", args.checkpoint_path)
    if args.seed is not None:
        oc.OmegaConf.update(config, "seed", args.seed)
    if args.time_delta is not None:
        oc.OmegaConf.update(config.model, "time_delta", args.time_delta)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    utils.set_seed(config.seed, use_device_specific_seeds=True)

    # Initialize tokenizer - turn off HuggingFace parallelism warnings
    logger.info("⏳ Loading tokenizer...")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model.embed_model_name,
        use_fast=config.data.use_fast_tokenizer,
        use_auth_token=config.data.use_auth_token,
    )

    embed_mat, embed_dim = layers.auto_extract_embed_mat(config.model.embed_model_name)
    inner_model = layers.MaskConditionalTransformer(
        embed_dim=config.model.bottleneck_dim
        if config.model.bottleneck_dim
        else embed_dim,
        model_dim=config.model.model_dim,
        max_seq_len=config.model.seq_len,
        head_dim=config.model.head_dim,
        num_heads=config.model.num_heads,
    )
    model = diffusion.TextSed(
        model=inner_model,
        embed_mat=embed_mat,
        noise_schedule=diffusion.get_noise_schedule(config.model.noise_schedule),
        bottleneck_dim=config.model.bottleneck_dim,
    )

    logger.info(f"⏳ Loading checkpoint from {config.train.checkpoint_path}")
    checkpoint = torch.load(config.train.checkpoint_path)
    # Load EMA model state for inference
    model.load_state_dict(checkpoint["model_ema"], strict=True)
    # Move model to GPU if available before loading optimizer state
    if torch.cuda.is_available():
        model.cuda()

    shape = (
        config.train.num_samples,
        config.model.seq_len,
        config.model.bottleneck_dim if config.model.bottleneck_dim else embed_dim,
    )
    logger.info("🏁 Starting generation...")
    generate(config, model, shape, tokenizer, device=args.device)
