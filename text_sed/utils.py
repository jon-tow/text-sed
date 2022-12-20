import logging
import os
import sys
from functools import partial
from typing import *

import datasets
import torch
import torch.nn as nn
import torch.distributed as dist
import transformers
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler

from .layers import LearnedAbsolutePositionalEmbedding

logger = logging.getLogger(__name__)


Shape = NewType("Shape", Tuple[int, ...])


# Exponential Moving Average


@torch.no_grad()
def ema_update(model: nn.Module, ema_model: nn.Module, decay: float = 0.999) -> None:
    """Updates the moving average of the model parameters.
    Reference: https://github.com/crowsonkb/k-diffusion/blob/5b3af030dd83e0297272d861c19477735d0317ec/k_diffusion/utils.py#L86
    """
    model_params = dict(model.named_parameters())
    ema_params = dict(ema_model.named_parameters())
    assert model_params.keys() == ema_params.keys()
    for name, param in model.named_parameters():
        ema_params[name].mul_(decay).add_(param, alpha=1 - decay)
    
    model_buffers = dict(model.named_buffers())
    ema_buffers = dict(ema_model.named_buffers())
    assert model_buffers.keys() == ema_buffers.keys()
    for name, buffer in model.named_buffers():
        ema_buffers[name].copy_(buffer)


def get_timestamp() -> str:
    import datetime

    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def set_seed(seed: int, use_device_specific_seeds: bool = False):
    import random

    import numpy as np

    if use_device_specific_seeds:
        seed = get_rank() + seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_dtype(
    dtype: Union[str, torch.dtype], config: Optional[transformers.AutoConfig] = None
) -> torch.dtype:
    """Converts `str` -> `torch.dtype` when possible."""
    if dtype is None and config is not None:
        _torch_dtype = config.torch_dtype
    elif isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_grouped_params(
    model: torch.nn.Module,
    weight_decay: float,
    whitelist_weight_modules: Tuple[torch.nn.Module] = (torch.nn.Linear,),
    blacklist_weight_modules: Tuple[torch.nn.Module] = (
        torch.nn.LayerNorm,
        torch.nn.Embedding,
        LearnedAbsolutePositionalEmbedding,
    ),
):
    """Removes weight decay from parameters with names containing any of the
    strings in `no_decay`.
    Reference: https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L215
    """
    decay = set()
    no_decay = set()
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters():
            full_param_name = "%s.%s" % (module_name, param_name) if module_name else param_name  # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if param_name.endswith("bias") or len(param.shape) == 1:
                # all biases will not be decayed
                no_decay.add(full_param_name)
            elif param_name.endswith("weight") and isinstance(module, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(full_param_name)
            elif param_name.endswith("weight") and isinstance(module, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(full_param_name)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay set!" % (
        str(param_dict.keys() - union_params),
    )
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]
    return optim_groups


# Data utils


def tokenize_fn(
    examples: List[dict],
    max_length: int,
    tokenizer: Any,
    text_attr: Optional[str] = "text",
    padding: Optional[str] = "max_length",
):
    return tokenizer(
        examples[text_attr],
        add_special_tokens=False,
        padding=padding,
        max_length=max_length,
        truncation=True,
    )


def text_dataloader(
    *,
    dataset: datasets.Dataset,
    tokenizer: Any,
    per_gpu_batch_size: int,
    max_seq_len: int,
    num_workers: Optional[int] = 1,
    use_infinite_sampler: bool = False,
):
    tokenized_dataset = dataset.map(
        partial(tokenize_fn, tokenizer=tokenizer, max_length=max_seq_len),
        batched=True,
        num_proc=num_workers,
    )
    tokenized_dataset.set_format("pt", columns=["input_ids", "attention_mask"])
    data_collator = transformers.DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
    )
    if use_infinite_sampler:
        sampler = BatchSampler(
            RandomSampler(dataset, replacement=True, num_samples=int(1e100)),
            batch_size=per_gpu_batch_size,
            drop_last=False,
        )
    else:
        sampler = BatchSampler(
            RandomSampler(dataset),
            batch_size=per_gpu_batch_size,
            drop_last=False,
        )
    dataloader = DataLoader(
        tokenized_dataset,
        sampler=sampler,
        drop_last=True,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


def flatten_dict(d: dict, parent_key: Optional[str] = "") -> dict:
    """
    Flattens a dict-of-dicts, replacing any nested key names with that name
    prepended with the parents' key names.
    """
    flat_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat_d.update(flatten_dict(v, parent_key=f"{k}_"))
        else:
            flat_d[f"{parent_key}{k}"] = v.item() if isinstance(v, torch.Tensor) else v
    return flat_d


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions.
    Reference: @crowsonkb's `append_dims`.
    """
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


# Distributed utils


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def print_rank_0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


# Logging utils


def init_logger(
    output_dir: str,
    stdout_only=False,
):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [stdout_handler]
    if not stdout_only:
        file_handler = logging.FileHandler(filename=os.path.join(output_dir, "run.log"))
        handlers.append(file_handler)
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process() else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    return logger
