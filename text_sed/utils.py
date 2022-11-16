import logging
import os
import sys
from functools import partial
from typing import *

import datasets
import torch
import torch.distributed as dist
import transformers
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler

Shape = NewType("Shape", Tuple[int, ...])
Tensor = NewType("Tensor", torch.Tensor)


def get_dtype(
    dtype: Union[str, torch.dtype],
    config: Optional[transformers.AutoConfig] = None
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
    no_decay: List[str] = ["bias", "LayerNorm.weight"]
):
    """Removes weight decay from parameters with names containing any of the
    strings in `no_decay`.
    Reference: HuggingFace `transformers` library.
    """
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


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
    tokenized_dataset.set_format("pt", columns=["input_ids"])
    data_collator = transformers.DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
    )
    if use_infinite_sampler:
        sampler = BatchSampler(
            RandomSampler(dataset, replacement=True, num_samples=int(1e100)),
            batch_size=per_gpu_batch_size,
            drop_last=False
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


def append_dims(x: Tensor, target_dims: int) -> Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions.
    Reference: @crowsonkb's `append_dims`.
    """
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
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


# Logging utils


def init_logger(
    logger: logging.Logger,
    output_dir: str,
    stdout_only=False,
):
    if dist.is_initialized():
        dist.barrier()
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [stdout_handler]
    if not stdout_only:
        file_handler = logging.FileHandler(
            filename=os.path.join(output_dir, "run.log"))
        handlers.append(file_handler)
    logger.setLevel(logging.INFO)
    for handler in handlers:
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(handler)
