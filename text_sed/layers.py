import math

import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from functools import partial

from typing import NewType, Literal, Optional, Tuple

DType = NewType("DType", torch.dtype)
Shape = NewType("Shape", Tuple[int, ...])
Tensor = NewType("Tensor", torch.Tensor)
NamedTensor = Literal  # __Naive__ named tensor
Generator = NewType("Generator", torch.Generator)


class ConditionScaleAndBias(nn.Module):
    """FiLM-style conditioning: https://distill.pub/2018/feature-wise-transformations/"""
    def __init__(self, dim: int):
        super().__init__()
        self.conditioner = nn.Sequential(
            nn.SiLU(), # nn.Sigmoid()
            nn.Linear(dim, 2 * dim),
        )

    def forward(
        self,
        inputs: NamedTensor["batch", "pos", "dim"],
        conds: NamedTensor["batch"],
    ) -> NamedTensor["batch", "pos", "dim"]:
        scale, bias = torch.chunk(self.conditioner(conds), chunks=2, dim=-1)
        return (1 + scale) * inputs + bias


# Embedding Layers


def auto_extract_embed_mat(
    model_name: str = "bert-base-uncased",
) -> Tuple[NamedTensor["vocab", "embed"], int]:
    """Extracts a pre-trained word embedding lookup matrix, E ϵ Rᴰˣⱽ, from the
    specified model.
    """
    # Extract the pre-trained word embedding lookup table.
    embed_model = transformers.AutoModel.from_pretrained(model_name)
    embeddings = embed_model.get_input_embeddings()
    embed_mat = embeddings.get_parameter("weight").detach()
    embed_dim = embeddings.embedding_dim
    del embed_model
    return embed_mat, embed_dim


def fixed_position_embedding(
    dim: int,
    num_pos: int,
    max_period: Optional[int] = 10_000,
) -> Tuple[NamedTensor["...", "pos", "dim"]]:
    inv_freq = 1.0 / (max_period ** (torch.arange(0, dim, 2) / dim))
    sinusoid = torch.einsum("i , j -> i j", torch.arange(num_pos), inv_freq)
    return torch.sin(sinusoid), torch.cos(sinusoid)


class FixedPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: Optional[int] = 10_000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def __call__(
        self,
        input: NamedTensor["...", "pos", "dim"],
        seq_dim: Optional[int] = -2,
    ) -> NamedTensor["...", "pos", "dim"]:
        """Returns the sinuosidal position embeddings for the given input."""
        sincos = fixed_position_embedding(
            self.dim,
            num_pos=input.shape[seq_dim],
            max_period=self.max_period,
        )
        return torch.concat([sincos[0], sincos[1]], dim=-1).to(input.device)


# TODO: Add Kat's Fourier embedding: https://github.com/crowsonkb/k-diffusion/blob/f4e99857772fc3a126ba886aadf795a332774878/k_diffusion/layers.py#L219


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: Optional[int] = 10_000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, time: NamedTensor["batch"]) -> NamedTensor["batch dim"]:
        # Reference: https://github.com/magenta/music-spectrogram-diffusion
        min_timescale, max_timescale = 1.0, self.max_period
        num_timescale, ratio_scale = (self.dim // 2), (max_timescale / min_timescale)
        log_timescale = math.log(ratio_scale) / (num_timescale - 1.0)
        inv_timescale = min_timescale * torch.exp(
            torch.arange(0, num_timescale, device=time.device) * -log_timescale
        )
        timescale = time[:, None] * inv_timescale[None, :]
        return torch.concat([torch.sin(timescale), torch.cos(timescale)], -1)


# Attention Helpers


def multihead_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: float,
    bias: Optional[Tensor] = 0.0,
) -> Tensor:
    """Scaled Dot-Product Attention ("Soft Look-Up Table")."""
    score = torch.einsum("... h s d, ... h S d -> ... h s S", q, k)
    score = score * scale + bias
    weight = F.softmax(score, dim=-1)
    attn = torch.einsum("... h s S, ... h S d -> ... h s d", weight, v)
    return attn


def split_fused_proj(proj: Tensor, dims: Shape) -> Tuple[Tensor, ...]:
    return torch.split(proj, dims, dim=-1)


def split_heads(x: Tensor, num_heads: int) -> Tensor:
    # s = seq_len, h = num_heads, d = head_dim
    return rearrange(x, "... s (h d) -> ... h s d", h=num_heads)


def merge_heads(x: Tensor) -> Tensor:
    """Concatenates the input multi-heads (reverse of `split_heads`)."""
    # s = seq_len, h = num_heads, d = head_dim, (h d) = h * d = embed_dim
    return rearrange(x, "... h s d -> ... s (h d)")  # "concat"


# Transformer


class ParallelEncoderBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        head_dim: int,
        num_heads: int,
        ff_mult: Optional[int] = 4,  # Inner ff upscale factor (d_ff * 4)
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = head_dim ** (-0.5)  # Scaled dot-product attention factor: 1 / √dₖ
        self.norm = nn.LayerNorm(model_dim)

        # Fused input projection: ((Wᵢq, Wᵢᵏ, Wᵢᵛ), (W1, W2))
        # 1 matmul for all input projections.
        attn_dims = 3 * (num_heads * head_dim,)  # (multi-q, multi-k, multi-v)
        ff_dims = 2 * (ff_mult * model_dim,)  # 2 * [4 * model_dim]
        self.fused_dims = (*attn_dims, *ff_dims)
        self.proj_in = nn.Linear(model_dim, sum(self.fused_dims), bias=False)

        # Output projections
        self.attn_proj = nn.Linear(num_heads * head_dim, model_dim, bias=False)
        self.ff_proj = nn.Linear(ff_mult * model_dim, model_dim, bias=False)

    def forward(
        self,
        inputs: NamedTensor["...", "pos", "dim"],
    ) -> Tensor:
        # Pre-Norm: [..., pos, dim]
        units = self.norm(inputs)

        # Input Projects: [..., pos, *fused_dims]
        proj_units = self.proj_in(units)
        q, k, v, ff, ff_gate = split_fused_proj(proj_units, self.fused_dims)

        # Attention
        # [..., num_heads, pos, head_dim]
        q, k, v = map(partial(split_heads, num_heads=self.num_heads), (q, k, v))
        attn = multihead_attn(q, k, v, scale=self.scale)
        concat = merge_heads(attn)  # [..., pos, (num_heads * head_dim)]

        # Output projection: [..., pos, model]
        attn_out = self.attn_proj(concat)
        ff_out = self.ff_proj(ff * F.gelu(ff_gate, approximate="tanh"))
        return inputs + attn_out + ff_out


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        word_embed_dim: int,
        model_dim: int,
        *,
        dropout: Optional[float] = 0.1,
        head_dim: Optional[int] = 64,
        num_heads: Optional[int] = 16,
        num_layers: Optional[int] = 12,
        ff_mult: Optional[int] = 4,
    ):
        super().__init__()

        time_dim = 4 * model_dim 
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(model_dim),
            nn.Linear(model_dim, time_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(time_dim, model_dim),
        )
        self.pos_embed = FixedPositionEmbedding(model_dim)
        self.embed_drop = nn.Dropout(dropout)

        # 2x b/c of self-conditioning concat of input and condition signal
        self.in_proj = nn.Linear(2 * word_embed_dim, model_dim)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, word_embed_dim), 
        )

        self.blocks = nn.ModuleList([
            ParallelEncoderBlock(model_dim, head_dim, num_heads, ff_mult=ff_mult)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        inputs: NamedTensor["batch", "pos", "dim"],
        self_cond: NamedTensor["batch", "pos", "dim"],
        time: NamedTensor["batch"],
    ) -> NamedTensor["batch", "pos", "embed"]:
        time_embed = rearrange(self.time_embed(time), "... d -> ... 1 d")
        inputs = self.in_proj(torch.concat([inputs, self_cond], dim=-1))
        hidden = inputs + self.pos_embed(inputs) + time_embed
        hidden = self.embed_drop(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return self.out_proj(hidden)
