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


# Embedding Layers


def auto_extract_embed_mat(
    model_name: str = "bert-base-uncased"
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


class EmbeddingReader(nn.Module):
    def __init__(
        self,
        init_embed: NamedTensor["vocab", "embed"],
        scale: Optional[float] = None,
        use_normalization: Optional[bool] = True,
    ):
        super().__init__()
        self.vocab_size, self.embed_dim = init_embed.shape
        # Fixed norm: √D
        self.scale = scale if scale else math.sqrt(self.embed_dim)

        _init_embed = init_embed.float().detach()

        # Discrete-to-continuous fixed read-in matrix
        in_weight = _init_embed.clone()
        if use_normalization:
            in_weight = self.scale * F.normalize(in_weight, dim=-1)
        self.register_buffer("in_weight", in_weight)

        # Continous-to-discrete learnable read-out matrix
        self.out_weight = nn.Parameter(
            rearrange(_init_embed.clone(), "v d -> d v"))  # Eᵀ ϵ Rᴰˣⱽ

    def embed(
        self, inputs: NamedTensor["...", "pos"]
    ) -> NamedTensor["...", "pos", "vocab"]:
        """Returns embedded inputs using the specified embedding lookup matrix."""
        one_hots = F.one_hot(inputs, num_classes=self.vocab_size).float()
        return torch.einsum("... i j, ... j k -> ... i k", one_hots, self.in_weight)

    def unembed(
        self, embeds: NamedTensor["...", "pos", "embed"]
    ) -> NamedTensor["...", "pos", "vocab"]:
        """Returns the logits from the given embeddings."""
        return torch.einsum("... s d, ... d v -> ... s v", embeds, self.out_weight)


def fixed_position_embedding(
    dim: int,
    num_pos: int,
    max_period: Optional[int] = 10_000,
) -> Tuple[NamedTensor["...", "pos", "dim"]]:
    inv_freq = 1. / (max_period ** (torch.arange(0, dim, 2) / dim))
    sinusoid = torch.einsum('i , j -> i j', torch.arange(num_pos), inv_freq)
    return torch.sin(sinusoid), torch.cos(sinusoid)


class FixedPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: Optional[int] = 10_000):
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
            max_period=self.max_period)
        return torch.concat([sincos[0], sincos[1]], dim=-1)


# TODO: Add Kat's Fourier embedding: https://github.com/crowsonkb/k-diffusion/blob/f4e99857772fc3a126ba886aadf795a332774878/k_diffusion/layers.py#L219


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: Optional[int] = 10_000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(
        self, time: NamedTensor["batch"]
    ) -> NamedTensor["batch dim"]:
        # Reference: https://github.com/magenta/music-spectrogram-diffusion
        min_timescale, max_timescale = 1., self.max_period
        num_timescale, ratio_scale = (
            self.dim // 2), (max_timescale / min_timescale)
        log_timescale = math.log(ratio_scale) / (num_timescale - 1.0)
        inv_timescale = min_timescale * torch.exp(
            torch.arange(0, num_timescale) * -log_timescale)
        timescale = time[:, None] * inv_timescale[None, :]

        return torch.concat([torch.sin(timescale), torch.cos(timescale)], -1)


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int, time_dim: Optional[int] = None):
        super().__init__()
        time_dim = time_dim if time_dim else 4 * embed_dim
        self.embed = nn.Sequential(
            SinusoidalTimeEmbedding(embed_dim),
            nn.Linear(embed_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, embed_dim),
            nn.GELU(),
        )

    def forward(self, time: NamedTensor["batch"]) -> NamedTensor["batch", "time_dim"]:
        return self.embed(time)


# Attention Helpers


def multihead_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: float,
    bias: Tensor = torch.Tensor([0.0])
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
        ff_mult: int = 4,  # Inner ff upscale factor (d_ff * 4)
    ):
        super().__init__()

        self.norm = nn.LayerNorm(model_dim)
        # Scaled dot-product attention factor: 1 / √dₖ
        self.scale = math.sqrt(head_dim) ** -1.0
        self.num_heads = num_heads

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
        # Pre-Norm: [..., seq, dim]
        units = self.norm(inputs)

        # Input Projects: [..., seq, *fused_dims]
        proj_units = self.proj_in(units)
        q, k, v, ff, ff_gate = split_fused_proj(proj_units, self.fused_dims)

        # Attention
        # [..., num_heads, seq, head_dim]
        q, k, v = map(partial(split_heads, num_heads=self.num_heads), (q, k, v))
        attn = multihead_attn(q, k, v, scale=self.scale)
        concat = merge_heads(attn)  # [..., seq, (num_heads * head_dim)]

        # Output projection: [..., seq, model]
        attn_out = self.attn_proj(concat)
        ff_out = self.ff_proj(ff * F.gelu(ff_gate, approximate='tanh'))
        return inputs + attn_out + ff_out


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        model_dim: int,
        time_dim: int,
        *,
        head_dim: Optional[int] = 64,
        num_heads: Optional[int] = 16,
        num_layers: Optional[int] = 12,
        ff_mult: Optional[int] = 4,
        use_self_cond: bool = True,
    ):
        super().__init__()

        time_channels = 4 * model_dim
        self.time_embed = TimeEmbedding(model_dim, time_dim)
        self.pos_embed = FixedPositionEmbedding(model_dim)

        # 2x b/c of concat noise & estimates
        in_embed_dim = 2 * embed_dim if use_self_cond else embed_dim
        self.in_proj = nn.Linear(in_embed_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, embed_dim)  # E'

        layers = []
        for _ in range(num_layers):
            layers.append(ParallelEncoderBlock(
                model_dim,
                head_dim,
                num_heads,
                ff_mult=ff_mult,
            ))
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        inputs: NamedTensor["batch", "pos", "embed"],
        time: NamedTensor["batch"],
    ) -> NamedTensor["batch", "pos", "embed"]:
        time_embed = rearrange(self.time_embed(time), "... d -> ... 1 d")
        hidden = self.in_proj(inputs) + self.pos_embed(inputs) + time_embed
        for layer in self.layers:
            hidden = layer(hidden)
        return self.out_proj(hidden)
