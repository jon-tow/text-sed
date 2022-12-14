import math
import random

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


@torch.jit.script
def fused_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


class FusedGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return fused_gelu(x)


class ConditionScaleAndBias(nn.Module):
    """FiLM-style conditioning: https://distill.pub/2018/feature-wise-transformations/"""

    def __init__(self, dim: int):
        super().__init__()
        self.conditioner = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim),
        )

    def forward(
        self,
        inputs: NamedTensor["batch", "pos", "dim"],
        conds: NamedTensor["batch"],
    ) -> NamedTensor["batch", "pos", "dim"]:
        scale, bias = torch.chunk(self.conditioner(conds), chunks=2, dim=-1)
        return (1 + scale) * inputs + bias


# Word/Token Embedding Layers


def auto_extract_embed_mat(
    model_name: str = "bert-base-uncased",
) -> Tuple[NamedTensor["vocab", "embed"], int]:
    """Extracts a pre-trained word embedding lookup matrix, E ϵ Rᴰˣⱽ, from the
    specified model.
    """
    # Extract the pre-trained word embedding lookup table
    embed_model = transformers.AutoModel.from_pretrained(model_name)
    embeddings = embed_model.get_input_embeddings()  # Word embeddings layer
    embed_mat = embeddings.get_parameter("weight").detach()
    embed_dim = embeddings.embedding_dim
    del embed_model
    return embed_mat, embed_dim


class PretrainedEmbedding(nn.Module):
    def __init__(
        self,
        embed_mat: NamedTensor["vocab", "embed"],
        use_normalization: bool = True,
        freeze: bool = False,
    ):
        super().__init__()
        self.use_normalization = use_normalization
        _, embed_dim = embed_mat.shape
        self.scale = math.sqrt(embed_dim)  # √D
        self.embed = nn.Embedding.from_pretrained(
            embed_mat.detach().clone(), freeze=freeze
        )

    def forward(
        self, x: NamedTensor["batch", "vocab"]
    ) -> NamedTensor["batch", "embed"]:
        embeds = self.embed(x)
        if self.use_normalization:
            embeds = F.normalize(embeds, p=2, dim=-1) * self.scale
        return embeds


class PretrainedUnEmbedding(nn.Module):
    def __init__(
        self, embed_mat: NamedTensor["vocab", "embed"], use_renormalization: bool = True
    ):
        super().__init__()
        self.renormalize = use_renormalization
        vocab_size, embed_dim = embed_mat.shape
        # LM head style scoring
        self.unembed = nn.Linear(embed_dim, vocab_size, bias=False)
        with torch.no_grad():
            self.unembed.weight.copy_(embed_mat.detach().clone())

    def forward(
        self, x: NamedTensor["batch", "pos", "dim"]
    ) -> NamedTensor["batch", "pos", "vocab"]:
        # CDCD Framework: Apply L2-normalisation to the embedding estimate
        # before calculating the score estimate (__renormalisation__)
        if self.renormalize:
            x = F.normalize(x, p=2, dim=-1)
        return self.unembed(x)


# Position Embedding Layers


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_positional_embedding(
    x: torch.Tensor,
    freqs: torch.Tensor,
    seq_dim: int = -2,
):
    seq_len = x.shape[seq_dim]
    freqs = freqs[-seq_len:, :]
    return (x * torch.cos(freqs)) + (rotate_half(x) * torch.sin(freqs))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (max_period ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: int, seq_dim: int = -2):
        seq_len = x.shape[seq_dim]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1).to(x.device)


def fixed_positional_embedding(
    dim: int,
    num_pos: int,
    max_period: int = 10_000,
) -> Tuple[NamedTensor["...", "pos", "dim"]]:
    inv_freq = 1.0 / (max_period ** (torch.arange(0, dim, 2) / dim))
    sinusoid = torch.einsum("i , j -> i j", torch.arange(num_pos), inv_freq)
    return torch.sin(sinusoid), torch.cos(sinusoid)


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: Optional[int] = 10_000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(
        self,
        input: NamedTensor["...", "pos", "dim"],
        seq_dim: Optional[int] = -2,
    ) -> NamedTensor["...", "pos", "dim"]:
        """Returns the sinuosidal position embeddings for the given input."""
        sincos = fixed_positional_embedding(
            self.dim,
            num_pos=input.shape[seq_dim],
            max_period=self.max_period,
        )
        return torch.concat([sincos[0], sincos[1]], dim=-1).to(input.device)


class LearnedAbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, init_scale=1.0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.init_scale = init_scale
        self.weight = nn.Parameter(torch.empty(size=(max_seq_len, dim)))
        self.reset_params()

    def reset_params(self):
        nn.init.normal_(self.weight, std=0.01 * self.init_scale)

    def forward(
        self,
        x: NamedTensor["...", "pos", "dim"],
        start_pos: int = 0,
        seq_dim: int = -2,
    ) -> NamedTensor["pos", "dim"]:
        end_pos = start_pos + x.shape[seq_dim]
        assert end_pos <= self.max_seq_len, (
            f"Positional embedding doesnt exist for position {end_pos}. This is likely due to "
            f"feeding a sequence that's longer than the context length n_ctx={self.dim}."
        )
        return self.weight[start_pos:end_pos]


# Time Embedding Layers


class RandomFourierEmbedding(nn.Module):
    """Kat's Fourier embedding.
    Reference: https://github.com/crowsonkb/k-diffusion/blob/f4e99857772fc3a126ba886aadf795a332774878/k_diffusion/layers.py#L219
    """

    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer(
            "weight", torch.randn([out_features // 2, in_features]) * std
        )

    def forward(self, inputs: NamedTensor["batch"]) -> NamedTensor["batch", "dim"]:
        f = 2 * math.pi * inputs @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


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


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_mult: int = 1,
        use_fourier: bool = True,
        max_period: Optional[int] = 10_000,
    ):
        super().__init__()
        time_dim = ff_mult * dim
        self.use_fourier = use_fourier
        if use_fourier:
            embed = RandomFourierEmbedding(1, dim)
        else:
            embed = SinusoidalTimeEmbedding(dim, max_period=max_period)
        self.time_embed = nn.Sequential(
            embed,
            nn.Linear(dim, time_dim),
            FusedGELU(),
            nn.Linear(time_dim, dim),
        )

    def forward(self, time: NamedTensor["batch"]) -> NamedTensor["batch dim"]:
        if self.use_fourier:
            # Append extra dimension to time for proper matmul broadcasting
            time = time[:, None]
        return self.time_embed(time)


# Attention Helpers


def multihead_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: float,
    bias: Optional[Tensor] = 0.0,
) -> Tensor:
    """Scaled Dot-Product Attention ("Soft Look-Up Table")."""
    score = torch.einsum("... h s d, ... h S d -> ... h s S", q, k)
    score = score * softmax_scale + bias
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
        use_conditioner: bool = True,
        use_rotary: bool = True,
        rotary_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.softmax_scale = head_dim ** (-0.5)  # Scaled dot-product attention factor: 1 / √dₖ
        self.norm = nn.LayerNorm(model_dim)
        self.gelu = FusedGELU()

        rotary_embed_dim = max(head_dim // 2 if rotary_dim is None else rotary_dim, 32)
        self.rotary_pos_embed = (
            RotaryPositionalEmbedding(rotary_embed_dim) if use_rotary else None
        )
        self.conditioner = ConditionScaleAndBias(model_dim) if use_conditioner else None

        # Fused input projection: ((Wᵢq, Wᵢᵏ, Wᵢᵛ), (W1, W2))
        # 1 matmul for all input projections
        attn_dims = 3 * (num_heads * head_dim,)  # (multi-q, multi-k, multi-v)
        ff_dims = 2 * (ff_mult * model_dim,)  # 2 * [4 * model_dim]
        self.fused_dims = (*attn_dims, *ff_dims)
        self.proj_in = nn.Linear(model_dim, sum(self.fused_dims), bias=False)

        # Output projections
        self.attn_proj = nn.Linear(num_heads * head_dim, model_dim, bias=False)
        self.ff_proj = nn.Linear(ff_mult * model_dim, model_dim, bias=True)

    def forward(
        self,
        inputs: NamedTensor["...", "pos", "dim"],
        time_embeds: Optional[NamedTensor["batch dim"]] = None,
    ) -> Tensor:
        # Pre-Norm: [..., pos, dim]
        units = self.norm(inputs)
        if self.conditioner:
            units = self.conditioner(units, time_embeds)

        # Input Projects: [..., pos, *fused_dims]
        proj_units = self.proj_in(units)
        q, k, v, ff, ff_gate = split_fused_proj(proj_units, self.fused_dims)

        # Self-Attention
        # [..., num_heads, pos, head_dim]
        q, k, v = map(partial(split_heads, num_heads=self.num_heads), (q, k, v))
        if self.rotary_pos_embed:
            pos_embeds = self.rotary_pos_embed(k)
            rot_dim = pos_embeds.shape[-1]

            q_left, q_right = q[..., :rot_dim], q[..., rot_dim:]
            k_left, k_right = k[..., :rot_dim], k[..., rot_dim:]

            q_left = apply_rotary_positional_embedding(q_left, pos_embeds)
            k_left = apply_rotary_positional_embedding(k_left, pos_embeds)

            k = torch.cat([k_left, k_right], dim=-1)
            q = torch.cat([q_left, q_right], dim=-1)
        attn = multihead_attn(q, k, v, softmax_scale=self.softmax_scale)
        concat = merge_heads(attn)  # [..., pos, (num_heads * head_dim)]

        # Output projection: [..., pos, model_dim]
        attn_out = self.attn_proj(concat)
        ff_out = self.ff_proj(ff * self.gelu(ff_gate))
        return inputs + attn_out + ff_out


class MaskConditionalTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        model_dim: int,
        max_seq_len: int,
        *,
        head_dim: Optional[int] = 64,
        num_heads: Optional[int] = 16,
        num_layers: Optional[int] = 12,
        ff_mult: Optional[int] = 4,
        use_abs_pos_embed: bool = False,
        use_rotary: bool = True,
        rotary_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.time_embed = TimeEmbedding(model_dim)
        self.pos_embed = (
            LearnedAbsolutePositionalEmbedding(model_dim, max_seq_len)
            if use_abs_pos_embed
            else None
        )

        # 2x b/c of self-conditioning concat of input and condition signal
        self.in_proj = nn.Linear(2 * embed_dim, model_dim)
        self.blocks = nn.ModuleList([
            ParallelEncoderBlock(
                model_dim,
                head_dim,
                num_heads,
                ff_mult=ff_mult,
                use_rotary=use_rotary,
                rotary_dim=rotary_dim,
            )
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, embed_dim),
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        # TODO: Update args for span(conditional)-masking
        noisy_embeds: NamedTensor["batch", "pos", "dim"],
        prev_embeds: NamedTensor["batch", "pos", "dim"],
        time: NamedTensor["batch"],
    ) -> NamedTensor["batch", "pos", "embed"]:
        """
        Conditioning tokens c are defined by a binary conditioning mask m set to
        1 on conditioning positions and 0 on positions to be infilled.

        Args:
            # embeds (c): Token embeddings for clean positions.
            noisy_embeds (x): Corrupted `embeds` embeddings.
            prev_embeds (p): Previous predicted embeddings for self-conditioning.
        """
        time_embeds = rearrange(self.time_embed(time), "... d -> ... 1 d")
        cond_embeds = self.in_proj(torch.concat([noisy_embeds, prev_embeds], dim=-1))
        if self.pos_embed:
            pos_embeds = self.pos_embed(cond_embeds)
            cond_embeds += pos_embeds
        hidden_states = cond_embeds
        for block in self.blocks:
            hidden_states = block(hidden_states, time_embeds)
        hidden_states = self.out_proj(hidden_states)
        return self.final_norm(hidden_states)


# Masking helpers


def get_prefix_mask(seq_len: int) -> Tensor:
    """Returns a prefix mask of random length in [0, (seq_len / 2) - 1)."""
    indices = torch.arange(0, seq_len)
    prefix_len = random.randint(0, (seq_len // 2) - 1)
    mask = (indices > prefix_len).float()
    return mask


def get_span_mask(seq_len: int, max_num_spans: int) -> Tensor:
    """Returns a binary mask of span partitions for a sequence
    where 1s indicates a span region who's tokens will be kept to
    condition on and 0s indicate a span region who's tokens will be
    masked out for corruption -> infilling.

    Example:
    >>> get_span_mask(10, 3)
    >>> [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],

    Handwritten Example:

    >>> empty_mask  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    >>> span_starts = [2, 6, 10]
    >>> positions   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    >>> span_mask   = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0,  1,  1]  # Final Mask

    Args:
        seq_len (L): The length of the sequences.
        max_num_spans (M): is the maximum number of spans.

    Reference: Strudel et al. "Self-Conditioned Embedding For Text Generation"
        https://arxiv.org/abs/2211.04236
    """
    # TODO: Make this batched so each sequence has different random spans
    num_spans = random.randint(1, max_num_spans)  # (n)
    if num_spans == 1:
        # If there is only one span we just do unconditional generation and
        # zero out all conditional positions for infilling
        return torch.zeros(seq_len, dtype=torch.bool)
    # Sample uniformly without replacement n - 1 integers (i1, ..., i(n-1)) in [0, seq_len)
    # and sort them in increasing order to satisfy the condition 0 < i1 < i2 < ... < i(n-1) < seq_len
    span_starts = sorted(random.sample(range(1, seq_len), num_spans - 1))  # (n - 1)
    # m is defined using the even spans for conditioning (1) and odd spans for infilling (0)
    mask = torch.zeros(seq_len, dtype=torch.bool)
    i_span_starts = list(enumerate(span_starts))
    for i, start in i_span_starts[:-1]:
        if i % 2 == 0:
            mask[start : span_starts[i + 1]] = True
    last_i, last_span_start = i_span_starts[-1]
    if last_i % 2 == 0:
        mask[last_span_start:] = True
    # Flip mask with probability 0.5 so the positions from 0 to i1 aren't always False(0)
    mask = mask if random.random() < 0.5 else ~mask
    return mask
