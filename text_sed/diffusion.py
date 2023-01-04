import functools
import random
from typing import Callable, Literal, NewType, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from text_sed.layers import (
    PretrainedEmbedding,
    PretrainedUnembedding,
    get_prefix_mask,
    get_span_mask,
)

from . import utils

Device = NewType("Device", torch.device)
DType = NewType("DType", torch.dtype)
Shape = NewType("Shape", Tuple[int, ...])
Tensor = NewType("Tensor", torch.Tensor)
NamedTensor = Literal  # **Naive** named tensor
Generator = NewType("Generator", torch.Generator)


# Loss functions


def cross_entropy_loss(
    logits: NamedTensor["batch", "pos", "vocab"],
    labels: NamedTensor["batch", "pos"],
    mask: Optional[NamedTensor["batch", "pos", "1"]] = None,
) -> float:
    """
    Args:
        logits: The unnormalized label scores.
        labels: The ground truth labels. NOTE: These will be one-hot encoded.
    """
    mask = utils.default(mask, torch.ones_like(labels)[:, :, None])
    num_tokens = mask.sum()

    logits -= torch.max(logits, dim=-1, keepdim=True)[0]
    one_hot_targets = F.one_hot(labels, num_classes=logits.shape[-1])
    predicted_logits = torch.sum(one_hot_targets * logits, dim=-1)
    loss = -predicted_logits + torch.logsumexp(logits, dim=-1)
    loss = torch.sum(loss[:, :, None] * mask) / num_tokens

    # Compute the fraction of correct predictions per batch
    correct = (torch.argmax(logits, dim=-1) == labels).float()[:, :, None]
    correct = torch.sum(correct * mask) / num_tokens
    return dict(loss=loss, correct=correct)


# Noise Schedules
# Reference: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# @ unixpickle 🤘🥒


def get_noise_schedule(name: str, **kwargs) -> Callable:
    """Returns ᾱ schedules"""
    if name == "cosine":
        return cosine_alpha_bar_schedule(**kwargs)
    elif name == "linear":
        return linear_schedule(**kwargs)
    raise ValueError(f"Schedule `{name}` is not available.")


def linear_schedule(start: float, end: float) -> Tensor:
    """Linear noise-variance (β) schedule."""

    def scheduler(num_steps: int):
        return torch.linspace(start, end, num_steps)

    return scheduler


def cosine_alpha_bar(
    time: float,
    offset: float = 0.0002,
) -> Tensor:
    """Cosine noise-variance ᾱ scheduler (ᾱ[t] = Πᵗα[i] where α[i] = (1 - β[i]))
    for continuous time parameterization.

    Reference: Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models".
        2021. https://arxiv.org/pdf/2102.09672.pdf

    Args:
        offset: Small offset to prevent βₜ from beeing too small near
            t = 0.
    """
    return torch.cos(((time + offset) / (1 + offset)) * torch.pi / 2) ** 2


def cosine_alpha_bar_schedule(
    offset: float = 0.0002,
) -> Tensor:
    """Cosine noise-variance (β) scheduler

    Reference: Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models".
        2021. https://arxiv.org/pdf/2102.09672.pdf

    Args:
        offset: Small offset to prevent βₜ from beeing too small near
            t = 0.
    """

    def scheduler(num_steps: float):
        return cosine_alpha_bar(time=num_steps, offset=offset)

    return scheduler


# Samplers


def get_sampler(name: str, **kwargs) -> Callable:
    """Returns sampler step function"""
    if name == "ddim":
        return functools.partial(ddim_step, **kwargs)
    elif name == "ddpm":
        return functools.partial(ddpm_step, **kwargs)
    else:
        raise ValueError(f"Sampler `{name}` is not available.")


def ddim_step(
    noisy_inputs: Tensor,  # xₜ
    pred_inputs: Tensor,   # x̃₀
    time_now: Tensor,      # t
    time_next: Tensor,     # t - 1
    schedule: Callable,
) -> Tensor:               # xₜ₋₁
    """Denoising diffusion implicit model step with η = 0. Estimates x₀ at
    time_next with the DDIM updating rule.

    References:
    - Song et al. "Denoising Diffusion Implicit Models". 2020.
        https://arxiv.org/pdf/2010.02502.pdf
    - Lilian Weng.
        https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#speed-up-diffusion-model-sampling
    """
    xₜ, x̃ₒ = noisy_inputs, pred_inputs
    ᾱₜ, ᾱₙ = schedule(time_now), schedule(time_next)  # ᾱₙ = ᾱₜ₋₁
    ϵ = (xₜ - torch.sqrt(ᾱₜ) * x̃ₒ) * torch.rsqrt(1 - ᾱₜ)
    # Next estimate (xₙ := xₜ₋₁)
    xₙ = torch.sqrt(ᾱₙ) * x̃ₒ + torch.sqrt(1 - ᾱₙ) * ϵ
    return xₙ


def ddpm_step(
    noisy_inputs: Tensor,  # xₜ
    pred_inputs: Tensor,   # x̃₀
    time_now: Tensor,      # t
    time_next: Tensor,     # t - 1
    schedule: Callable,
) -> Tensor:               # xₜ₋₁
    """Denoising diffusion implicit model step with η = 1. Estimates x₀ at
    time_next with the DDPM updating rule.

    Reference: Ho et al. "Denoising Diffusion Probabilistic Models". 2020.
        https://arxiv.org/abs/2006.11239
    """
    xₜ, x̃ₒ = noisy_inputs, pred_inputs
    γₜ = schedule(time_now)
    ᾱₜ = γₜ / schedule(time_next)
    σₜ = torch.sqrt(1 - ᾱₜ)
    z = torch.randn_like(σₜ)
    ϵ = (xₜ - torch.sqrt(γₜ) * x̃ₒ) * torch.rsqrt(1 - γₜ)
    # Next estimate (xₙ := xₜ₋₁)
    xₙ = torch.rsqrt(ᾱₜ) * (xₜ - ((1 - ᾱₜ) * torch.rsqrt(1 - γₜ)) * ϵ) + σₜ * z
    return xₙ


# Diffusion


def corrupt(
    inputs: Tensor,      # x₀
    time: Tensor,        # t
    schedule: Callable,  # ᾱ schedule
) -> Tensor:
    """q sampler: q(xₜ | xₒ) ~ N(xₒ * √ᾱₜ, (1 - ᾱₜ)I)
    Arbitrary time q-sampler for forward diffusion processing (corruption).

    Reference: Ho et al. "Denoising Diffusion Probabilistic Models". 2020.
        https://arxiv.org/abs/2006.11239
    """
    noise = torch.randn_like(inputs)  # ϵ

    signal_rate = torch.sqrt(schedule(time))     # √ᾱₜ
    noise_rate = torch.sqrt(1 - schedule(time))  # √(1 - ᾱₜ)

    signal_rate = utils.append_dims(signal_rate, inputs.ndim)
    noise_rate = utils.append_dims(noise_rate, inputs.ndim)
    return signal_rate * inputs + noise_rate * noise


class TextSed(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        embed_mat: NamedTensor["vocab", "embed"],
        *,
        use_self_cond: bool = True,
        noise_schedule: Callable = get_noise_schedule("cosine"),
        bottleneck_dim: Optional[int] = None,
        mask_type: str = "span",
        max_num_spans: int = 9,
        prefix_rate: float = 0.75,
    ):
        super().__init__()
        self.model = model
        self.use_self_cond = use_self_cond
        self.noise_schedule = noise_schedule

        assert mask_type in ("span", "prefix", "random")
        self.mask_type = mask_type
        self.mask_max_num_spans = max_num_spans
        self.mask_prefix_rate = prefix_rate

        _, embed_dim = embed_mat.shape
        self.embed_dim = embed_dim
        self.hidden_size = bottleneck_dim or embed_dim

        # Discrete-to-continuous fixed read-in matrix: E ϵ Rⱽˣᴰ
        self.read_in = nn.Sequential(
            PretrainedEmbedding(embed_mat, use_normalization=True),
            *[
                # Bottleneck layer to shrink word embeddings: D → D'
                nn.Linear(embed_dim, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
            ]
            if bottleneck_dim
            else [nn.Identity()],
        )
        # Continous-to-discrete learnable read-out matrix as an LM head
        self.read_out = nn.Sequential(
            *[
                # "Add a linear output projection layer E′ which takes the output of
                # the transformer y ∈ Rᴺˣᴰ and projects each element (yᵢ) 1 ≤ i ≤ N
                # back to the same size as the word embeddings, `embed_dim`."
                nn.Linear(self.hidden_size, embed_dim),
                nn.LayerNorm(embed_dim),
            ]
            if bottleneck_dim
            else [nn.Identity()],
            # Initalize read-out (R) to: Eᵀ ϵ Rᴰˣⱽ
            PretrainedUnembedding(embed_mat, use_renormalization=False),
        )  # E′

    def _get_conditioning_mask(
        self,
        batch_size: int,
        num_pos: int,
        device: torch.device,
    ) -> NamedTensor["batch", "pos"]:
        """Returns a mask for the conditioning positions."""
        masks = []
        if self.mask_type == "span":
            mask_func = functools.partial(get_span_mask, max_num_spans=self.mask_max_num_spans)
        elif self.mask_type == "prefix":
            mask_func = functools.partial(get_prefix_mask, rate=self.mask_prefix_rate)
        # TODO: Try batching this better - no loop!
        for _ in range(batch_size):
            masks.append(mask_func(num_pos))
        return torch.stack(masks).to(device)

    def forward(
        self,
        input_ids: NamedTensor["batch", "pos", "embed"],
        attention_mask: Optional[NamedTensor["batch", "pos"]] = None,
        cond_mask: Optional[NamedTensor["batch", "pos"]] = None,
        use_self_cond: bool = True,
    ) -> NamedTensor["batch", "pos", "embed"]:
        """
        Args:
            input_ids: The input token sequence.
            attention_mask: The attention mask for the input sequence.
            cond_mask: Mask with 1s for conditioning positions and 0s for infilling positions.
        """
        batch_size, num_pos = input_ids.shape[0], input_ids.shape[1]
        attention_mask = utils.default(
            attention_mask, torch.ones((batch_size, num_pos), device=input_ids.device)
        )

        # Discrete-to-continuous token embeddings
        embeds = self.read_in(input_ids)

        # Select random timesteps
        time = torch.rand((batch_size,), device=embeds.device)
        noisy_embeds = corrupt(embeds, time, schedule=self.noise_schedule)

        # Create masks for conditioning and infilling positions
        attention_mask: NamedTensor["batch", "pos", "1"] = attention_mask[..., None]
        cond_mask = utils.default(
            cond_mask,
            self._get_conditioning_mask(batch_size, num_pos, device=input_ids.device),
        )
        cond_mask: NamedTensor["batch", "pos", "1"] = cond_mask[..., None]
        # Remove padding positions from the conditioning/infilling masks
        # TODO (jon-tow): We shouldn't need to do this - remove this once verified
        cond_mask = cond_mask * attention_mask
        infill_mask = (1 - cond_mask) * attention_mask

        # Create re-usable embeddings
        noisy_embeds = infill_mask * noisy_embeds
        cond_embeds = torch.zeros_like(embeds, dtype=embeds.dtype)
        prev_embeds = torch.zeros_like(embeds, dtype=embeds.dtype)

        # Compute self-conditioning estimate and get the conditioning embeddings
        if use_self_cond and random.random() > 0.5:
            with torch.no_grad():
                prev_embeds = self.model(
                    noisy_embeds=noisy_embeds,
                    cond_embeds=cond_embeds,
                    prev_embeds=prev_embeds,
                    infill_mask=infill_mask,
                    time=time,
                ).detach()
            # Get the ("clean") conditioning embeddings: c1 c2 n1 n2 n3 c3 -> c1 c2 0 0 0 c3
            cond_embeds = cond_mask * embeds

        # Predict embeddings
        pred_embeds = self.model(
            noisy_embeds=noisy_embeds,
            cond_embeds=cond_embeds,
            prev_embeds=infill_mask * prev_embeds,
            infill_mask=infill_mask,
            time=time,
        )

        # Diffusion loss
        loss_mse = F.mse_loss(pred_embeds, embeds, reduction="mean")

        # Reconstruction loss
        logits = self.read_out(pred_embeds)
        loss_recon = cross_entropy_loss(logits, labels=input_ids, mask=attention_mask)

        total_loss = loss_mse + loss_recon["loss"]
        return total_loss, utils.flatten_dict(
            dict(
                total_loss=total_loss.item(),
                loss_mse=loss_mse.item(),
                loss_recon=loss_recon["loss"].item(),
                correct=loss_recon["correct"],
            )
        )

    @torch.no_grad()
    def generate(
        self,
        shape: NamedTensor["batch", "pos", "embed"],
        num_steps: int,
        *,
        sampler: Callable = ddim_step,
        use_clamp: bool = False,
        time_delta: float = 0.0,
        input_ids: Optional[NamedTensor["batch", "pos"]] = None,
        cond_mask: Optional[NamedTensor["batch", "pos"]] = None,
        guide_name: Optional[str] = None,
        guide_scale: float = 1.0,
        device: Device = "cuda:0",
    ) -> NamedTensor:
        """p sampler
        Sampler for the reverse diffusion process (denoising).

        Args:
            time_delta: Asymmetric time interval shift, t → (t - Δ).
            input_ids: Conditioning input sequence.
            cond_mask: The conditioning mask for the input sequence.
            guidance_name: The name of the guidance technique to use for the
                generation. Choices: ["self_guidance", "class_free_guidance"]
            guide_scale: The guidance scale >= 1.0.
            use_clamp: Whether to clamp predicted embeddings to the range
                [-1, 1] before each diffusion sampling step.
        """
        use_guidance = utils.exists(guide_name)
        cond_mask: NamedTensor["batch", "pos", "1"] = utils.default(
            cond_mask, torch.zeros(shape[:-1], device=device)[..., None]).bool()
        infill_mask: NamedTensor["batch", "pos", "1"] = (~cond_mask).float()

        # Sample start embedding from the normal prior eₜ ~ qₜ
        embed_t = torch.randn(shape, device=device)  # eₜ
        embed_pred = torch.zeros_like(embed_t)       # eₒ

        for step in range(num_steps):
            # Get time for current and next states. NOTE: (1 - ...) to process in reverse
            time_now = torch.tensor([1 - step / num_steps], device=device)
            time_next = torch.tensor([
                torch.maximum(
                    torch.tensor(1 - (step + 1 + time_delta) / num_steps),
                    torch.tensor(0.0),
                )
            ], device=device)

            if use_guidance and guide_name == "self_guidance":
                # Predict start embeds (eₒ) without self-conditioning
                embed_pred_uncond = self.model(
                    noisy_embeds=embed_t,
                    cond_embeds=torch.zeros_like(embed_t),
                    prev_embeds=torch.zeros_like(embed_t),
                    infill_mask=infill_mask,
                    time=time_now,
                )
                # Predict start embeds (eₒ) with self-conditioning
                embed_pred_selfcond = self.model(
                    noisy_embeds=embed_t,
                    cond_embeds=torch.zeros_like(embed_t),
                    prev_embeds=infill_mask * embed_pred_uncond,
                    infill_mask=infill_mask,
                    time=time_now,
                )
                # Apply self-conditioning guidance
                embed_pred = guide_scale * embed_pred_selfcond + (1.0 - guide_scale) * embed_pred_uncond
            elif use_guidance and guide_name == "class_free_guidance":
                # Predict start embeds (eₒ) without conditioning
                embed_pred_uncond = self.model(
                    noisy_embeds=embed_t,
                    cond_embeds=torch.zeros_like(embed_t),
                    prev_embeds=torch.zeros_like(embed_t),
                    infill_mask=infill_mask,
                    time=time_now,
                )
                # Predict start embeds (eₒ) with conditioning inputs
                if utils.exists(input_ids):
                    cond_embeds = self.read_in(input_ids)
                else:
                    cond_embeds = torch.zeros_like(embed_t)
                embed_pred_cond = self.model(
                    noisy_embeds=embed_t,
                    cond_embeds=cond_mask * cond_embeds,
                    prev_embeds=infill_mask * embed_pred_uncond,
                    infill_mask=infill_mask,
                    time=time_now
                )
                # Apply classifier-free guidance
                embed_pred = guide_scale * embed_pred_cond + (1.0 - guide_scale) * embed_pred_uncond
            else:
                # Self-conditioned prediction using the previous predictions, ẽₒ
                embed_pred = self.model(
                    noisy_embeds=infill_mask * embed_t,
                    cond_embeds=cond_mask * embed_t,
                    prev_embeds=infill_mask * embed_pred,
                    infill_mask=infill_mask,
                    time=time_now,
                )

            if use_clamp:
                # Clamping Trick (see footnote 6 in the paper):
                #   The model additionally maps the predicted vector fθ(xₜ, t) to
                #   its nearest word embedding sequence.
                # Li et al. "Diffusion-LM Improves Controllable Text Generation". 2022
                embed_pred = torch.clamp(embed_pred, -1.0, 1.0)

            # Estimate embeds at time_next eₜ₋₁
            embed_t = sampler(
                embed_t, embed_pred, time_now, time_next, self.noise_schedule
            )

        # Token decoding: continous embeddings to discrete tokens
        logits = self.read_out(embed_pred)
        tokens = torch.argmax(logits, dim=-1)
        return tokens
