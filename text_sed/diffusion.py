import functools
import math
import random
from typing import Callable, Literal, NewType, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from text_sed.layers import PretrainedEmbedding, PretrainedUnEmbedding

from . import utils

Device = NewType("Device", torch.device)
DType = NewType("DType", torch.dtype)
Shape = NewType("Shape", Tuple[int, ...])
Tensor = NewType("Tensor", torch.Tensor)
NamedTensor = Literal  # **Naive** named tensor
Generator = NewType("Generator", torch.Generator)


# Noise Schedules
# Reference: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# @ unixpickle 🤘🥒


def get_noise_schedule(schedule_name: str, **kwargs) -> Callable:
    """Returns ᾱ schedules"""
    if schedule_name == "cosine":
        return cosine_alpha_bar_schedule(**kwargs)
    elif schedule_name == "linear":
        return linear_schedule(**kwargs)
    raise ValueError(f"Schedule `{schedule_name}` is not available.")


def linear_schedule(start: float, end: float) -> Tensor:
    """Linear noise-variance (β) schedule."""

    def scheduler(num_steps: int):
        return torch.linspace(start, end, num_steps)

    return scheduler


def cosine_alpha_bar(
    time: float,
    offset: Optional[float] = 0.0002,
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
    offset: Optional[float] = 0.0002,
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


def get_sampler(sampler_name: str, **kwargs) -> Callable:
    """Returns sampler step function"""
    if sampler_name == "ddim":
        return functools.partial(ddim_step, **kwargs)
    elif sampler_name == "ddpm":
        return functools.partial(ddpm_step, **kwargs)
    else:
        raise ValueError(f"Sampler `{sampler_name}` is not available.")


def ddim_step(
    noisy_inputs: Tensor,  # xₜ
    pred_inputs: Tensor,  # x̃₀
    time_now: Tensor,  # t
    time_next: Tensor,  # t - 1
    schedule: Callable,
    scale: float = 1.0,
) -> Tensor:  # xₜ₋₁
    """Denoising diffusion implicit model step with η = 0. Estimates x₀ at
    time_next with the DDIM updating rule.

    References:
    - Song et al. "Denoising Diffusion Implicit Models". 2020.
        https://arxiv.org/pdf/2010.02502.pdf
    - Lilian Weng.
        https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#speed-up-diffusion-model-sampling
    - Clamping Trick (see footnote 6 in the paper):
        Li et al. "Diffusion-LM Improves Controllable Text Generation". 2022
    """
    # TODO: Remove unicode characters after coming up with good var names.
    xₜ, x̃ₒ = noisy_inputs, pred_inputs
    x̃ₒ = x̃ₒ.clamp(-scale, scale)
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
    scale: float = 1.0,
) -> Tensor:               # xₜ₋₁
    """Denoising diffusion implicit model step with η = 1. Estimates x₀ at
    time_next with the DDPM updating rule.

    References:
    - Ho et al. "Denoising Diffusion Probabilistic Models". 2020.
        https://arxiv.org/abs/2006.11239
    - Clamping Trick (see footnote 6 in the paper):
        Li et al. "Diffusion-LM Improves Controllable Text Generation". 2022
    """
    xₜ, x̃ₒ = noisy_inputs, pred_inputs
    x̃ₒ = x̃ₒ.clamp(-scale, scale)
    γₜ = schedule(time_now)
    ᾱₜ = γₜ / schedule(time_next)
    σₜ = torch.sqrt(1 - ᾱₜ)
    z = torch.randn_like(σₜ)
    ϵ = (xₜ - torch.sqrt(γₜ) * x̃ₒ) * torch.rsqrt(1 - γₜ)
    xₙ = torch.rsqrt(ᾱₜ) * (xₜ - ((1 - ᾱₜ) * torch.rsqrt(1 - γₜ)) * ϵ) + σₜ * z
    return xₙ


# Diffusion


def corrupt(
    inputs: Tensor,  # x₀
    time: Tensor,    # t
    schedule: Callable,  # ᾱ schedule
) -> Tensor:
    """q sampler: q(xₜ | xₒ) ~ N(xₒ * √ᾱₜ, (1 - ᾱₜ)I)
    Arbitrary time q-sampler for forward diffusion processing (corruption).

    Reference
    - Ho et al. "Denoising Diffusion Probabilistic Models". 2020.
        https://arxiv.org/abs/2006.11239
    """
    noise = torch.randn_like(inputs)  # ϵ

    signal_rate = torch.sqrt(schedule(time))     # √ᾱₜ
    noise_rate = torch.sqrt(1 - schedule(time))  # √(1 - ᾱₜ)

    signal_rate = utils.append_dims(signal_rate, inputs.ndim)
    noise_rate = utils.append_dims(noise_rate, inputs.ndim)
    return signal_rate * inputs + noise_rate * noise


def cross_entropy_loss(
    logits: Tensor,
    targets: Tensor,
    z_loss: Optional[float] = 0.0,
) -> float:
    """Mesh-transformer-jax style cross entropy loss.
    Args:
        logits: The unnormalized label scores.
        targets: The ground truth labels. These should be one-hot encoded.
    """
    logits -= torch.max(logits, dim=-1, keepdim=True)[0]
    one_hot_targets = F.one_hot(targets, num_classes=logits.shape[-1])
    predicted_logits = torch.sum(one_hot_targets * logits, dim=-1)
    loss = -predicted_logits + torch.logsumexp(logits, dim=-1)
    # Add the auxiliary z-loss term
    loss += torch.mean(z_loss * (1e-4 * torch.square(torch.logsumexp(logits, dim=-1))))
    loss = reduce(loss, "b ... -> 1", "mean")[0]
    # Compute the fraction of correct predictions per batch:
    correct = (torch.argmax(logits, dim=-1) == targets).float()
    correct = reduce(correct, "b ... -> 1", "mean")[0]
    return dict(loss=loss, correct=correct)


class TextSed(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        embed_mat: NamedTensor["vocab", "embed"],
        use_self_cond: bool = True,
        noise_schedule: Callable = get_noise_schedule("cosine"),
        bottleneck_dim: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.use_self_cond = use_self_cond
        self.noise_schedule = noise_schedule

        _, embed_dim = embed_mat.shape

        # Discrete-to-continuous fixed read-in matrix: E ϵ Rⱽˣᴰ
        self.read_in = nn.Sequential(
            PretrainedEmbedding(embed_mat, use_normalization=True),
            *[
                # Bottleneck layer to shrink word embeddings: D → D'
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, bottleneck_dim)
            ] if bottleneck_dim else [nn.Identity()],
        )
        # Continous-to-discrete learnable read-out matrix
        self.read_out = nn.Sequential(
            *[
                # "Add a linear output projection layer E′ which takes the output of
                # the transformer y ∈ Rᴺˣᴰ and projects each element (yᵢ) 1 ≤ i ≤ N
                # back to the same size as the word embeddings, `embed_dim`."
                nn.Linear(bottleneck_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            ] if bottleneck_dim else [nn.Identity()],
            # Initalize read-out (R) to: Eᵀ ϵ Rᴰˣⱽ
            PretrainedUnEmbedding(embed_mat, use_renormalization=False),
         ) # E′

    def forward(
        self,
        inputs: NamedTensor["batch", "pos", "embed"],
        use_self_cond: Optional[bool] = True,
        z_loss: Optional[float] = 0.0,
    ) -> NamedTensor["batch", "pos", "embed"]:
        """
        Args:
            - inputs: The input token sequence.
        """
        batch_size = inputs.shape[0]

        # Discrete-to-continuous data embedding (token/word -> embedding)
        embeds = self.read_in(inputs)

        # Select random timesteps and corrupt
        time = torch.rand((batch_size,), device=embeds.device)
        noisy_embeds = corrupt(embeds, time, schedule=self.noise_schedule)

        # Compute self-conditioning estimate
        cond_embeds = torch.zeros_like(noisy_embeds, dtype=noisy_embeds.dtype)
        if use_self_cond and random.random() > 0.5:
            with torch.no_grad():
                cond_embeds = self.model(noisy_embeds, cond_embeds, time).detach()

        # Predict embeddings
        pred_embeds = self.model(noisy_embeds, self_cond=cond_embeds, time=time)
        logits = self.read_out(pred_embeds)

        # Diffusion and Reconstruction loss
        loss_mse = F.mse_loss(pred_embeds, target=embeds, reduction="mean")
        loss_recon = cross_entropy_loss(logits, targets=inputs, z_loss=z_loss)
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
    def sample(
        self,
        shape: Shape,
        num_steps: int,
        *,
        sampler: Optional[Callable] = ddim_step,
        # conds: Optional[NamedTensor["batch", "pos", "embed"]] = None,
        # guide_scale: Optional[float] = None,
        time_delta: Optional[float] = 0.0,
        device: Optional[Device] = "cuda:0",
    ) -> NamedTensor:
        """p sampler
        Sampler for the reverse diffusion process (denoising).

        TODO: Remove unicode characters from code after coming up with concise
        names.

        Args:
            time_delta: Asymmetric time interval shift, t → (t - Δ)
        """
        # Sample start embedding from the normal prior eₜ ~ qₜ
        eₜ = torch.randn(shape, device=device)
        ẽₒ = torch.zeros_like(eₜ)

        for step in range(num_steps):
            # Get time for current and next states
            # (NOTE: (1 - ...) to process in reverse)
            time_now = torch.tensor([1 - step / num_steps], device=device)
            time_next = torch.tensor([
                torch.maximum(
                    torch.tensor(1 - (step + 1 + time_delta) / num_steps),
                    torch.tensor(0.0),
                )],
                device=device,
            )

            # if guide_scale is not None and conds is None:  # Self-conditioning guidance
            #     # Predict start embeds (eₒ) without self-cond
            #     ũₒ = self.model(eₜ, torch.zeros_like(eₜ), time_now)
            #     # Predict start embeds (eₒ) with self-conditiong
            #     c̃ₒ = self.model(eₜ, ũₒ, time_now)
            #     # Apply self-conditioning guidance
            #     ẽₒ = guide_scale * c̃ₒ + (1.0 - guide_scale) * ũₒ 
            # elif guide_scale is not None and conds is not None:  # Classifier Free Guidance
            #     # Predict start embeds (eₒ) without self-cond
            #     cond_embeds = self.read_in(conds)
            #     ũₒ = self.model(eₜ, torch.zeros_like(eₜ), time_now)
            #     # Predict start embeds (eₒ) with self-conditiong
            #     c̃ₒ = self.model(eₜ, ũₒ, time_now)
            #     # ẽₒ = guide_scale * ũₒ
            # else:
            # Self-conditioned prediction using the previous predictions, ẽₒ
            ẽₒ = self.model(eₜ, ẽₒ, time_now)

            # Estimate embeds at time_next eₜ₋₁
            eₜ = sampler(eₜ, ẽₒ, time_now, time_next, self.noise_schedule)

        # Token decoding: continous embeddings to discrete tokens
        logits = self.read_out(ẽₒ)
        tokens = torch.argmax(logits, -1)
        return tokens
