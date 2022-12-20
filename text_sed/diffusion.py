import functools
import random
from typing import Callable, Literal, NewType, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from text_sed.layers import PretrainedEmbedding, PretrainedUnEmbedding, get_span_mask

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
    targets: NamedTensor["batch", "pos"],
    mask: Optional[NamedTensor["batch", "pos", "1"]] = None,
) -> float:
    """
    Args:
        logits: The unnormalized label scores.
        targets: The ground truth labels. NOTE: These will be one-hot encoded.
    """
    if mask is None:
        mask = torch.ones_like(targets)[:, :, None]

    num_ids = mask.sum()

    logits -= torch.max(logits, dim=-1, keepdim=True)[0]
    one_hot_targets = F.one_hot(targets, num_classes=logits.shape[-1])
    predicted_logits = torch.sum(one_hot_targets * logits, dim=-1)
    loss = -predicted_logits + torch.logsumexp(logits, dim=-1)
    loss = torch.sum(loss[:, :, None] * mask) / num_ids

    # Compute the fraction of correct predictions per batch:
    correct = (torch.argmax(logits, dim=-1) == targets).float()[:, :, None]
    correct = torch.sum(correct * mask) / mask.sum()
    return dict(loss=loss, correct=correct)


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
) -> Tensor:  # xₜ₋₁
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
    pred_inputs: Tensor,  # x̃₀
    time_now: Tensor,  # t
    time_next: Tensor,  # t - 1
    schedule: Callable,
) -> Tensor:  # xₜ₋₁
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
    inputs: Tensor,  # x₀
    time: Tensor,  # t
    schedule: Callable,  # ᾱ schedule
) -> Tensor:
    """q sampler: q(xₜ | xₒ) ~ N(xₒ * √ᾱₜ, (1 - ᾱₜ)I)
    Arbitrary time q-sampler for forward diffusion processing (corruption).

    Reference: Ho et al. "Denoising Diffusion Probabilistic Models". 2020.
        https://arxiv.org/abs/2006.11239
    """
    noise = torch.randn_like(inputs)  # ϵ

    signal_rate = torch.sqrt(schedule(time))  # √ᾱₜ
    noise_rate = torch.sqrt(1 - schedule(time))  # √(1 - ᾱₜ)

    signal_rate = utils.append_dims(signal_rate, inputs.ndim)
    noise_rate = utils.append_dims(noise_rate, inputs.ndim)
    return signal_rate * inputs + noise_rate * noise


class TextSed(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        embed_mat: NamedTensor["vocab", "embed"],
        use_self_cond: bool = True,
        noise_schedule: Callable = get_noise_schedule("cosine"),
        bottleneck_dim: Optional[int] = None,
        max_num_spans: int = 9,
    ):
        super().__init__()
        self.model = model
        self.use_self_cond = use_self_cond
        self.noise_schedule = noise_schedule
        self.max_num_spans = max_num_spans

        _, embed_dim = embed_mat.shape
        self.embed_dim = embed_dim
        self.hidden_size = bottleneck_dim or embed_dim

        # Discrete-to-continuous fixed read-in matrix: E ϵ Rⱽˣᴰ
        self.read_in = nn.Sequential(
            PretrainedEmbedding(embed_mat, use_normalization=True),
            *[
                # Bottleneck layer to shrink word embeddings: D → D'
                nn.Linear(embed_dim, self.hidden_size),
                # nn.LayerNorm(self.hidden_size),
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
            PretrainedUnEmbedding(embed_mat, use_renormalization=False),
        )  # E′

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
        """
        batch_size, num_pos = input_ids.shape[0], input_ids.shape[1]
        attention_mask: NamedTensor["batch", "pos", "1"] = attention_mask[:, :, None]
        # utils.print_rank_0(f"input_ids.shape: {input_ids.shape}")
        # utils.print_rank_0(f"attention_mask.shape: {attention_mask.shape}")

        # Discrete-to-continuous data embedding (token/word -> embedding)
        embeds = self.read_in(input_ids)
        # utils.print_rank_0(f"embeds.shape: {embeds.shape}")

        # Get random span masks if not provided
        if cond_mask is None:
            # Conditioning Mask: 0s for condition position and 1s for infilling positions
            # c1 c2 n1 n2 n3 c3 -> 0 0 1 1 1 0
            cond_mask = get_span_mask(num_pos, max_num_spans=self.max_num_spans)
            # TODO: When `cond_mask` are batched, replace the first `None` in the next line with `:`
            cond_mask: NamedTensor["batch", "pos", "1"] = cond_mask[
                None, :, None
            ].expand_as(attention_mask)

            # Infilling Mask: 1s for infilling positions and 0s for condition positions
            # c1 c2 n1 n2 n3 c3 -> 1 1 0 0 0 1
            # infill_mask = ~cond_mask
        else:
            cond_mask: NamedTensor["batch", "pos", "1"] = cond_mask[:, :, None]

        # utils.print_rank_0(f"cond_mask.shape: {cond_mask.shape}")
        # utils.print_rank_0(f"cond_mask: {cond_mask[0]}")
        # utils.print_rank_0(f"attention_mask: {attention_mask[0]}")

        # Mask out padding positions using the attention mask
        cond_mask = cond_mask.to(attention_mask.device)
        cond_mask = attention_mask * cond_mask
        # utils.print_rank_0(f"cond_mask * attn mask: {cond_mask[0]}")
        # utils.print_rank_0(f"~cond_mask * attn mask: {1 - cond_mask[0]}")

        # Get the ("clean") conditioning embeddings: c1 c2 n1 n2 n3 c3 -> c1 c2 0 0 0 c3
        cond_embeds = (1 - cond_mask) * embeds * attention_mask  # Remember to re-mask the padding
        # utils.print_rank_0(f"embeds: {embeds[0]}")
        # utils.print_rank_0(f"cond_embeds: {cond_embeds[0]}")

        # Select random timesteps
        time = torch.rand((batch_size,), device=embeds.device)
        noisy_embeds = corrupt(embeds, time, schedule=self.noise_schedule)

        # Compute self-conditioning estimate
        prev_embeds = torch.zeros_like(noisy_embeds, dtype=noisy_embeds.dtype)
        if use_self_cond and random.random() > 0.5:
            with torch.no_grad():
                prev_embeds = self.model(
                    noisy_embeds=noisy_embeds,  # `x`
                    prev_embeds=prev_embeds,  # `p`
                    cond_embeds=cond_embeds,  # `c`
                    cond_mask=cond_mask,  # `m`
                    time=time,
                ).detach()

        # Predict embeddings
        pred_embeds = self.model(
            noisy_embeds=noisy_embeds,  # `x`
            prev_embeds=prev_embeds,  # `p`
            cond_embeds=cond_embeds,  # `c`
            cond_mask=cond_mask,  # `m`
            time=time,
        )
        # Diffusion loss (Masked MSE)
        # NOTE: Only compute mse loss on the infilling positions
        loss_mse = torch.sum(cond_mask * (pred_embeds - embeds) ** 2) / cond_mask.sum()

        # Get lm-head logits
        logits = self.read_out(pred_embeds)
        # Reconstruction loss
        loss_recon = cross_entropy_loss(logits, targets=input_ids, mask=attention_mask)

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
        shape: Shape,  # ["batch", "pos", "embed"],
        num_steps: int,
        *,
        sampler: Callable = ddim_step,
        use_clamp: bool = False,
        time_delta: float = 0.0,
        cond_embeds: Optional[NamedTensor["batch", "pos", "embed"]] = None,
        cond_mask: Optional[NamedTensor["batch", "pos"]] = None,
        guide_scale: Optional[float] = None,
        device: Device = "cuda:0",
    ) -> NamedTensor:
        """p sampler
        Sampler for the reverse diffusion process (denoising).

        TODO: Remove unicode characters from code after coming up with concise
        names.

        Args:
            time_delta: Asymmetric time interval shift, t → (t - Δ)
            use_clamp: Whether to clamp predicted embeddings to the range
                [-1, 1] before each diffusion sampling step.
        """

        if cond_embeds is None:
            cond_embeds = torch.zeros(shape, device=device)

        if cond_mask is None:
            cond_mask = torch.ones(shape, device=device)

        # Sample start embedding from the normal prior eₜ ~ qₜ
        eₜ_prev = torch.randn(shape, device=device)
        pred_eₒ = torch.zeros_like(eₜ_prev)
        for step in range(num_steps):
            # Get time for current and next states
            # (NOTE: (1 - ...) to process in reverse)
            time_now = torch.tensor([1 - step / num_steps], device=device)
            time_next = torch.tensor(
                [
                    torch.maximum(
                        torch.tensor(1 - (step + 1 + time_delta) / num_steps),
                        torch.tensor(0.0),
                    )
                ],
                device=device,
            )

            if (
                guide_scale is not None and cond_embeds is None
            ):  # Self-conditioning guidance
                # Predict start embeds (eₒ) without self-cond
                ũₒ = self.model(eₜ_prev, torch.zeros_like(eₜ_prev), time_now)
                # Predict start embeds (eₒ) with self-conditiong
                c̃ₒ = self.model(eₜ_prev, ũₒ, time_now)
                # Apply self-conditioning guidance
                pred_eₒ = guide_scale * c̃ₒ + (1.0 - guide_scale) * ũₒ
            # elif guide_scale is not None and conds is not None:  # Classifier Free Guidance
            #     # Predict start embeds (eₒ) without self-cond
            #     cond_embeds = self.read_in(conds)
            #     ũₒ = self.model(eₜ_prev, torch.zeros_like(eₜ_prev), time_now)
            #     # Predict start embeds (eₒ) with self-conditiong
            #     c̃ₒ = self.model(eₜ_prev, ũₒ, time_now)
            #     pred_eₒ = guide_scale * ũₒ
            else:
                # Self-conditioned prediction using the previous predictions, ẽₒ
                pred_eₒ = self.model(
                    noisy_embeds=eₜ_prev,
                    prev_embeds=pred_eₒ,
                    cond_embeds=cond_embeds,
                    cond_mask=cond_mask,
                    time=time_now,
                )

            if use_clamp:
                # Clamping Trick (see footnote 6 in the paper):
                #   The model additionally maps the predicted vector fθ(xₜ, t) to
                #   its nearest word embedding sequence.
                # Li et al. "Diffusion-LM Improves Controllable Text Generation". 2022
                pred_eₒ = torch.clamp(pred_eₒ, -1.0, 1.0)

            # Estimate embeds at time_next eₜ₋₁
            eₜ_prev = sampler(
                eₜ_prev, pred_eₒ, time_now, time_next, self.noise_schedule
            )

        # Token decoding: continous embeddings to discrete tokens
        logits = self.read_out(pred_eₒ)
        tokens = torch.argmax(logits, -1)
        return tokens


# def corrupt(
#     inputs: Tensor,      # x₀
#     time: Tensor,        # t
#     schedule: Callable,  # ᾱ schedule
#     mask: Optional[Tensor] = None,    # m
# ) -> Tensor:
#     """q sampler: q(xₜ | xₒ) ~ N(xₒ * √ᾱₜ, (1 - ᾱₜ)I)
#     Arbitrary time q-sampler for forward diffusion processing (corruption).

#     Args:
#         mask: The infilling mask. If None, add noise to all elements.
#     Reference: Ho et al. "Denoising Diffusion Probabilistic Models". 2020.
#         https://arxiv.org/abs/2006.11239
#     """
#     noise = torch.randn_like(inputs)  # ϵ
#     if mask is not None:
#         # Only add noise to infilling positions
#         noise = mask * noise
#         # TODO: Do we need to also mask inputs? What does this do the signal rate
#         # scaled positions?
#         infill_inputs = mask * inputs

#     signal_rate = torch.sqrt(schedule(time))     # √ᾱₜ
#     noise_rate = torch.sqrt(1 - schedule(time))  # √(1 - ᾱₜ)

#     signal_rate = utils.append_dims(signal_rate, inputs.ndim)
#     noise_rate = utils.append_dims(noise_rate, inputs.ndim)

#     # Add Gaussian noise to the embeddings for noisy (infilling) positions
#     noisy_inputs = signal_rate * infill_inputs + noise_rate * noise
#     clean_inputs = ~mask * inputs
#     # Add back the conditionining positions
#     return noisy_inputs + clean_inputs
