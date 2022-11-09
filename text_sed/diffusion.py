import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import EmbeddingReader
from .utils import flatten_dict

from einops import reduce

# Torch Type Hints
from typing import Callable, NewType, Literal, Optional, Tuple
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
    elif schedule_name == "sqrt":
        return sqrt_alpha_bar_schedule(**kwargs)
    elif schedule_name == "linear":
        return linear_schedule(**kwargs)
    raise ValueError(f"Schedule `{schedule_name}` is not available.")


def linear_schedule(start: float, end: float) -> Tensor:
    """Linear noise-variance (β) schedule."""
    def scheduler(num_steps: int):
        return torch.linspace(start, end, num_steps)
    return scheduler


def betas_for_alpha_bar(
    alpha_bar: Callable[[float], Tensor],
    max_beta: Optional[float] = 0.999,
) -> Tensor:
    """Create a β schedule that discretizes the given ᾱ function,
    ᾱ[t] = Πᵗα[i] where α[i] = (1 - β[i]) from time t = [0, 1].

    Args:
        num_steps: The number of betas to create.
        alpha_bar: Callable that takes a timestep argument from
            [0, 1] and produces the cumulative prod of (1 - β)
        max_beta: The max β to use.
    """
    def scheduler(num_steps: int):
        betas = []
        for i in range(num_steps):
            a1 = alpha_bar(i / num_steps)
            a2 = alpha_bar((i + 1) / num_steps)
            betas.append(min(1 - a2 / a1, max_beta))
        return torch.Tensor(betas)
    return scheduler


def cosine_beta_schedule(
    offset: Optional[float] = 0.0002,
    max_beta: Optional[float] = 0.999,
) -> Tensor:
    """Cosine noise-variance (β) scheduler

    Reference: Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models".
        2021. https://arxiv.org/pdf/2102.09672.pdf

    Args:
        offset: Small offset to prevent βₜ from beeing too small near
            t = 0.
    """
    def scheduler(num_steps: int):
        return torch.cos(((num_steps + offset) / (1 + offset)) * (torch.pi / 2)) ** 2
    return betas_for_alpha_bar(scheduler, max_beta)


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
    return torch.cos(((time + offset) / (1 + offset)) * (torch.pi / 2)) ** 2


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


def sqrt_alpha_bar(
    time: float,
    # max_t: int,  # TODO: Use if time is NOT in [0, 1)
    offset: Optional[float] = 1e-4,
) -> Tensor:
    """Square-root noise schedule - useful for language modeling.

    Reference:
    - Li et al. "Diffusion-LM Improves Controllable Text Generation". 2022.
        https://arxiv.org/pdf/2205.14217.pdf#page=15

    TODO: This leads to NaNs

    Args:
        time: Continous time-step in [0, 1)
        offset: Start noise level.
    """
    # Normalize if your input is not in [0, 1)
    # return 1.0 - torch.sqrt((time / max_t) + offset)
    return 1.0 - torch.sqrt(time + offset)


def sqrt_alpha_bar_schedule(
    offset: Optional[float] = 1e-4,
) -> Callable:
    def scheduler(num_steps: float):
        return sqrt_alpha_bar(time=num_steps, offset=offset)

    return scheduler


# Diffusion


def left_broadcast_to(x: Tensor, shape: Shape):
    x = x.reshape(x.shape + (1,) * (len(shape) - x.ndim))
    return torch.broadcast_to(x, shape)


def corrupt(
    inputs: Tensor,      # x₀
    time: Tensor,        # t
    schedule: Callable,  # ᾱ schedule
) -> Tensor:
    """q sampler: q(xₜ | xₒ) ~ N(xₒ * √ᾱₜ, (1 - ᾱₜ)I)
    Arbitrary time q-sampler for forward diffusion processing (corruption).

    Reference:
    - "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
        https://arxiv.org/abs/2006.11239
    """
    noise = torch.randn(inputs.shape, device=inputs.device)  # ϵ

    signal_rate = torch.sqrt(schedule(time))     # √ᾱₜ
    noise_rate = torch.sqrt(1 - schedule(time))  # √(1 - ᾱₜ)

    signal_rate = left_broadcast_to(signal_rate, inputs.shape)
    noise_rate = left_broadcast_to(noise_rate, inputs.shape)
    return signal_rate * inputs + noise_rate * noise


def ddim_step(
    noisy_inputs: Tensor,  # xₜ
    pred_inputs: Tensor,   # x̃₀
    time_now: Tensor,      # t
    time_next: Tensor,     # t - 1
    schedule: Callable,
    scale: Optional[float] = 1.0,
) -> Tensor:               # xₜ₋₁
    """Denoising diffusion implicit model step with η = 0. Estimates x₀ at
    time_next with the DDIM updating rule.

    References:
    - Song et al. "Denoising Diffusion Implicit Models" 2020.
        https://arxiv.org/pdf/2010.02502.pdf
    - Lilian Weng.
        https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#speed-up-diffusion-model-sampling
    """
    # TODO: Remove unicode characters after coming up with good var names.
    xₜ, x̃ₒ = noisy_inputs, pred_inputs
    # x̃ₒ = torch.clip(x̃ₒ, -scale, scale)
    ᾱₜ, ᾱₙ = schedule(time_now), schedule(time_next)  # ᾱₙ = ᾱₜ₋₁
    ϵ = (xₜ - torch.sqrt(ᾱₜ) * x̃ₒ) * torch.rsqrt(1 - ᾱₜ)
    # Next estimate (xₙ := xₜ₋₁)
    xₙ = torch.sqrt(ᾱₙ) * x̃ₒ + torch.sqrt(1 - ᾱₙ) * ϵ
    return xₙ


def cross_entropy_loss(
    logits: Tensor,
    targets: Tensor,
    z_loss: Optional[float] = 0.0,
) -> float:
    """Cross-entropy loss function for logits and targets.

    If z_loss > 0, then an auxiliary loss equal to z_loss*log(z)^2
    will be added to the cross entropy loss (z = softmax normalization constant).
    The two uses of z_loss are:
      1. To keep the logits from drifting too far from zero, which can cause
         unacceptable roundoff errors in bfloat16.
      2. To encourage the logits to be normalized log-probabilities.

    Reference:
    - https://github.com/google-research/t5x/blob/45103dd897e214d4e7818de5c64a3811ff6daf25/t5x/losses.py#L26

    Args:
        logits: The unnormalized label scores.
        targets: The ground truth labels. These should be one-hot encoded.
    """
    logits_sum = torch.logsumexp(logits, dim=-1, keepdim=True)
    log_softmax = logits - logits_sum
    one_hot_targets = F.one_hot(targets, num_classes=logits.shape[-1])
    cross_entropy = -torch.sum(one_hot_targets * log_softmax, dim=-1)

    # Add aux z-loss term
    log_z = torch.squeeze(logits_sum, dim=-1)
    aux_z_loss = z_loss * torch.square(log_z)

    loss = cross_entropy + aux_z_loss
    return reduce(loss, "b ... -> 1", "mean")[0]


class TextSed(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        embed_mat: NamedTensor["vocab", "embed"],
        embed_scale: Optional[float] = None,
        noise_schedule: Optional[Callable] = get_noise_schedule("cosine"),
        use_self_cond: Optional[bool] = True,
    ):
        super().__init__()
        self.model = model
        self.use_self_cond = use_self_cond
        self.noise_schedule = noise_schedule
        self.reader = EmbeddingReader(embed_mat, embed_scale)

    @torch.no_grad()
    def sample(
        self,
        shape: Shape,
        num_steps: int,
        time_delta: Optional[float] = 0.0,
        use_self_cond: Optional[bool] = True,
        step_fn: Optional[Callable] = ddim_step,
        # dtype: Optional[DType] = None
        device: Optional[Device] = "cuda",
    ) -> NamedTensor:
        """p sampler
        Sampler for the reverse diffusion process (denoising).

        Args:
            time_delta: Asymmetric time interval shift, t → (t - Δ)
        """
        # Sample from the normal prior xₜ ~ qₜ
        rand_embeds = torch.randn(shape, device=device)
        pred_embeds = torch.zeros_like(rand_embeds, device=device)

        for step in range(num_steps):
            # Get time for current and next states
            # (NOTE: (1 - ...) to process in reverse)
            time_now = torch.tensor([1 - step / num_steps], device=device)
            time_next = torch.tensor([
                torch.maximum(
                    torch.tensor(1 - (step + 1 + time_delta) / num_steps),
                    torch.tensor(0.0)
                )], device=device)

            # Predict start embeds
            if not use_self_cond:
                pred_embeds = torch.zeros_like(rand_embeds, device=device)
            pred_embeds = self.model(
                torch.concat([rand_embeds, pred_embeds], -1),
                time_now)

            # Estimate embeds at time_next
            rand_embeds = step_fn(
                rand_embeds,
                pred_embeds,
                time_now,
                time_next,
                self.noise_schedule)

        # Token decoding: continous embeddings to discrete tokens
        pred_logits = self.reader.unembed(pred_embeds)
        pred_tokens = torch.argmax(pred_logits, -1)
        return pred_tokens

    def forward(
        self,
        input: NamedTensor["batch", "pos", "embed"],
        use_self_cond: Optional[bool] = True,
        z_loss: Optional[float] = 0.0,
    ) -> NamedTensor["batch", "pos", "embed"]:
        batch_size = input.shape[0]

        # Discrete-to-continuous data embedding (token/word -> embedding)
        embeds = self.reader.embed(input)

        # Select random timesteps and corrupt
        time = torch.rand((batch_size,), device=embeds.device)
        noisy_embeds = corrupt(embeds, time, schedule=self.noise_schedule)

        # Compute self-conditioning estimate
        cond_embeds = torch.zeros_like(noisy_embeds, dtype=noisy_embeds.dtype)
        with torch.no_grad():
            if use_self_cond and torch.rand((1,)).item() > 0.5:
                cond_embeds = self.model(
                    torch.concat([noisy_embeds, cond_embeds], -1),
                    time
                )

        # Predict embeddings
        pred_embeds = self.model(torch.concat(
            [noisy_embeds, cond_embeds], -1), time)
        logits = self.reader.unembed(pred_embeds)

        # Diffusion and Reconstruction loss
        loss_mse = F.mse_loss(pred_embeds, target=embeds, reduction='mean')
        loss_recon = cross_entropy_loss(logits, targets=input, z_loss=z_loss)
        loss = loss_mse + loss_recon

        return loss, flatten_dict(
            dict(
                total_loss=loss,
                loss_mse=loss_mse,
                loss_recon=loss_recon,
            )
        )
