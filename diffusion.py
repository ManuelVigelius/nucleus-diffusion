"""Nucleus-constrained discrete diffusion: schedule, corruption operator, training inputs, sampling.

Conventions
-----------
x0, x : LongTensor (B, C, H, W) with values in [0, K)
probs : FloatTensor (B, C, H, W, K)
alphas: python list, alphas[k-1] = alpha_k = P(subpixel keeps its true value in step k's input)
models: list, models[k-1] = M_k
"""
import math
from contextlib import nullcontext

import torch

K = 16  # number of levels per subpixel (4 bits)


def keep_schedule(num_steps, kind="cosine"):
    """alpha_k for k = 1..N. alpha_1 = 0 (pure noise), increasing afterwards."""
    alphas = []
    for k in range(1, num_steps + 1):
        t = (k - 1) / num_steps
        if kind == "cosine":
            a = 1.0 - math.cos(math.pi / 2 * t)
        elif kind == "linear":
            a = t
        else:
            raise ValueError(f"unknown schedule {kind!r}")
        alphas.append(a)
    return alphas


def nucleus_mask(probs, top_p):
    """Bool mask (..., K) of the top-p nucleus: the smallest set of values whose cumulative
    probability is >= top_p. Always contains the argmax. top_p >= 1 -> all values."""
    if top_p >= 1.0:
        return torch.ones(probs.shape, dtype=torch.bool, device=probs.device)
    sp, idx = probs.sort(dim=-1, descending=True)
    mass_before = sp.cumsum(dim=-1) - sp       # probability mass strictly before each entry
    keep_sorted = mass_before < top_p           # include entries until mass >= top_p is reached
    keep_sorted[..., 0] = True
    return torch.zeros_like(keep_sorted).scatter(-1, idx, keep_sorted)


def uniform_from_mask(mask):
    """Draw one index uniformly from the True entries of `mask` (..., K), vectorised.
    argmax of iid uniforms restricted to the allowed set is uniform over that set."""
    r = torch.rand(mask.shape, device=mask.device)
    return r.masked_fill(~mask, -1.0).argmax(dim=-1)


def sample_from_probs(probs, top_p=1.0):
    """Sample from `probs` restricted to its top-p nucleus (renormalised).
    Uses the exponential race: argmax(p / Exp(1)) ~ Categorical(p / sum p)."""
    p = probs * nucleus_mask(probs, top_p)
    e = torch.empty_like(p).exponential_()
    return (p / e).argmax(dim=-1)


def corrupt(x0, probs, alpha, top_p, include_x0, return_stats=False):
    """Per-subpixel: keep x0 with prob alpha, else replace by a value drawn uniformly from the
    top-p nucleus of `probs` (optionally unioned with x0). top_p = 1 -> uniform discrete diffusion
    (probs may be None in that case)."""
    nuc = None
    if top_p >= 1.0:
        noise = torch.randint(0, K, x0.shape, device=x0.device)
    else:
        nuc = nucleus_mask(probs, top_p)
        allowed = nuc
        if include_x0:
            allowed = allowed.scatter(-1, x0.unsqueeze(-1), True)
        noise = uniform_from_mask(allowed)
    keep = torch.rand(x0.shape, device=x0.device) < alpha
    x = torch.where(keep, x0, noise)
    if not return_stats:
        return x
    if nuc is None:
        stats = {"nucleus_size": float(K), "x0_outside_nucleus": 0.0}
    else:
        stats = {
            "nucleus_size": nuc.sum(-1).float().mean().item(),
            "x0_outside_nucleus": 1.0 - nuc.gather(-1, x0.unsqueeze(-1)).float().mean().item(),
        }
    stats["input_eq_x0"] = (x == x0).float().mean().item()
    return x, stats


@torch.no_grad()
def make_inputs(x0, frozen_models, alphas, top_p, include_x0, amp=nullcontext, return_stats=False):
    """Training input for M_k where k = len(frozen_models) + 1. Regenerated fresh every call.
    frozen_models = [M_1, ..., M_{k-1}] (EMA weights, eval mode).
    Returns (x, stats); stats describes the last corruption (None for k = 1 or if not requested)."""
    k = len(frozen_models) + 1
    if k == 1:
        return torch.randint(0, K, x0.shape, device=x0.device), None
    if top_p >= 1.0:
        # The nucleus is everything, so the frozen models' outputs do not matter: skip them.
        out = corrupt(x0, None, alphas[k - 1], 1.0, include_x0, return_stats)
        return out if return_stats else (out, None)
    x = torch.randint(0, K, x0.shape, device=x0.device)
    for j, model in enumerate(frozen_models[:-1]):  # model = M_{j+1}, produces input for M_{j+2}
        with amp():
            probs = model(x).float().softmax(dim=-1)
        x = corrupt(x0, probs, alphas[j + 1], top_p, include_x0)
    with amp():
        probs = frozen_models[-1](x).float().softmax(dim=-1)
    out = corrupt(x0, probs, alphas[k - 1], top_p, include_x0, return_stats)
    return out if return_stats else (out, None)


def frozen_forwards_per_iter(k, top_p, marginal_stage1=True):
    """Number of frozen ViT forward passes needed to build one training input for M_k."""
    if top_p >= 1.0 or k == 1:
        return 0
    return k - 2 if marginal_stage1 else k - 1


@torch.no_grad()
def sample(models, alphas, shape, device, top_p=1.0, temperature=1.0, refine_steps=0,
           final="sample", amp=nullcontext, return_trajectory=False):
    """Generate images with all N stages, mirroring training.

    refine_steps: extra predictor-corrector passes per stage (k >= 2). After M_k predicts x0_hat,
    x0_hat is re-noised exactly like M_k's training inputs (keep prob alpha_k, noise uniform over
    M_{k-1}'s nucleus united with x0_hat) and M_k is applied again.
    """
    assert temperature > 0
    N = len(models)
    x = torch.randint(0, K, shape, device=device)
    prev_probs = None
    trajectory = []
    for k in range(1, N + 1):
        model = models[k - 1]
        passes = 1 + (refine_steps if k >= 2 else 0)
        for r in range(passes):
            if r > 0:
                x = corrupt(x0_hat, prev_probs, alphas[k - 1], top_p, include_x0=True)
            with amp():
                logits = model(x)
            probs = (logits.float() / temperature).softmax(dim=-1)
            if k == N and final == "argmax":
                x0_hat = probs.argmax(dim=-1)
            else:
                x0_hat = sample_from_probs(probs, top_p)
        trajectory.append((x, x0_hat))
        if k < N:
            x = corrupt(x0_hat, probs, alphas[k], top_p, include_x0=False)
            prev_probs = probs
    return (x0_hat, trajectory) if return_trajectory else x0_hat
