import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion import K, corrupt, keep_schedule, nucleus_mask, sample_from_probs  # noqa: E402


def chi2_uniform(values, support, n_bins=K):
    """Pearson chi-square statistic for `values` being uniform over `support`, plus dof."""
    counts = np.bincount(values, minlength=n_bins)
    assert counts[[i for i in range(n_bins) if i not in support]].sum() == 0, "value outside support"
    obs = counts[list(support)]
    exp = len(values) / len(support)
    return ((obs - exp) ** 2 / exp).sum(), len(support) - 1


def chi2_ok(stat, dof):
    # loose bound: mean dof, std sqrt(2 dof); 6 sigma -> essentially never flaky
    return stat < dof + 6 * np.sqrt(2 * dof)


def dist(*pairs):
    p = torch.zeros(K)
    for i, v in pairs:
        p[i] = v
    return p


@pytest.mark.parametrize("probs,top_p,expected", [
    (dist((3, 0.5), (7, 0.3), (1, 0.2)), 0.5, {3}),           # exactly reaches top_p with one value
    (dist((3, 0.5), (7, 0.3), (1, 0.2)), 0.51, {3, 7}),
    (dist((3, 0.5), (7, 0.3), (1, 0.2)), 0.8, {3, 7}),
    (dist((3, 0.5), (7, 0.3), (1, 0.2)), 0.81, {3, 7, 1}),
    (dist((3, 0.5), (7, 0.3), (1, 0.2)), 0.99, {3, 7, 1}),    # zero-prob values never enter
    (dist((5, 0.9), (2, 0.1)), 0.1, {5}),                     # always contains the argmax
    (dist((5, 0.9), (2, 0.1)), 0.0, {5}),
    (torch.full((K,), 1.0 / K), 0.25 + 1e-4, set(range(5))),  # uniform: ceil(top_p*K) values
    (dist((0, 0.6), (1, 0.4)), 1.0, set(range(K))),           # top_p = 1 -> everything
])
def test_nucleus_hand_made(probs, top_p, expected):
    m = nucleus_mask(probs.view(1, 1, 1, 1, K), top_p).view(K)
    assert set(torch.nonzero(m).flatten().tolist()) == expected


def test_nucleus_batched_matches_reference():
    torch.manual_seed(0)
    probs = torch.distributions.Dirichlet(torch.full((K,), 0.3)).sample((2, 3, 4, 5))
    m = nucleus_mask(probs, 0.9)
    for p, mm in zip(probs.view(-1, K), m.view(-1, K)):
        order = torch.argsort(p, descending=True)
        cum, ref = 0.0, set()
        for i in order.tolist():
            ref.add(i)
            cum += p[i].item()
            if cum >= 0.9:
                break
        assert set(torch.nonzero(mm).flatten().tolist()) == ref


def test_noise_uniform_within_nucleus():
    torch.manual_seed(0)
    n = 200_000
    probs = dist((2, 0.4), (9, 0.3), (11, 0.2), (4, 0.06), (0, 0.04)).expand(n, 1, 1, 1, K)
    x0 = torch.full((n, 1, 1, 1), 2)  # x0 inside the nucleus {2, 9, 11}
    x = corrupt(x0, probs, alpha=0.0, top_p=0.85, include_x0=False)
    stat, dof = chi2_uniform(x.flatten().numpy(), {2, 9, 11})
    assert chi2_ok(stat, dof), stat


def test_include_x0_adds_true_value_to_support():
    torch.manual_seed(0)
    n = 200_000
    probs = dist((2, 0.5), (9, 0.4), (11, 0.1)).expand(n, 1, 1, 1, K)
    x0 = torch.full((n, 1, 1, 1), 13)  # outside the nucleus {2, 9}
    x, st = corrupt(x0, probs, alpha=0.0, top_p=0.8, include_x0=True, return_stats=True)
    stat, dof = chi2_uniform(x.flatten().numpy(), {2, 9, 13})
    assert chi2_ok(stat, dof), stat
    assert st["x0_outside_nucleus"] == 1.0 and st["nucleus_size"] == 2.0
    # without include_x0, x0 is never produced
    x = corrupt(x0, probs, alpha=0.0, top_p=0.8, include_x0=False)
    assert set(x.unique().tolist()) == {2, 9}


def test_top_p_one_is_uniform_over_all_values():
    torch.manual_seed(0)
    n = 320_000
    x0 = torch.randint(0, K, (n, 1, 1, 1))
    peaked = dist((0, 1.0)).expand(n, 1, 1, 1, K)  # probs are ignored when top_p = 1
    for probs in (None, peaked):
        x = corrupt(x0, probs, alpha=0.0, top_p=1.0, include_x0=False)
        stat, dof = chi2_uniform(x.flatten().numpy(), set(range(K)))
        assert chi2_ok(stat, dof), stat
        assert abs((x == x0).float().mean().item() - 1 / K) < 0.005


def test_alpha_one_returns_x0():
    torch.manual_seed(0)
    x0 = torch.randint(0, K, (8, 3, 32, 32))
    probs = torch.softmax(torch.randn(8, 3, 32, 32, K) * 3, -1)
    for top_p in (0.5, 0.9, 1.0):
        for inc in (True, False):
            assert torch.equal(corrupt(x0, probs, 1.0, top_p, inc), x0)


def test_alpha_zero_never_keeps_x0_outside_noise_support():
    torch.manual_seed(0)
    n = 100_000
    x0 = torch.randint(0, K, (n, 1, 1, 1))
    probs = torch.zeros(n, 1, 1, 1, K)
    probs.scatter_(-1, ((x0 + 1) % K).unsqueeze(-1), 1.0)  # nucleus = {x0 + 1}, never x0
    x = corrupt(x0, probs, alpha=0.0, top_p=0.9, include_x0=False)
    assert (x == x0).sum() == 0
    assert torch.equal(x, (x0 + 1) % K)


def test_keep_rate_matches_alpha():
    torch.manual_seed(0)
    n = 400_000
    x0 = torch.zeros(n, 1, 1, 1, dtype=torch.long)
    probs = dist((5, 1.0)).expand(n, 1, 1, 1, K)  # noise is always 5, so x == x0 iff kept
    x = corrupt(x0, probs, alpha=0.3, top_p=0.9, include_x0=False)
    assert abs((x == 0).float().mean().item() - 0.3) < 0.005


def test_sample_from_probs_restricted_and_renormalised():
    torch.manual_seed(0)
    n = 200_000
    probs = dist((1, 0.5), (4, 0.25), (6, 0.15), (8, 0.1)).expand(n, 1, 1, 1, K)
    x = sample_from_probs(probs, top_p=0.75).flatten().numpy()  # nucleus {1, 4}
    counts = np.bincount(x, minlength=K)
    assert counts.sum() == counts[1] + counts[4]
    assert abs(counts[1] / n - 2 / 3) < 0.005


def test_schedule():
    a = keep_schedule(8)
    assert a[0] == 0.0 and all(b > c for b, c in zip(a[1:], a[:-1])) and a[-1] < 1.0
