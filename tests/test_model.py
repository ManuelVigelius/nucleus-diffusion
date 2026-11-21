import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ViT  # noqa: E402


@pytest.mark.parametrize("C,S", [(3, 32), (1, 28)])
def test_shapes(C, S):
    m = ViT(C, S, dim=64, depth=2, heads=4)
    x = torch.randint(0, 16, (2, C, S, S))
    assert m(x).shape == (2, C, S, S, 16)
    assert m.grid ** 2 == (S // 2) ** 2


def test_default_size():
    n = sum(p.numel() for p in ViT(3, 32).parameters())
    assert 4e6 < n < 10e6, n


@pytest.mark.parametrize("C,S", [(3, 32), (1, 28)])
def test_patchify_roundtrip_is_local(C, S):
    """With depth 0 and no positional mixing, changing one subpixel may only change the logits
    of the patch it belongs to -> patchify/unpatchify are consistent inverses."""
    torch.manual_seed(0)
    m = ViT(C, S, dim=32, depth=0, heads=2).eval()
    x = torch.randint(0, 16, (1, C, S, S))
    y = x.clone()
    c, i, j = C - 1, 5, 9
    y[0, c, i, j] = (x[0, c, i, j] + 3) % 16
    diff = (m(x) - m(y)).abs().sum(-1)[0]  # (C, S, S)
    changed = (diff > 1e-6).nonzero()
    pi, pj = i // 2, j // 2
    assert len(changed) > 0
    assert all(r[1] // 2 == pi and r[2] // 2 == pj for r in changed.tolist())


def test_marginal_counts_and_flip():
    from model import Marginal
    # 4 images of 1x1x2 pixels: left pixel values [3,3,3,5], right pixel always 7
    data = torch.tensor([[3, 7], [3, 7], [3, 7], [5, 7]]).view(4, 1, 1, 2)
    m = Marginal(1, 1)
    m.logits = torch.zeros(1, 1, 2, 16)
    p = m.fit(data, pseudocount=1e-9).logits.exp()
    assert torch.allclose(p[0, 0, 0, [3, 5]], torch.tensor([0.75, 0.25]), atol=1e-6)
    assert torch.allclose(p[0, 0, 1, 7], torch.tensor(1.0), atol=1e-6)
    pf = m.fit(data, flip=True, pseudocount=1e-9).logits.exp()
    assert torch.allclose(pf[0, 0, 0], pf[0, 0, 1], atol=1e-6)          # symmetric under h-flip
    assert torch.allclose(pf[0, 0, 0, [3, 5, 7]], torch.tensor([0.375, 0.125, 0.5]), atol=1e-6)
    out = m(torch.randint(0, 16, (5, 1, 1, 2)))                            # ignores its input
    assert out.shape == (5, 1, 1, 2, 16)
