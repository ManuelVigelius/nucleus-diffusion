"""Shared helpers: data, device/precision, run directories, checkpoints, EMA, image grids."""
import copy
import json
import os
from contextlib import nullcontext
from functools import partial

import numpy as np
import torch
from PIL import Image

from diffusion import K, keep_schedule
from model import Marginal, ViT

DATASETS = {"cifar": dict(channels=3, size=32), "fmnist": dict(channels=1, size=28)}

# Config keys that define the method / architecture; a run directory must not mix them.
RUN_KEYS = ["dataset", "num_steps", "schedule", "top_p", "include_x0", "stage1",
            "patch_size", "d_in", "dim", "depth", "heads", "mlp_ratio"]


# ----------------------------------------------------------------------------- data
HF_FILES = {  # Hugging Face parquet mirrors (PNG bytes + label)
    "cifar": ("uoft-cs/cifar10", "plain_text/{split}-00000-of-00001.parquet", "img"),
    "fmnist": ("zalando-datasets/fashion_mnist", "fashion_mnist/{split}-00000-of-00001.parquet", "image"),
}


def _download_hf(name, split, data_dir):
    """Raw uint8 images (N, C, H, W) from the HF parquet mirror. Honours $HF_ENDPOINT."""
    import io
    import urllib.request
    import pyarrow.parquet as pq
    repo, fname, col = HF_FILES[name]
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/datasets/{repo}/resolve/main/{fname.format(split=split)}"
    path = os.path.join(data_dir, f"hf_{name}_{split}.parquet")
    if not os.path.exists(path):
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path + ".tmp")
        os.replace(path + ".tmp", path)
    imgs = [np.asarray(Image.open(io.BytesIO(r["bytes"])))
            for r in pq.read_table(path, columns=[col]).column(col).to_pylist()]
    x = np.stack(imgs)
    return x.transpose(0, 3, 1, 2) if x.ndim == 4 else x[:, None]


def _download_torchvision(name, train, data_dir):
    import torchvision
    if name == "cifar":
        return torchvision.datasets.CIFAR10(data_dir, train=train, download=True).data.transpose(0, 3, 1, 2)
    return torchvision.datasets.FashionMNIST(data_dir, train=train, download=True).data.numpy()[:, None]


def load_split(name, train, data_dir, device):
    """Whole split as a uint8 tensor (N, C, H, W) of 4-bit values (0..15), kept on `device`.
    Downloads from the Hugging Face mirror (torchvision as fallback) and caches a .npy in data_dir."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}")
    split = "train" if train else "test"
    os.makedirs(data_dir, exist_ok=True)
    cache = os.path.join(data_dir, f"{name}_{split}_uint8.npy")
    if os.path.exists(cache):
        x = np.load(cache)
    else:
        try:
            x = _download_hf(name, split, data_dir)
        except Exception as e:  # no pyarrow, no network to HF, ...
            print(f"HF download failed ({type(e).__name__}: {e}); falling back to torchvision")
            x = _download_torchvision(name, train, data_dir)
        x = np.ascontiguousarray(x, dtype=np.uint8)
        ds = DATASETS[name]
        assert x.shape[1:] == (ds["channels"], ds["size"], ds["size"]), x.shape
        np.save(cache, x)
    return (torch.from_numpy(x) // 16).to(torch.uint8).contiguous().to(device)


class Batches:
    """Infinite shuffled minibatches from an in-memory split, with optional random h-flips."""

    def __init__(self, data, batch_size, flip):
        self.data, self.bs, self.flip = data, batch_size, flip
        self.perm, self.pos = None, len(data)

    def __next__(self):
        if self.pos + self.bs > len(self.data):
            self.perm, self.pos = torch.randperm(len(self.data), device=self.data.device), 0
        idx = self.perm[self.pos:self.pos + self.bs]
        self.pos += self.bs
        x = self.data[idx].long()
        if self.flip:
            f = torch.rand(len(x), 1, 1, 1, device=x.device) < 0.5
            x = torch.where(f, x.flip(-1), x)
        return x


def decode(x):
    """4-bit values -> uint8 for viewing."""
    return (x.long() * 16 + 8).clamp(0, 255).to(torch.uint8)


def save_grid(rows, path, scale=2):
    """rows: list of (n, C, H, W) 4-bit tensors, one grid row each. Writes a PNG."""
    rows = [decode(r).cpu() for r in rows]
    n, C, H, W = rows[0].shape
    pad = 2
    canvas = torch.full((C, len(rows) * (H + pad) + pad, n * (W + pad) + pad), 255, dtype=torch.uint8)
    for i, r in enumerate(rows):
        for j in range(n):
            y, x = pad + i * (H + pad), pad + j * (W + pad)
            canvas[:, y:y + H, x:x + W] = r[j]
    img = canvas.permute(1, 2, 0).numpy()
    img = Image.fromarray(img[..., 0] if C == 1 else img)
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path)


def save_grid_square(x, path, scale=2):
    """x: (n, C, H, W). Lays the samples out in a near-square grid."""
    n = len(x)
    cols = int(np.ceil(np.sqrt(n)))
    rows = [x[i:i + cols] for i in range(0, n, cols)]
    if len(rows[-1]) < cols:  # pad last row with blank (white) images
        blank = torch.full((cols - len(rows[-1]),) + tuple(x.shape[1:]), 15, dtype=x.dtype, device=x.device)
        rows[-1] = torch.cat([rows[-1], blank])
    save_grid(rows, path, scale)


# ----------------------------------------------------------------------------- device / precision
def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_amp(device, mode="auto"):
    """Returns (autocast context factory, use_grad_scaler, description).
    CUDA: bf16 if supported else fp16 (+GradScaler). CPU/MPS: fp32."""
    if mode == "off" or device.type != "cuda":
        return nullcontext, False, "fp32"
    if mode == "bf16" or (mode == "auto" and torch.cuda.is_bf16_supported()):
        return partial(torch.autocast, "cuda", dtype=torch.bfloat16), False, "bf16"
    return partial(torch.autocast, "cuda", dtype=torch.float16), True, "fp16+GradScaler"


# ----------------------------------------------------------------------------- runs / checkpoints
def run_name(cfg):
    name = f"{cfg['dataset']}_N{cfg['num_steps']}_{cfg['schedule']}_p{cfg['top_p']:g}"
    if not cfg["include_x0"]:
        name += "_nox0"
    if cfg["stage1"] != "marginal":
        name += "_s1vit"
    if cfg.get("tag"):
        name += f"_{cfg['tag']}"
    return name


def uses_marginal(cfg, k):
    return k == 1 and cfg["stage1"] == "marginal"


def build_model(cfg, k=None):
    """M_k. Stage 1 is the data marginal (no trained network) unless --stage1 vit."""
    ds = DATASETS[cfg["dataset"]]
    if k is not None and uses_marginal(cfg, k):
        return Marginal(ds["channels"], ds["size"], levels=K)
    return ViT(ds["channels"], ds["size"], levels=K, patch_size=cfg["patch_size"], d_in=cfg["d_in"],
               dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"], mlp_ratio=cfg["mlp_ratio"])


def alphas_from_cfg(cfg):
    return keep_schedule(cfg["num_steps"], cfg["schedule"])


def ckpt_path(run_dir, k):
    return os.path.join(run_dir, f"stage_{k}.pt")


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_config(cfg, run_dir):
    path = os.path.join(run_dir, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        diff = {k: (old.get(k), cfg.get(k)) for k in RUN_KEYS if old.get(k) != cfg.get(k)}
        if diff:
            raise SystemExit(f"Config in {run_dir} conflicts with current args (old, new): {diff}. "
                             f"Use a different --out_dir or --tag.")
    os.makedirs(run_dir, exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(path + ".tmp", path)


def load_config(run_dir):
    with open(os.path.join(run_dir, "config.json")) as f:
        return json.load(f)


def load_frozen(cfg, run_dir, k, device):
    """EMA weights of a finished stage k, in eval mode with grads disabled."""
    path = ckpt_path(run_dir, k)
    if not os.path.exists(path):
        raise SystemExit(f"Missing checkpoint for stage {k}: {path}. Train stage {k} first.")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not ck.get("done"):
        raise SystemExit(f"Stage {k} is not finished ({ck['iter']}/{ck['total_iters']} iters): {path}. "
                         f"Resume it with --stage {k} first.")
    model = build_model(cfg, k)
    model.load_state_dict(ck["ema"])
    model.to(device).eval().requires_grad_(False)
    return model


class EMA:
    """Exponential moving average of model weights, with warmup so short runs still get sane weights."""

    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.step = 0

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1 - d)

