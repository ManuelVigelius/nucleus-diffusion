"""Generate samples with all N trained stages.

    python sample.py --run_dir runs/cifar_N8_cosine_p0.9 --n 64 --temperature 1.0
    python sample.py --run_dir ... --refine_steps 2      # 2 extra predictor-corrector passes per stage
"""
import argparse
import os

import torch

import common
from diffusion import sample


def load_models(run_dir, device):
    cfg = common.load_config(run_dir)
    models = [common.load_frozen(cfg, run_dir, k, device) for k in range(1, cfg["num_steps"] + 1)]
    return cfg, models


@torch.no_grad()
def generate(cfg, models, n, device, top_p, temperature, refine_steps=0, final="sample",
             batch_size=256, amp=None, trajectory=False):
    """Returns (n, C, H, W) 4-bit samples (on CPU), and optionally the per-stage trajectory
    of the first batch as a list of (x_input, x0_hat) pairs."""
    ds = common.DATASETS[cfg["dataset"]]
    alphas = common.alphas_from_cfg(cfg)
    amp = amp or common.get_amp(device)[0]
    out, traj = [], None
    for i in range(0, n, batch_size):
        b = min(batch_size, n - i)
        x, t = sample(models, alphas, (b, ds["channels"], ds["size"], ds["size"]), device, top_p=top_p,
                      temperature=temperature, refine_steps=refine_steps, final=final, amp=amp,
                      return_trajectory=True)
        out.append(x.cpu())
        traj = traj if traj is not None else t
    out = torch.cat(out)
    return (out, traj) if trajectory else out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--top_p", type=float, default=None, help="default: the run's training top_p")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--refine_steps", type=int, default=0,
                   help="extra corrector passes per stage k>=2 (re-noise x0_hat at level alpha_k, re-apply M_k)")
    p.add_argument("--final", default="sample", choices=["sample", "argmax"],
                   help="last stage: sample from the nucleus or take the argmax")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None, help="PNG path (default: <run_dir>/samples_<settings>.png)")
    p.add_argument("--trajectory", action="store_true",
                   help="also save a grid of each stage's input and x0_hat for the first 8 samples")
    args = p.parse_args(argv)

    device = common.get_device(args.device)
    torch.manual_seed(args.seed)
    cfg, models = load_models(args.run_dir, device)
    top_p = cfg["top_p"] if args.top_p is None else args.top_p
    x, traj = generate(cfg, models, args.n, device, top_p, args.temperature, args.refine_steps,
                       args.final, args.batch_size, trajectory=True)

    tag = f"p{top_p:g}_T{args.temperature:g}_r{args.refine_steps}_{args.final}"
    out = args.out or os.path.join(args.run_dir, f"samples_{tag}.png")
    common.save_grid_square(x, out)
    print(f"saved {len(x)} samples to {out}")
    if args.trajectory:
        rows = [r[:8] for pair in traj for r in pair]
        tpath = os.path.splitext(out)[0] + "_trajectory.png"
        common.save_grid(rows, tpath)
        print(f"saved trajectory (rows: input_k, x0_hat_k for k=1..{len(traj)}) to {tpath}")


if __name__ == "__main__":
    main()
