"""FID of generated samples vs. the 4-bit-quantised training set (clean-fid).

    python eval_fid.py --run_dir runs/cifar_N8_cosine_p0.9 --n 10000

Also reports the FID of quantised real test images vs. quantised train as a floor.
Reference images/statistics are cached under --data_dir.
"""
import argparse
import os
import shutil

import torch
from PIL import Image

import common
from sample import generate, load_models


def write_pngs(x, folder):
    """x: (n, C, H, W) 4-bit tensor -> RGB PNGs in `folder` (fresh)."""
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    imgs = common.decode(x).cpu()
    if imgs.shape[1] == 1:
        imgs = imgs.expand(-1, 3, -1, -1)
    for i, im in enumerate(imgs.permute(0, 2, 3, 1).numpy()):
        Image.fromarray(im).save(os.path.join(folder, f"{i:06d}.png"))


def ensure_reference(dataset, data_dir, split):
    """Folder of quantised real images for `split` (written once)."""
    folder = os.path.join(data_dir, f"fid_{dataset}_q4_{split}")
    done = os.path.join(folder, ".done")
    if not os.path.exists(done):
        print(f"writing quantised {split} images to {folder}")
        write_pngs(common.load_split(dataset, split == "train", data_dir, "cpu"), folder)
        open(done, "w").close()
    return folder


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--top_p", type=float, default=None, help="default: the run's training top_p")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--refine_steps", type=int, default=0)
    p.add_argument("--final", default="sample", choices=["sample", "argmax"])
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--skip_floor", action="store_true", help="don't compute the real-test-vs-train floor")
    args = p.parse_args(argv)

    try:
        from cleanfid import fid
    except ImportError:
        print("clean-fid is not installed (pip install clean-fid); skipping FID evaluation.")
        return

    device = common.get_device(args.device)
    torch.manual_seed(args.seed)
    cfg, models = load_models(args.run_dir, device)
    top_p = cfg["top_p"] if args.top_p is None else args.top_p

    ref_dir = ensure_reference(cfg["dataset"], args.data_dir, "train")
    ref_name = f"{cfg['dataset']}_q4_train"
    fid_device = device if device.type == "cuda" else torch.device("cpu")
    if not fid.test_stats_exists(ref_name, mode="clean"):
        fid.make_custom_stats(ref_name, ref_dir, mode="clean", device=fid_device)

    results = {}
    if not args.skip_floor:
        test_dir = ensure_reference(cfg["dataset"], args.data_dir, "test")
        results["real_test_vs_train (floor)"] = fid.compute_fid(
            test_dir, dataset_name=ref_name, mode="clean", dataset_split="custom", device=fid_device)

    print(f"generating {args.n} samples (top_p={top_p}, T={args.temperature}, refine={args.refine_steps})")
    x = generate(cfg, models, args.n, device, top_p, args.temperature, args.refine_steps,
                 args.final, args.batch_size)
    tag = f"p{top_p:g}_T{args.temperature:g}_r{args.refine_steps}_{args.final}_n{args.n}"
    gen_dir = os.path.join(args.run_dir, f"fid_samples_{tag}")
    write_pngs(x, gen_dir)
    results[f"samples ({tag})"] = fid.compute_fid(
        gen_dir, dataset_name=ref_name, mode="clean", dataset_split="custom", device=fid_device)

    with open(os.path.join(args.run_dir, "fid.txt"), "a") as f:
        for name, v in results.items():
            line = f"FID {name}: {v:.3f}"
            print(line)
            f.write(line + "\n")


if __name__ == "__main__":
    main()
