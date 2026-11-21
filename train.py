"""Stage-wise training of nucleus-constrained discrete diffusion.

    python train.py --stages 1-8 --top_p 0.9 --out_dir /content/drive/MyDrive/ddiff
    python train.py --stage 3 ...        # needs finished checkpoints for stages 1, 2
    python train.py --smoke              # tiny end-to-end check (2 stages x 50 iters + sampling)
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

import common
from diffusion import K, frozen_forwards_per_iter, make_inputs
from model import Marginal

LN2 = math.log(2.0)


def str2bool(s):
    if s.lower() in ("1", "true", "yes", "y"):
        return True
    if s.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {s!r}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    # data / run
    a("--dataset", default="cifar", choices=list(common.DATASETS))
    a("--data_dir", default="./data")
    a("--out_dir", default="./runs", help="parent dir; each run gets a subfolder named after its hyperparameters")
    a("--tag", default="", help="optional suffix for the run folder name")
    # method
    a("--num_steps", type=int, default=8)
    a("--schedule", default="cosine", choices=["cosine", "linear"])
    a("--top_p", type=float, default=0.9, help="1.0 = plain uniform discrete diffusion")
    a("--include_x0", type=str2bool, default=True, help="add true x0 to the nucleus during training")
    a("--stage1", default="marginal", choices=["marginal", "vit"],
      help="stage 1 sees pure noise, so its optimal model is the per-subpixel data marginal (counted, "
           "no training). 'vit' trains a network instead (for comparison)")
    # model
    a("--patch_size", type=int, default=2)
    a("--d_in", type=int, default=32)
    a("--dim", type=int, default=256)
    a("--depth", type=int, default=6)
    a("--heads", type=int, default=4)
    a("--mlp_ratio", type=float, default=4.0)
    # which stages
    a("--stage", type=int, default=None, help="train a single stage")
    a("--stages", default=None, help="range like 1-8 (default: all)")
    # optimisation
    a("--iters_per_stage", type=int, default=40000)
    a("--batch_size", type=int, default=128)
    a("--lr", type=float, default=3e-4)
    a("--warmup", type=int, default=1000)
    a("--weight_decay", type=float, default=0.05)
    a("--grad_clip", type=float, default=1.0)
    a("--ema_decay", type=float, default=0.999)
    a("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"])
    a("--compile", action="store_true", help="torch.compile the trained and frozen models")
    # logging
    a("--log_every", type=int, default=100)
    a("--eval_every", type=int, default=2000)
    a("--eval_batches", type=int, default=10)
    a("--save_every", type=int, default=2000)
    a("--device", default="auto")
    a("--seed", type=int, default=0)
    a("--smoke", action="store_true", help="tiny model, N=3 (marginal + 2 trained stages x 50 iters), then sample")
    args = p.parse_args(argv)

    if args.smoke:
        # stage 1 is the counted marginal; stages 2-3 are trained (stage 3 uses a frozen ViT)
        args.num_steps, args.iters_per_stage, args.batch_size = 3, 50, 32
        args.dim, args.depth, args.heads, args.d_in = 64, 2, 2, 16
        args.log_every, args.eval_every, args.save_every, args.eval_batches = 10, 25, 25, 2
        args.warmup, args.stage, args.stages = 10, None, None
        args.tag = (args.tag + "_smoke").strip("_")
    return args


def stage_list(args):
    if args.stage is not None and args.stages is not None:
        raise SystemExit("use either --stage or --stages")
    if args.stage is not None:
        ks = [args.stage]
    elif args.stages is not None:
        lo, _, hi = args.stages.partition("-")
        ks = list(range(int(lo), int(hi or lo) + 1))
    else:
        ks = list(range(1, args.num_steps + 1))
    for k in ks:
        if not 1 <= k <= args.num_steps:
            raise SystemExit(f"stage {k} outside 1..{args.num_steps}")
    return ks


# ----------------------------------------------------------------------------- metrics
def logit_metrics(logits, x0, n_quantile=200_000):
    """CE (nats/bits per subpixel), x0 accuracy, predictive entropy (bits) mean/p10/p90."""
    logits = logits.float()
    logp = logits.log_softmax(-1)
    ce = F.nll_loss(logp.reshape(-1, K), x0.reshape(-1)).item()
    acc = (logits.argmax(-1) == x0).float().mean().item()
    ent = -(logp.exp() * logp).sum(-1).flatten() / LN2
    sub = ent[torch.randint(0, ent.numel(), (min(n_quantile, ent.numel()),), device=ent.device)]
    return {"ce_nats": ce, "ce_bits": ce / LN2, "acc": acc, "entropy_bits": ent.mean().item()}, sub.cpu()


def finish_entropy(m, ent_samples):
    q = torch.quantile(ent_samples, torch.tensor([0.1, 0.9]))
    m["entropy_p10"], m["entropy_p90"] = q[0].item(), q[1].item()
    return m


def fmt(m):
    keys = ["ce_nats", "ce_bits", "acc", "entropy_bits", "entropy_p10", "entropy_p90",
            "nucleus_size", "x0_outside_nucleus", "input_eq_x0", "img_per_s", "lr"]
    return " ".join(f"{k}={m[k]:.4g}" for k in keys if k in m)


def log(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


@torch.no_grad()
def evaluate(model, frozen, test_batches, cfg, alphas, amp, n_batches):
    sums, ents = {}, []
    for _ in range(n_batches):
        x0 = next(test_batches)
        x, cst = make_inputs(x0, frozen, alphas, cfg["top_p"], cfg["include_x0"], amp, return_stats=True)
        with amp():
            logits = model(x)
        m, e = logit_metrics(logits, x0, n_quantile=50_000)
        m.update(cst or {})
        ents.append(e)
        for key, v in m.items():
            sums[key] = sums.get(key, 0.0) + v
    m = {key: v / n_batches for key, v in sums.items()}
    return finish_entropy(m, torch.cat(ents))


@torch.no_grad()
def save_stage_grid(model, frozen, cfg, alphas, amp, test_data, stage_dir):
    """Rows: test images / M_k's noisy inputs / M_k's argmax predictions."""
    x0 = test_data[:16].long()
    x, _ = make_inputs(x0, frozen, alphas, cfg["top_p"], cfg["include_x0"], amp)
    with amp():
        pred = model(x).argmax(-1)
    path = os.path.join(stage_dir, "grid.png")
    common.save_grid([x0, x, pred], path)
    return path


def fit_marginal_stage(cfg, args, path, stage_dir, metrics_path, alphas, amp, train_data, test_data):
    """Stage 1 without training: its input is pure noise, so M_1 = per-subpixel data marginal."""
    flip = cfg["dataset"] == "cifar"
    model = common.build_model(cfg, 1).fit(train_data, flip=flip).to(train_data.device)
    print("[stage 1] M_1 = per-subpixel marginal of the train set (counted, no training)")
    for split, data in (("train", train_data), ("test", test_data)):
        n_batches = min(50, len(data) // args.batch_size)
        batches = common.Batches(data, args.batch_size, flip=flip and split == "train")
        m = evaluate(model, [], batches, cfg, alphas, amp, n_batches)
        print(f"[stage 1] {split.upper():5s} {fmt(m)}", flush=True)
        log(metrics_path, {"split": split, "stage": 1, "iter": 0, **m})
    common.atomic_save({"kind": "marginal", "ema": model.state_dict(), "iter": 0, "total_iters": 0,
                        "done": True, "config": cfg}, path)
    grid_path = save_stage_grid(model, [], cfg, alphas, amp, test_data, stage_dir)
    print(f"[stage 1] done. checkpoint {path}, grid {grid_path}")


# ----------------------------------------------------------------------------- training
def lr_at(it, args):
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    t = (it - args.warmup) / max(1, args.iters_per_stage - args.warmup)
    return args.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def make_optimizer(model, args):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.ndim >= 2 and name not in ("pos_embed", "value_embed") else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.99))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def train_stage(k, cfg, args, run_dir, device, train_data, test_data):
    path = common.ckpt_path(run_dir, k)
    stage_dir = os.path.join(run_dir, f"stage_{k}")
    os.makedirs(stage_dir, exist_ok=True)
    metrics_path = os.path.join(stage_dir, "metrics.jsonl")

    ck = torch.load(path, map_location="cpu", weights_only=False) if os.path.exists(path) else None
    if ck is not None and ck.get("done"):
        print(f"[stage {k}] already finished ({path}), skipping")
        return

    alphas = common.alphas_from_cfg(cfg)
    amp, use_scaler, amp_desc = common.get_amp(device, args.amp)
    if common.uses_marginal(cfg, k):
        return fit_marginal_stage(cfg, args, path, stage_dir, metrics_path, alphas, amp, train_data, test_data)
    # Loading also checks that stages 1..k-1 are finished. With top_p = 1 the inputs don't depend
    # on the frozen models, so make_inputs never runs them.
    frozen_used = [common.load_frozen(cfg, run_dir, j, device) for j in range(1, k)]
    nf = frozen_forwards_per_iter(k, cfg["top_p"], cfg["stage1"] == "marginal")

    model = common.build_model(cfg, k).to(device)
    ema = common.EMA(model, args.ema_decay)
    opt = make_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    start = 0
    if ck is not None:
        model.load_state_dict(ck["model"])
        ema.model.load_state_dict(ck["ema"])
        ema.step = ck["ema_step"]
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start = ck["iter"]
        print(f"[stage {k}] resuming from iteration {start}")

    fwd_model = torch.compile(model) if args.compile else model
    eval_model = torch.compile(ema.model) if args.compile else ema.model
    if args.compile and nf > 0:
        frozen_used = [m if isinstance(m, Marginal) else torch.compile(m) for m in frozen_used]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[stage {k}] alpha_k={alphas[k-1]:.4f} | params {n_params/1e6:.2f}M | precision {amp_desc} | "
          f"cost/iter: {nf} frozen ViT forward passes + 1 fwd/bwd (~{(nf + 3) / 3:.2f}x a plain training step)")

    torch.manual_seed(args.seed * 1000 + k * 7919 + start)
    train_batches = common.Batches(train_data, args.batch_size, flip=cfg["dataset"] == "cifar")
    test_batches = common.Batches(test_data, args.batch_size, flip=False)

    def save(it, done):
        common.atomic_save({"model": model.state_dict(), "ema": ema.model.state_dict(), "ema_step": ema.step,
                            "opt": opt.state_dict(), "scaler": scaler.state_dict(), "iter": it,
                            "total_iters": args.iters_per_stage, "done": done, "config": cfg}, path)

    model.train()
    sync(device)
    t0, imgs = time.time(), 0
    for it in range(start, args.iters_per_stage):
        log_now = (it + 1) % args.log_every == 0 or it + 1 == args.iters_per_stage
        lr = lr_at(it, args)
        for g in opt.param_groups:
            g["lr"] = lr

        x0 = next(train_batches)
        x, cst = make_inputs(x0, frozen_used, alphas, cfg["top_p"], cfg["include_x0"], amp, return_stats=log_now)
        with amp():
            logits = fwd_model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, K), x0.reshape(-1))

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        ema.update(model)
        imgs += len(x0)

        if log_now:
            m, ent = logit_metrics(logits.detach(), x0)
            m = finish_entropy(m, ent)
            m.update(cst or {})
            sync(device)
            m["img_per_s"] = imgs / (time.time() - t0)
            m["lr"] = lr
            print(f"[stage {k}] it {it+1:6d} train {fmt(m)}", flush=True)
            log(metrics_path, {"split": "train", "stage": k, "iter": it + 1, **m})
            t0, imgs = time.time(), 0

        if (it + 1) % args.eval_every == 0 or it + 1 == args.iters_per_stage:
            m = evaluate(eval_model, frozen_used, test_batches, cfg, alphas, amp, args.eval_batches)
            print(f"[stage {k}] it {it+1:6d} TEST  {fmt(m)}", flush=True)
            log(metrics_path, {"split": "test", "stage": k, "iter": it + 1, **m})
            t0, imgs = time.time(), 0

        if (it + 1) % args.save_every == 0 and it + 1 < args.iters_per_stage:
            save(it + 1, done=False)

    save(args.iters_per_stage, done=True)

    grid_path = save_stage_grid(ema.model, frozen_used, cfg, alphas, amp, test_data, stage_dir)
    print(f"[stage {k}] done. checkpoint {path}, grid {grid_path}")


def main(argv=None):
    args = parse_args(argv)
    device = common.get_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    cfg = {key: getattr(args, key) for key in common.RUN_KEYS + ["tag"]}
    run_dir = os.path.join(args.out_dir, common.run_name(cfg))
    ks = stage_list(args)

    alphas = common.alphas_from_cfg(cfg)
    print(f"run dir : {run_dir}")
    print(f"device  : {device} | precision {common.get_amp(device, args.amp)[2]}")
    print("keep schedule alpha_k: " + ", ".join(f"{i+1}:{a:.4f}" for i, a in enumerate(alphas)))
    print(f"top_p={cfg['top_p']} include_x0={cfg['include_x0']} stage1={cfg['stage1']} stages={ks}")
    if cfg["top_p"] >= 1.0:
        print("top_p = 1.0: plain uniform discrete diffusion, no frozen forward passes needed")

    common.save_config({**cfg, "train_args": vars(args)}, run_dir)
    train_data = common.load_split(cfg["dataset"], True, args.data_dir, device)
    test_data = common.load_split(cfg["dataset"], False, args.data_dir, device)
    for k in ks:
        train_stage(k, cfg, args, run_dir, device, train_data, test_data)

    if args.smoke:
        import sample
        sample.main(["--run_dir", run_dir, "--n", "16", "--device", str(device), "--trajectory"])


if __name__ == "__main__":
    main()
