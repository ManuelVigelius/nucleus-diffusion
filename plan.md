I want to run a research experiment: a stage-wise discrete diffusion model on CIFAR-10 with
"nucleus-constrained" noise. Build a clean, small PyTorch codebase that I can run from the
console in Google Colab (`python train.py ...`, NOT a notebook). Use only torch, torchvision,
numpy, pillow, and optionally tqdm and clean-fid.

## Data
- CIFAR-10 via torchvision. Quantise every channel to 4 bits: v = x_uint8 // 16 (values 0..15).
  Decode for viewing with v*16 + 8.
- Add a `--dataset fmnist` option (Fashion-MNIST, 28x28, 1 channel, also 16 levels) for fast debugging.
- Images are tensors of shape (B, C, H, W) of ints in [0, 16). Each subpixel is one categorical variable.
- Use random horizontal flips on CIFAR.

## Method (read carefully, this is the novel part)
There are N = 8 diffusion steps (configurable `--num_steps`). Each step k = 1..N has its OWN
independent model M_k. Every model predicts the clean image x0: logits of shape (B, C, H, W, 16).
Models are trained sequentially: M_k is trained only after M_1..M_{k-1} are finished and frozen.

Keep schedule: alpha_k = probability that a subpixel keeps its true value in step k's input.
alpha_1 = 0 (step 1's input is pure uniform noise). For k >= 2, use a cosine-shaped increasing
schedule, e.g. alpha_k = 1 - cos(pi/2 * (k-1)/N). Make it configurable and print it at startup.

Corruption operator, applied independently per subpixel:
  corrupt(x0, probs, alpha, top_p, include_x0):
    allowed = top-p nucleus of probs (smallest set of values whose cumulative probability
              >= top_p; always contains at least the argmax)
    if include_x0: allowed |= onehot(x0)      # training only; at inference x0_hat is already inside
    noise  = a value drawn UNIFORMLY from `allowed`
    keep   = Bernoulli(alpha)
    return where(keep, x0, noise)
  With top_p = 1.0 this must reduce exactly to standard uniform discrete diffusion.
  Implement it vectorised: no Python loops over pixels.

Building training inputs for M_k (NO caching; regenerate fresh for every batch, under torch.no_grad()):
    x = uniform random ints in [0,16)                # input to M_1
    for j in 1..k-1:
        probs_j = softmax(M_j(x))                    # frozen model, eval mode, no grad
        x = corrupt(x0, probs_j, alpha_{j+1}, top_p, include_x0=True)
    train M_k on (x -> x0) with mean cross-entropy over subpixels.

Sampling (inference), mirroring training exactly:
    x = uniform noise
    for k in 1..N:
        probs = softmax(M_k(x) / temperature)
        x0_hat = sample from probs restricted to its top-p nucleus (renormalised)
        if k < N: x = corrupt(x0_hat, probs, alpha_{k+1}, top_p, include_x0=False)
    return x0_hat   (also offer argmax of the final step as an option)

## Model: a small ViT, one instance per step
- Input embedding: each subpixel value gets a learned embedding (separate table per channel,
  16 entries, dim d_in, e.g. 32). Concatenate the channel embeddings per pixel, group pixels into
  non-overlapping patches (patch size 2 for CIFAR -> 256 tokens; also works for 28x28 fmnist),
  then apply a linear projection to model dim.
- Learned positional embedding. Pre-LN transformer blocks. Defaults: dim 256, depth 6, heads 4,
  MLP ratio 4, no dropout. Roughly 5-10M params. Everything configurable.
- Output head: a linear layer from each token to patch_size^2 * C * 16 logits, reshaped back to (B, C, H, W, 16).
- No timestep conditioning (each step has its own model).

## Training
- `python train.py --stage k` trains one stage, loading frozen checkpoints for stages < k.
  `python train.py --stages 1-8` trains them in sequence. Fail clearly if an earlier checkpoint is missing.
- AdamW, lr 3e-4 with warmup and cosine decay, weight decay 0.05, grad clipping at 1.0.
  Keep an EMA of the weights (decay 0.999) and use the EMA weights for frozen models, eval and sampling.
- Mixed precision: bf16 if supported, otherwise fp16 with GradScaler (Colab T4 has no bf16).
  Use torch.compile only behind a flag.
- Default `--iters_per_stage` 40000, batch size 128. All configurable.
- Checkpoints go to `--out_dir` (I will point it at Google Drive, e.g. /content/drive/MyDrive/ddiff).
  Save every few thousand iterations, and RESUME automatically if a checkpoint for the current stage
  exists (Colab disconnects). Save the config as JSON next to the checkpoints.
- Before running, print an estimate of the cost per iteration (number of frozen forward passes).

## Metrics (log to stdout and append to a metrics.jsonl file per stage)
Every `--log_every` iterations:
- train CE (nats and bits per subpixel), x0 accuracy
- MEAN ENTROPY in bits of M_k's predicted distributions, averaged over subpixels and batch. This is
  the key metric. Also log the 10th and 90th percentiles.
- for the corruption that produced M_k's input (k >= 2): mean nucleus size, fraction of subpixels whose
  true x0 was OUTSIDE the nucleus, and fraction of input subpixels equal to x0
- throughput (img/s)
Every `--eval_every` iterations, on the test split: the same metrics without gradients.
At the end of each stage, save a PNG grid comparing test images, M_k's noisy inputs and M_k's argmax predictions.

## Sampling and evaluation
- `python sample.py --n 64 --top_p ... --temperature ...` writes a PNG grid using all N stages.
- `python eval_fid.py --n 10000`: FID of generated samples against the 4-bit-quantised CIFAR-10 train set
  using clean-fid. Also report the FID of quantised real test images against quantised train as a floor.
  Make this optional and skip gracefully if clean-fid is not installed.

## Experiments I will run (make them one-flag changes)
- Baseline:  --top_p 1.0  (plain uniform discrete diffusion, same pipeline)
- Method:    --top_p 0.99 and --top_p 0.9
- Optional ablation flag --include_x0 false
Each run uses its own out_dir subfolder named after its key hyperparameters.

## Quality requirements
- Unit tests (pytest, CPU, fast) for the corruption operator: the nucleus is correct on hand-made
  distributions; noise is empirically uniform within the nucleus (chi-square-style check); top_p=1.0
  gives uniform over all 16 values; alpha=1 returns x0; alpha=0 never keeps x0 unless it is drawn as noise.
- Shape tests for the ViT on CIFAR and fmnist sizes.
- A `--smoke` flag that trains 2 stages for 50 iterations each on CPU/GPU with a tiny model, then samples,
  so I can confirm the whole pipeline works in under 2 minutes.
- A README with the exact Colab console commands: mounting Drive, installing deps, smoke test,
  full baseline run, method run, sampling, FID.
Keep the code readable, a handful of files, with no framework abstractions beyond what's needed.
Run the unit tests and the smoke test yourself before you finish.