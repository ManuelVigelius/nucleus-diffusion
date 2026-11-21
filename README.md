# Nucleus-constrained stage-wise discrete diffusion

CIFAR-10 (or Fashion-MNIST) quantised to 4 bits per channel. Every subpixel is a 16-way categorical.
There are N stages, each with its **own** model M_k that predicts x0. Stages are trained one after another.
M_1 sees pure uniform noise, which carries no information about x0. Its optimal prediction is therefore
the per-subpixel data marginal p(x0[c,h,w]), so M_1 is **counted from the training set** (flip-symmetrised
for CIFAR). Only M_2..M_N are ViTs (`--stage1 vit` trains a ViT for stage 1 instead, for comparison).
There is no class conditioning and no timestep input. Each M_k sees only its noisy image.
M_k's input is built fresh for every batch: start from uniform noise, run the frozen M_1..M_{k-1}, and after
each one corrupt the true x0. Each subpixel keeps x0 with probability alpha_{j+1}. Otherwise it is
replaced by a value drawn uniformly from the top-p nucleus of M_j's prediction (plus x0 itself when
`--include_x0 true`). With `--top_p 1.0` this is exactly uniform discrete diffusion.

| file | content |
|---|---|
| `diffusion.py` | schedule, nucleus, `corrupt`, training-input chain, sampler |
| `model.py` | the ViT |
| `common.py` | data, precision, checkpoints, EMA, image grids |
| `train.py` | stage-wise training (`--stage k` / `--stages a-b` / `--smoke`) |
| `sample.py` | sample PNG grid (+ per-stage trajectory) |
| `eval_fid.py` | FID against the quantised train set (clean-fid, optional) |
| `tests/` | pytest unit tests (CPU, ~2 s) |

Data comes from the Hugging Face parquet mirrors (`uoft-cs/cifar10`, `zalando-datasets/fashion_mnist`;
set `$HF_ENDPOINT` to use another mirror). It is cached as `<data_dir>/<name>_<split>_uint8.npy`.
If the HF download fails, or pyarrow is missing, it falls back to torchvision.

Each run writes into `<out_dir>/<dataset>_N<steps>_<schedule>_p<top_p>[_nox0][_<tag>]/`:
`config.json`, `stage_k.pt` (model, EMA, optimiser; resumed automatically),
`stage_k/metrics.jsonl`, `stage_k/grid.png` (test images / M_k inputs / M_k argmax), and samples.

## Colab (console, not notebook cells)

Open a terminal (or prefix each line with `!` in a single cell):

```bash
# 1. Mount Drive (this one has to run in a notebook cell once):
#    from google.colab import drive; drive.mount('/content/drive')

# 2. Code + deps
cd /content && git clone https://github.com/ManuelVigelius/nucleus-diffusion.git && cd nucleus-diffusion
pip install -q tqdm clean-fid        # torch / torchvision / numpy / pillow / pyarrow are preinstalled

# 3. Unit tests + smoke test (< 2 min; N=3: counted M_1 + 2 tiny ViT stages x 50 iters, then sampling)
python -m pytest -q tests
python train.py --smoke --out_dir /content/runs

# 4. Baseline: plain uniform discrete diffusion (no frozen forwards, so every stage costs the same)
OUT=/content/drive/MyDrive/ddiff
python train.py --stages 1-8 --top_p 1.0  --out_dir $OUT --data_dir /content/data

# 5. Method
python train.py --stages 1-8 --top_p 0.99 --out_dir $OUT --data_dir /content/data
python train.py --stages 1-8 --top_p 0.9  --out_dir $OUT --data_dir /content/data

# Optional ablation: don't add the true x0 to the training nucleus
python train.py --stages 1-8 --top_p 0.9 --include_x0 false --out_dir $OUT --data_dir /content/data
```

After a disconnect, re-run the **same command**. Finished stages are skipped, and the current
stage resumes from its last checkpoint (`--save_every`, default 2000 iterations). You can also
train one stage per session, e.g. `--stage 3`. This needs stages 1 and 2 to be finished.

```bash
# 6. Sampling (all N stages)
RUN=$OUT/cifar_N8_cosine_p0.9
python sample.py --run_dir $RUN --n 64 --temperature 1.0 --trajectory
python sample.py --run_dir $RUN --n 64 --top_p 0.9 --temperature 0.9 --final argmax
python sample.py --run_dir $RUN --n 64 --refine_steps 2      # predictor-corrector, see below

# 7. FID vs 4-bit-quantised CIFAR-10 train (+ quantised test-vs-train floor)
python eval_fid.py --run_dir $RUN --n 10000 --data_dir /content/data
```

Results are appended to `$RUN/fid.txt`. Reference PNGs and clean-fid stats are built once and cached.

## Notes

- **Cost.** Training M_k needs k-2 frozen ViT forward passes per iteration (M_1 is a lookup table).
  These run in no-grad mode at about 1/3 the cost of a forward+backward pass, so stage 8 costs about
  (6+3)/3 = 3x stage 2.
  The script prints this estimate at the start of each stage. With `--top_p 1.0` the frozen
  forwards are skipped entirely, because they cannot affect the input.
- **Precision.** bf16 autocast on Ampere+ GPUs and fp16 + GradScaler on a T4. fp32 on CPU/MPS.
  `--compile` enables torch.compile.
- **Frozen models, eval and sampling** all use the EMA weights (decay 0.999, with warmup).
- **Metrics** (stdout + `metrics.jsonl`): CE in nats and bits, x0 accuracy, predictive entropy in bits
  (mean / p10 / p90), mean nucleus size, fraction of x0 outside the nucleus, fraction of input equal
  to x0, and img/s. Test metrics are logged every `--eval_every` iterations.
- **Refinement (`--refine_steps R`).** M_k was trained on `corrupt(x0, nucleus(M_{k-1}) ∪ {x0}, alpha_k)`.
  After M_k predicts x0_hat, the sampler re-noises x0_hat with exactly that distribution and applies M_k
  again, R extra times. It uses M_{k-1}'s nucleus ∪ {x0_hat} and keep probability alpha_k.
  This is a predictor-corrector loop that stays on M_k's training distribution. It only applies
  for k ≥ 2: stage 1 sees pure noise, so refining there would just redraw an independent sample.
- Quick debugging: `--dataset fmnist` (28x28 grayscale, 196 tokens with patch 2).
