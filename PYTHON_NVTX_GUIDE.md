# Implementing NVTX instrumentation for the Python (`timm`) reference

Companion to `PROFILING_GUIDE.md`. That guide covers the two **C++** captures
(`reports/cpp_b16`, `reports/omp_b16`). This is the step-by-step to add a matching
**Python** capture (`reports/py_b16`) so all three implementations show up in the same
`nsys stats --report nvtx_pushpop_sum` shape and can be compared bar-for-bar.

Design goal, mirroring the C++ side exactly:
- C++ gates its ranges behind `-DUSE_NVTX` so the normal (non-profiling) build is
  untouched. Python gates them behind an env var, `USE_NVTX=1`, so `run_py.sh` /
  `elaborate.sh` keep running exactly as today (zero overhead, zero behavior change)
  when the var isn't set.
- Same range names as `include/vit_nvtx.h`: `patch_embed`, `conv2d`, `pos_embed`,
  `blk_attn`, `blk_mlp`, `linear`, `attention`, `layernorm`, `gelu`, `pool`.
- Only the code path actually exercised by profiling (`VisionTransformer.timed_forward`
  → `Block.timed_forward`) gets touched. `forward()` / `Block.forward()` (the untimed,
  non-profiling inference path) are left alone.

**No GPU is involved.** Despite the `torch.cuda.nvtx` name, NVTX ranges are just named
CPU-side markers that `nsys` records onto its timeline — the exact same way the C++
captures already use them (`--trace=nvtx,osrt`, no CUDA trace, no kernels). The Python
model runs entirely on `.cpu()` today (`timm_train_vit/vit.py` sets
`device = torch.device('cpu')`), so this whole capture is CPU-only, matching the note at
the top of `profile.slurm` ("CPU + OpenMP only — no GPU yet"). GPUs only enter the
picture later, when you profile your colleagues' CUDA kernels — at which point you'd add
`cuda` to `--trace` and actually need a GPU node. Not now.

Total: 1 new file, 4 edited files, 1 extended script. ~30 minutes.

---

## Step 0 — sanity check (no GPU needed)

Correction from an earlier draft of this guide: `torch.cuda.nvtx.range_push`/`range_pop`
do **not** need a live GPU or CUDA context — they're a thin binding straight into the
NVTX marker library, gated only on PyTorch being a **CUDA-enabled build** (i.e. the
`torch._C._nvtx` symbol exists). Verified directly on the login node, which has no GPU
at all (`torch.cuda.is_available()` is `False` there):

```bash
python3 -c "import torch; torch.cuda.nvtx.range_push('x'); torch.cuda.nvtx.range_pop(); print('ok')"
# -> ok, even with torch.cuda.is_available() == False
```

So this whole Python-NVTX capture, like the rest of the current profiling (see the
comment at the top of `profile.slurm`: "this profiling is CPU + OpenMP only (no GPU
yet)"), runs fine on a **CPU partition** (e.g. `dcgp_usr_prod`, no `--gres`). You don't
need to touch the `--gres=gpu:1` request in `profile.slurm` for this — that's there only
because `boost_usr_prod` requires it as a matter of partition policy, not because
anything here needs a GPU. That will change once your colleagues' CUDA kernels land and
you profile *those* — this capture, today, is CPU-only end to end.

If, for some other reason, `torch._C._nvtx` isn't available in your environment (e.g. a
CPU-only torch wheel with no CUDA support compiled in at all), fall back to the pure-NVTX
pip package (`uv pip install nvtx` into the venv, then `import nvtx as _nvtx_lib` and use
`_nvtx_lib.annotate(name)` as a context manager instead of `torch.cuda.nvtx` in Step 1
below — same range names, same `nsys stats` output). Everything past this point assumes
`torch.cuda.nvtx` worked, which it did above.

---

## Step 1 — new file: `timm_train_vit/timm/nvtx_utils.py`

Placed **inside** the vendored `timm` package (not next to `vit.py`) so every file under
`timm_train_vit/timm/` can do a plain `from timm.nvtx_utils import nvtx_range`,
regardless of caller's cwd — same style as the existing `from timm.layers import ...`
imports in `vision_transformer.py`.

```python
"""Toggleable NVTX ranges — the Python-side equivalent of the C++ -DUSE_NVTX build.

Enabled with USE_NVTX=1 in the environment. When unset, nvtx_range is a no-op
context manager, so the normal (non-profiling) run_py.sh path pays zero cost.
"""
import os
from contextlib import contextmanager, nullcontext

USE_NVTX = os.environ.get("USE_NVTX") == "1"

if USE_NVTX:
    import torch

    @contextmanager
    def nvtx_range(name):
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
else:
    def nvtx_range(name):
        return nullcontext()
```

---

## Step 2 — `timm_train_vit/timm/layers/patch_embed.py`

Adds `patch_embed` (outer) and `conv2d` (inner, around the actual `nn.Conv2d` call) —
matches the C++ `PatchEmbed::forward` / `Conv2d::forward` pair.

Add the import near the top, with the other relative imports:
```python
from .format import Format, nchw_to
from .helpers import to_2tuple
from .trace_utils import _assert
from timm.nvtx_utils import nvtx_range          # <-- add
```

Then wrap the body of `PatchEmbed.forward` (the whole method gets indented one level
under the outer range; only the `self.proj(x)` line gets the extra inner range):

```python
    def forward(self, x):
        with nvtx_range("patch_embed"):
            B, C, H, W = x.shape
            if self.img_size is not None:
                if self.strict_img_size:
                    _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
                    _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
                elif not self.dynamic_img_pad:
                    _assert(
                        H % self.patch_size[0] == 0,
                        f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                    )
                    _assert(
                        W % self.patch_size[1] == 0,
                        f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
                    )
            if self.dynamic_img_pad:
                pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
                pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
                x = F.pad(x, (0, pad_w, 0, pad_h))
            with nvtx_range("conv2d"):
                x = self.proj(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
            elif self.output_fmt != Format.NCHW:
                x = nchw_to(x, self.output_fmt)
            x = self.norm(x)
            return x
```

(Just re-indented the existing body one level and inserted the two `with` blocks — no
logic changed.)

---

## Step 3 — `timm_train_vit/timm/layers/mlp.py`

Adds `linear` around `fc1`/`fc2` and `gelu` around the activation — matches C++'s
`Mlp::forward` (fc1 → GELU → fc2, all wrapped in `linear`/`gelu` ranges).

Add the import:
```python
from .grn import GlobalResponseNorm
from .helpers import to_2tuple
from timm.nvtx_utils import nvtx_range          # <-- add
```

Replace `Mlp.forward`:
```python
    def forward(self, x):
        with nvtx_range("linear"):
            x = self.fc1(x)
        with nvtx_range("gelu"):
            x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        with nvtx_range("linear"):
            x = self.fc2(x)
        x = self.drop2(x)
        return x
```

---

## Step 4 — `timm_train_vit/timm/models/vision_transformer.py`

Four separate edits in this file.

### 4a. Import

Right after the existing `import time` (this file already has a project-specific
`import time` around line 52 for `timed_forward`):
```python
import time
from timm.nvtx_utils import nvtx_range          # <-- add
```

### 4b. `Attention.forward` — `linear` (qkv, proj) + `attention` (the QKᵀ/softmax/AV core)

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        with nvtx_range("linear"):
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        with nvtx_range("attention"):
            if self.fused_attn:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        with nvtx_range("linear"):
            x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

> Note for later comparison: this vendored `Attention` uses one fused `qkv` Linear
> (`dim → 3·dim`) instead of three separate q/k/v Linears like the C++ side, and may hit
> PyTorch's fused SDPA kernel instead of the manual QKᵀ→softmax→AV steps (`self.fused_attn`).
> The `linear`/`attention` **instance counts** in `nsys stats` won't match C++ 1:1 because
> of this — the **total time per range** is still the correct thing to compare.

### 4c. `Block.timed_forward` — `blk_attn` / `blk_mlp` + `layernorm`

Only `timed_forward` is touched (that's the one actually called from
`VisionTransformer.timed_forward`); `Block.forward` is left as-is since it's not on the
profiling path.

```python
    def timed_forward(self, x: torch.Tensor) :
        start_time = time.perf_counter()
        with nvtx_range("blk_attn"):
            with nvtx_range("layernorm"):
                xn = self.norm1(x)
            x = x + self.drop_path1(self.ls1(self.attn(xn)))
        end_time = time.perf_counter()
        attn_time = end_time - start_time

        start_time = time.perf_counter()
        with nvtx_range("blk_mlp"):
            with nvtx_range("layernorm"):
                xn = self.norm2(x)
            x = x + self.drop_path2(self.ls2(self.mlp(xn)))
        end_time = time.perf_counter()
        mlp_time = end_time - start_time
        return x, attn_time, mlp_time
```

(Only change vs. the original: `self.norm1(x)` / `self.norm2(x)` are pulled out into a
named `xn` so they can sit inside their own `layernorm` range before being fed to
`self.attn`/`self.mlp`. Same values, same result.)

### 4d. `_pos_embed` — `pos_embed`

Wrap the whole method body (it has multiple early `return`s — that's fine, a `with`
block's exit still runs on `return`):

```python
    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        with nvtx_range("pos_embed"):
            if self.pos_embed is None:
                return x.view(x.shape[0], -1, x.shape[-1])

            if self.dynamic_img_size:
                B, H, W, C = x.shape
                pos_embed = resample_abs_pos_embed(
                    self.pos_embed,
                    (H, W),
                    num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
                )
                x = x.view(B, -1, C)
            else:
                pos_embed = self.pos_embed

            to_cat = []
            if self.cls_token is not None:
                to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
            if self.reg_token is not None:
                to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

            if self.no_embed_class:
                x = x + pos_embed
                if to_cat:
                    x = torch.cat(to_cat + [x], dim=1)
            else:
                if to_cat:
                    x = torch.cat(to_cat + [x], dim=1)
                x = x + pos_embed

            return self.pos_drop(x)
```

(Re-indented one level, comments trimmed for brevity — keep the original comments in
your actual edit, only the indentation + outer `with` are new.)

### 4e. `pool` — `pool`

```python
    def pool(self, x: torch.Tensor, pool_type: Optional[str] = None) -> torch.Tensor:
        with nvtx_range("pool"):
            if self.attn_pool is not None:
                x = self.attn_pool(x)
                return x
            pool_type = self.global_pool if pool_type is None else pool_type
            x = global_pool_nlc(x, pool_type=pool_type, num_prefix_tokens=self.num_prefix_tokens)
            return x
```

### 4f. `VisionTransformer.timed_forward` — `layernorm` (final norm) + `linear` (head)

The `patch_embed`/`conv2d` and `pos_embed` ranges are already covered by steps 2 and 4d
(they live inside the methods this calls), so this only needs two small additions —
around the final `self.norm(x)` and the `self.head(x)` call:

```python
    def timed_forward(self, x: torch.Tensor) :
        times = []

        start_time = time.perf_counter()
        x = self.patch_embed(x)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

        start_time = time.perf_counter()
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

        for block in self.blocks :
            x, attn_time, mlp_time = block.timed_forward(x)
            times.append(attn_time)
            times.append(mlp_time)

        start_time = time.perf_counter()
        with nvtx_range("layernorm"):
            x = self.norm(x)
        x = self.pool(x)
        x = self.fc_norm(x)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

        start_time = time.perf_counter()
        x = self.head_drop(x)
        with nvtx_range("linear"):
            x = self.head(x)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

        return x, times
```

(Only the two `with nvtx_range(...)` lines are new; everything else — including the
`time.perf_counter()` timing that already existed — is unchanged.)

---

## Step 5 — sanity check: nothing broke, nothing slowed down

```bash
# 1. USE_NVTX unset -> nvtx_range is a no-op, output must be byte-identical to before
python3 timm_train_vit/vit.py models/vit_1.pt data/prof/pic_b16.cpic /tmp/py_before.cprd /tmp/m.csv
git stash            # temporarily undo the edits
python3 timm_train_vit/vit.py models/vit_1.pt data/prof/pic_b16.cpic /tmp/py_after.cprd /tmp/m.csv
git stash pop
python3 scripts/compare_cpred.py /tmp/py_before.cprd /tmp/py_after.cprd /tmp/cmp.txt 0.0 0.0
# expect: identical (threshold 0.0) -- the ranges must not change any values

# 2. USE_NVTX=1 -> the range_push/pop path is actually taken (no GPU, no nsys needed here)
USE_NVTX=1 python3 timm_train_vit/vit.py models/vit_1.pt data/prof/pic_b16.cpic /tmp/py_nvtx.cprd /tmp/m.csv
# should run without error. Without a profiler attached the ranges are silent no-ops;
# they only become visible when run under `nsys` (step 6). No `module load cuda` needed
# for this step -- that's only required for the `nsys` binary itself.
```

---

## Step 6 — wire it into `profile_nsys.sh`

Add a third capture, after the existing OpenMP one (around line 69, right before the
"5. NVTX region time summary" section). Uses `models/vit_1.pt` (not `.cvit` — the
Python path loads a PyTorch state dict, see `run_py.sh`):

```bash
echo
echo "=== Python capture -> reports/py_b${B} ==="
USE_NVTX=1 nsys profile $NSYS_ARGS -o "reports/py_b${B}" \
    python3 timm_train_vit/vit.py models/vit_1.pt "$PIC" "reports/py_prd_b${B}.cprd" "$MEAS"
```

And add its summary table next to the other two in step 5 of that script:
```bash
echo
echo "=== NVTX region summary: PYTHON (reports/py_b${B}) ==="
nsys stats --report nvtx_pushpop_sum --format table "reports/py_b${B}.nsys-rep" || true
```

Also update the final echo to mention the new report file, and the module-load comment
at the top of `profile.slurm` needs no change — `venv` is already activated there for
model/input generation, so `python3`/`torch`/`timm` are already on PATH for this new
capture too.

`models/vit_1.pt` is already produced by `create_models.sh`, which `profile_nsys.sh`
already calls in step 2 — no new prerequisite.

---

## Step 7 — run it and compare

```bash
sbatch profile.slurm
tail -f <jobid>.out
```

`profile.slurm` as written requests `boost_usr_prod` + `--gres=gpu:1`. That still works
for this all-CPU capture (the GPU just sits idle), but since nothing here needs it you
can also run on a CPU partition to avoid spending GPU-hours — switch to
`--partition=dcgp_usr_prod` (with its matching `--qos`) and drop the `--gres` line, as
the comment at the top of `profile.slurm` already suggests. Either way the three captures
run identically.

You should now see three `nsys stats --report nvtx_pushpop_sum` tables in the job
output — `cpp_b16`, `omp_b16`, `py_b16` — with the same range names
(`linear`, `attention`, `patch_embed`, `conv2d`, `pos_embed`, `blk_attn`, `blk_mlp`,
`layernorm`, `gelu`, `pool`) in each, so `Total Time` per range is directly comparable
row-for-row across all three implementations. To compare visually, `scp` all three
`.nsys-rep` files (see `PROFILING_GUIDE.md` §8) and open them side by side in the Nsight
Systems GUI — same NVTX row expansion, three timelines.

Remember the one apples-to-oranges caveat from step 4b: Python's `linear`/`attention`
**instance counts** differ from C++'s (fused qkv Linear, possibly fused SDPA kernel) —
compare `Total Time`, not `Instances`.
