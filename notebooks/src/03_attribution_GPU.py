# %% [markdown]
# # RUNTIME: local workstation GPU (NVIDIA RTX 3060 12 GB), `.venv-local` (deviation D4)
#
# Project: `qpain/paper-01` · Stage: `03 attribution` · Estimated time: measured by the smoke test, then the compute plan
#
# For every eligible model and seed (notebook 02, validation-only eligibility), computes Grad-CAM, Integrated Gradients,
# occlusion, the random-map and edge-map controls (and attention rollout for the two DeiT models, descriptive) on the
# fixed attribution list, all for the TRUE class, on a shared 14 x 14 grid, then deletion and insertion curves with a
# blurred fill (primary) and a mean-intensity fill (sensitivity) (analysis_plan.md sections 5-6, clarifications D8).
# Also: the Grad-CAM validity check on 200 validation images, the head-randomisation sanity check (seed 11), and the
# exploratory OCTDL DME/NORMAL images (blurred fill only). The speckle analysis is not run here (conditional, section 8).
#
# Run: `PAPER_ROOT=<project> python -m jupyter nbconvert --to notebook --execute notebooks/03_attribution_GPU.ipynb ...`
# Smoke test: `SMOKE=1` uses the smoke checkpoints of notebook 02 (seed 11, all six models), 1 attribution image per
# class, 1 OCTDL image per class and 20 validation images; prints the time per image for every model and method;
# nothing is written to outputs/. `DEVICE=cpu` forces the CPU.
# Source of this notebook: `notebooks/src/03_attribution_GPU.py` (edit the source, then rebuild with scripts/py-to-ipynb.py).
# Revised after Codex review 004 (results/004-review-notebook-03.md; decisions/006-notebook-03-review.md).

# %%
# Setup: paths and parameters fixed by analysis_plan.md sections 5-6 (and clarifications D8)
import os
import sys, io, json, time, hashlib, zipfile, random, platform, datetime, subprocess
import importlib.metadata as ilmd
import numpy as np, pandas as pd
from PIL import Image
from scipy.stats import rankdata
import torch, torch.nn.functional as F
import timm

ROOT = os.environ.get("PAPER_ROOT", "/content/paper-01")
RAW, OUT, CKPT = (os.path.join(ROOT, d) for d in ("data/raw", "outputs", "checkpoints"))
DERIVED = os.path.join(ROOT, "data", "derived")
SMOKE = os.environ.get("SMOKE", "0") == "1"
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
NOTEBOOK = "03_attribution_GPU"
RUNS_IN = os.path.join(CKPT, "smoke" if SMOKE else "runs")          # notebook 02's checkpoints
ATTR = os.path.join(CKPT, "attr_smoke" if SMOKE else "attr")         # this notebook's per-run results (local)
for d in (OUT, ATTR, DERIVED):
    os.makedirs(d, exist_ok=True)

SPLIT_SHA256 = "655b1785339d81d629ec8794e7698a6ada55161994a7fe02fab20a9073a506a6"           # D3e
OCTDL_INVENTORY_SHA256 = "357e26a6a263009ac160d161d0661bb956251b405505751fe3ec0e32348197cb"  # D7
RAW_SHA256 = {"ZhangLabData.zip": "acbc42869440eec719361942d0ba560f132742230262f22e78bfb93eb5ea4563",
              "OCTDL.zip": "637a72ff3319645aac30408d4d838189337ac0b8b0f9faabf7c8156d271f46f7"}
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
MODELS = [("efficientnet_b0", "convolutional"), ("mobilenetv3_large_100", "convolutional"), ("mobilevit_s", "hybrid"),
          ("efficientformer_l1", "hybrid"), ("deit_tiny_patch16_224", "transformer"), ("deit_small_patch16_224", "transformer")]
SEEDS = [11] if SMOKE else [11, 22, 33]
SEED = 20261119
IMG, GRID, CELL = 224, 14, 16          # 14 x 14 grid of 16 x 16-pixel cells
STEP = 14                              # cells per curve step: 15 points, 0..196 cells
KS = list(range(0, GRID * GRID + 1, STEP))
assert len(KS) == 15
BLUR_SIGMA = 11.0                      # Gaussian blur fill, sigma in pixels on the 224 x 224 input
N_RANDOM = 5                           # random orders per image for the random-map control
IG_STEPS = 32
GC_VALID_N, GC_VALID_SHARE = (20 if SMOKE else 200), 0.95
SANITY_SEED, SANITY_RHO = 11, 0.9
CHUNK = 128                            # images per forward batch
ORDER_CHUNK = 4                        # orders per perturbation batch in the curves (memory bound)
# Grad-CAM target layers, fixed before any test image is used (plan section 5); the second entry is a fallback tried
# only if the first fails the validity check on validation images. "tokens" = (B, N, C) reshaped to a square grid.
GC_LAYERS = {
    "efficientnet_b0": [("bn2", "conv"), ("blocks.6", "conv")],
    "mobilenetv3_large_100": [("blocks", "conv"), ("blocks.5", "conv")],
    "mobilevit_s": [("final_conv", "conv"), ("stages.4", "conv")],
    "efficientformer_l1": [("stages.3.blocks.2", "conv"), ("norm", "tokens")],
    "deit_tiny_patch16_224": [("blocks.11.norm1", "tokens_cls"), ("blocks.10.norm1", "tokens_cls")],
    "deit_small_patch16_224": [("blocks.11.norm1", "tokens_cls"), ("blocks.10.norm1", "tokens_cls")],
}
HEADS = {"efficientnet_b0": ["classifier"], "mobilenetv3_large_100": ["classifier"], "mobilevit_s": ["head.fc"],
         "efficientformer_l1": ["head", "head_dist"], "deit_tiny_patch16_224": ["head"], "deit_small_patch16_224": ["head"]}
ROLLOUT_MODELS = {"deit_tiny_patch16_224", "deit_small_patch16_224"}

torch.backends.cuda.matmul.allow_tf32 = True      # D8: float32 weights and activations, TF32 matrix products
torch.backends.cudnn.allow_tf32 = True
VERSIONS = {m: ilmd.version(m) for m in ("torch", "timm", "numpy", "pandas", "Pillow", "scipy")}
ENV = {"python": platform.python_version(), "platform": platform.platform(), "device": DEVICE, "packages": VERSIONS,
       "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None, "smoke": SMOKE}
if DEVICE == "cpu":
    torch.set_num_threads(min(12, os.cpu_count() or 1))
print(json.dumps(ENV, indent=1))

# %%
# Helpers (same conventions as notebook 02): hashing, atomic writes, manifest-recorded outputs
def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def atomic(path, write):
    tmp = path + ".tmp"
    write(tmp)
    os.replace(tmp, path)

def write_json(path, obj):
    def w(t):
        with open(t, "w", newline="\n") as f:
            json.dump(obj, f, indent=1, default=str)
    atomic(path, w)

def save_output(obj, filename, description, cell=""):
    assert not SMOKE, "smoke runs never write to outputs/"
    path = os.path.join(OUT, filename)
    if hasattr(obj, "to_csv"):
        comp = "gzip" if filename.endswith(".gz") else None
        atomic(path, lambda t: obj.to_csv(t, index=False, lineterminator="\n", compression=comp))
    else:
        write_json(path, obj)
    man = os.path.join(OUT, "MANIFEST.md")
    rows = [l for l in open(man).read().splitlines() if not l.startswith(f"| {filename} |")]
    rows.append(f"| {filename} | {NOTEBOOK} | {cell} | {description} |")
    def w(t):
        with open(t, "w", newline="\n") as f:
            f.write("\n".join(rows) + "\n")
    atomic(man, w)
    print("saved", filename)
    return path

# %%
# Frozen inputs and the images: attribution list (Kermany test), OCTDL exploratory test, validation sample for the
# Grad-CAM validity check. Images are decoded from the verified raw zips with notebook 02's preprocessing (greyscale,
# square bicubic resize to 224) and, in a full run, checked pixel for pixel against notebook 02's caches.
assert sha256_file(os.path.join(OUT, "kermany_split.csv")) == SPLIT_SHA256
assert sha256_file(os.path.join(OUT, "octdl_inventory.csv")) == OCTDL_INVENTORY_SHA256
for f, h in RAW_SHA256.items():
    assert sha256_file(os.path.join(RAW, f)) == h, f
ker = pd.read_csv(os.path.join(OUT, "kermany_split.csv"), dtype={"patient_id": str, "group": str})
# Rows come from the frozen split itself (class, patient, group); the published list must agree with them exactly
attr = ker.loc[ker["in_attribution_test"], ["member", "class", "patient_id", "group", "partition"]].copy()
assert set(attr["partition"]) == {"test"} and attr["member"].is_unique and not attr.duplicated(["patient_id", "class"]).any()
_lst = pd.read_csv(os.path.join(OUT, "attribution_test_list.csv"), dtype={"patient_id": str, "group": str})
_chk = attr.merge(_lst, on="member", how="outer", suffixes=("", "_list"), indicator=True)
assert (_chk["_merge"] == "both").all() and len(_lst) == len(attr), "attribution_test_list.csv and the split disagree on members"
assert all((_chk[c] == _chk[c + "_list"]).all() for c in ("class", "patient_id", "group")), "attribution_test_list.csv and the split disagree on class, patient or group"
octdl = pd.read_csv(os.path.join(OUT, "octdl_inventory.csv"), dtype={"patient_id": str})
octdl_t = octdl[octdl["in_exploratory_test"]].copy()
octdl_t["group"] = "octdl_" + octdl_t["group"].astype(str)
val = ker[ker["partition"] == "val"].sample(GC_VALID_N, random_state=SEED)       # validation only: Grad-CAM validity

if SMOKE:
    pick = lambda d, n: pd.concat([g.sort_values("member").head(n) for _, g in d.groupby("class")])
    attr, octdl_t = pick(attr, 1), pick(octdl_t, 1)
attr = attr.sort_values(["class", "member"]).reset_index(drop=True)
octdl_t = octdl_t.sort_values(["class", "member"]).reset_index(drop=True)
val = val.sort_values("member").reset_index(drop=True)
print("attribution images", len(attr), attr["class"].value_counts().to_dict(), "| OCTDL", len(octdl_t), octdl_t["class"].value_counts().to_dict(), "| validation", len(val))

def load_images(zip_name, members):
    arr = np.empty((len(members), IMG, IMG), dtype=np.uint8)
    with zipfile.ZipFile(os.path.join(RAW, zip_name)) as z:
        for i, m in enumerate(members):
            with Image.open(io.BytesIO(z.read(m))) as im:
                arr[i] = np.asarray(im.convert("L").resize((IMG, IMG), Image.BICUBIC))
    return arr

X = {"kermany": load_images("ZhangLabData.zip", attr["member"].tolist()),
     "octdl": load_images("OCTDL.zip", octdl_t["member"].tolist()),
     "val": load_images("ZhangLabData.zip", val["member"].tolist())}
Y = {"kermany": attr["class"].map(CLASSES.index).to_numpy(), "octdl": octdl_t["class"].map(CLASSES.index).to_numpy(),
     "val": val["class"].map(CLASSES.index).to_numpy()}
if not SMOKE:   # same pixels as the images the models were trained and evaluated on
    for name, df, key in (("kermany_test", ker[ker["partition"] == "test"], "kermany"), ("octdl_test", octdl[octdl["in_exploratory_test"]], "octdl"),
                          ("kermany_val", ker[ker["partition"] == "val"], "val")):
        cache = np.load(os.path.join(DERIVED, f"{name}.npy"), mmap_mode="r")
        pos = {m: i for i, m in enumerate(df.sort_values("member")["member"])}
        src = {"kermany": attr, "octdl": octdl_t, "val": val}[key]
        idx = [pos[m] for m in src["member"]]
        assert np.array_equal(np.asarray(cache[idx]), X[key]), f"pixels differ from notebook 02's cache: {name}"
    print("pixels identical to notebook 02's caches")

# %%
# Checkpoints and eligibility: each run's best checkpoint from notebook 02, verified by hash; only eligible models
markers = {}
for name, family in MODELS:
    for seed in SEEDS:
        run = f"{name}_s{seed}"
        d = json.load(open(os.path.join(RUNS_IN, run + ".json")))
        assert sha256_file(os.path.join(RUNS_IN, run + ".pt")) == d["weights_sha256"], f"{run}: checkpoint hash mismatch"
        assert d["classes"] == CLASSES and d["img"] == IMG
        markers[run] = d
if SMOKE:
    eligible = [m for m, _ in MODELS]
else:
    elig = pd.read_csv(os.path.join(OUT, "model_eligibility.csv"))
    ck = pd.read_csv(os.path.join(OUT, "model_checkpoints.csv")).set_index("run")["weights_sha256"]
    assert all(ck[r] == d["weights_sha256"] for r, d in markers.items()), "outputs/model_checkpoints.csv and the run markers disagree"
    eligible = elig.loc[elig["eligible"], "model"].tolist()
    print("not eligible (validation balanced accuracy < 0.85 on some seed):", elig.loc[~elig["eligible"], "model"].tolist())
print("eligible models:", eligible)
if not eligible:
    print("NO ELIGIBLE MODEL: no attribution is computed; notebook 04 reports every contrast as not estimable.")

def load_model(run):
    d = markers[run]
    m = timm.create_model(d["model"], pretrained=False, num_classes=len(CLASSES))
    m.load_state_dict(torch.load(os.path.join(RUNS_IN, run + ".pt"), map_location="cpu", weights_only=True), strict=True)
    m.eval().to(DEVICE)
    for p in m.parameters():
        p.requires_grad_(False)
    mean = torch.tensor(d["mean"], device=DEVICE).view(1, 3, 1, 1); std = torch.tensor(d["std"], device=DEVICE).view(1, 3, 1, 1)
    return m, mean, std

# %%
# Building blocks: normalisation, fills, the cell grid, maps to the 14 x 14 grid, curves and AUCs
CELL_ID = torch.arange(GRID * GRID, device=DEVICE).view(GRID, GRID).repeat_interleave(CELL, 0).repeat_interleave(CELL, 1)   # (224, 224)
_r = int(np.ceil(3 * BLUR_SIGMA))
_k = torch.exp(-0.5 * (torch.arange(-_r, _r + 1, dtype=torch.float32) / BLUR_SIGMA) ** 2)
BLUR_K = (_k / _k.sum()).to(DEVICE)

def blur(x):            # x: (N, 224, 224) in [0, 1]; separable Gaussian, reflect padding
    y = F.pad(x[:, None], (_r, _r, 0, 0), mode="reflect")
    y = F.conv2d(y, BLUR_K.view(1, 1, 1, -1))
    y = F.pad(y, (0, 0, _r, _r), mode="reflect")
    return F.conv2d(y, BLUR_K.view(1, 1, -1, 1))[:, 0]

def fills(x):           # the two fills of section 6: blurred copy (primary) and the image's mean intensity
    return {"blur": blur(x), "mean": x.mean(dim=(1, 2), keepdim=True).expand_as(x).contiguous()}

def norm(x01, mean, std):   # (N, 224, 224) in [0, 1] -> (N, 3, 224, 224) normalised as in training
    return (x01[:, None].expand(-1, 3, -1, -1) - mean) / std

def to_grid(m):         # (N, H, W) map at any resolution -> (N, 14, 14) by bilinear upsampling to 224 and 16 x 16 cell means
    if m.shape[-1] != IMG:
        m = F.interpolate(m[:, None].float(), size=(IMG, IMG), mode="bilinear", align_corners=False)[:, 0]
    return F.avg_pool2d(m[:, None].float(), CELL)[:, 0]

@torch.no_grad()
def prob_true(model, mean, std, x01, target):
    out = []
    for i in range(0, len(x01), CHUNK):
        out.append(model(norm(x01[i:i + CHUNK], mean, std)).float().softmax(1)[:, target])
    return torch.cat(out)

def tick():               # synchronized wall clock, so GPU timings are real (Codex review 004, point 5)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return time.time()

def image_rng(dataset, member):   # per-image seed: identical for every model, seed, method and fill
    return np.random.default_rng(int(hashlib.sha256(f"{SEED}|{dataset}|{member}".encode()).hexdigest()[:15], 16))

def order_from(sal, tiebreak):    # decreasing saliency, ties broken by the image's seeded random ranking
    return np.lexsort((tiebreak, -np.asarray(sal, dtype=np.float64).ravel()))

def curve_aucs(model, mean, std, x, target, orders, fill_maps):
    """x (224, 224) on the device; orders (O, 196). Returns {fill: (del_auc (O,), ins_auc (O,), del_curve, ins_curve)}."""
    O = len(orders)
    rank = torch.empty((O, GRID * GRID), dtype=torch.long, device=DEVICE)
    rank.scatter_(1, torch.as_tensor(orders, device=DEVICE), torch.arange(GRID * GRID, device=DEVICE).expand(O, -1))
    ks = torch.tensor(KS, device=DEVICE)
    xs = torch.tensor(KS, dtype=torch.float64) / (GRID * GRID)
    res = {}
    for fname, f in fill_maps.items():
        pd_, pi_ = [], []
        for o0 in range(0, O, ORDER_CHUNK):                          # a few orders at a time bounds memory
            m = rank[o0:o0 + ORDER_CHUNK, CELL_ID][:, None] < ks[None, :, None, None]   # (o, 15, 224, 224): cell affected at step k
            dele = torch.where(m, f, x).view(-1, IMG, IMG)
            ins = torch.where(m, x, f).view(-1, IMG, IMG)
            n = len(m)
            p = prob_true(model, mean, std, torch.cat([dele, ins]), target).double().cpu().view(2, n, len(KS))
            pd_.append(p[0]); pi_.append(p[1])
        pdel, pins = torch.cat(pd_), torch.cat(pi_)
        res[fname] = (torch.trapezoid(pdel, xs, dim=-1).numpy(), torch.trapezoid(pins, xs, dim=-1).numpy(), pdel.numpy(), pins.numpy())
    return res

# %%
# Explanation methods (all for the true class; maps returned on the 14 x 14 grid)
def get_module(model, path):
    m = model
    for p in path.split("."):
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m

def gradcam(model, mean, std, x01, targets, layer, kind):
    store = {}
    h = get_module(model, layer).register_forward_hook(lambda mod, i, o: store.__setitem__("a", o))
    maps, ok = [], []
    try:
        for i in range(0, len(x01), 32):
            xb = norm(x01[i:i + 32], mean, std).requires_grad_(False)
            with torch.enable_grad():
                xb = xb.clone().requires_grad_(True)      # makes the graph exist although weights are frozen
                logits = model(xb).float()
                a = store["a"]
                g, = torch.autograd.grad(logits[torch.arange(len(xb)), torch.as_tensor(targets[i:i + 32], device=DEVICE)].sum(), a)
            a, g = a.detach().float(), g.float()
            if kind != "conv":                            # tokens (B, N, C) -> (B, C, s, s)
                if kind == "tokens_cls":
                    a, g = a[:, 1:], g[:, 1:]
                s = int(round(a.shape[1] ** 0.5)); assert s * s == a.shape[1]
                a, g = (t.transpose(1, 2).reshape(t.shape[0], t.shape[2], s, s) for t in (a, g))
            cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1))
            grid = to_grid(cam)                           # validity is judged on the map that is scored (14 x 14)
            ok.append((torch.isfinite(grid).flatten(1).all(1) & (g.abs().amax(dim=(1, 2, 3)) > 0)
                       & (torch.nan_to_num(grid).flatten(1).std(1) > 1e-8)).cpu())
            maps.append(grid.cpu())
    finally:
        h.remove()
    return torch.cat(maps).numpy(), torch.cat(ok).numpy()

def integrated_gradients(model, mean, std, x, target):
    xn = norm(x[None], mean, std)
    x0 = norm(torch.zeros_like(x)[None], mean, std)                  # black baseline
    alphas = ((torch.arange(IG_STEPS, device=DEVICE) + 0.5) / IG_STEPS).view(-1, 1, 1, 1)   # midpoint rule
    with torch.enable_grad():
        path = (x0 + alphas * (xn - x0)).requires_grad_(True)
        g, = torch.autograd.grad(model(path).float()[:, target].sum(), path)
    attr_ = ((xn - x0)[0] * g.mean(0)).sum(0)                        # signed, summed over channels (D8)
    with torch.no_grad():                                            # completeness diagnostic (Codex review 004, point 6)
        lg = model(torch.cat([xn, x0])).float()[:, target]
    return to_grid(attr_[None])[0].cpu().numpy(), float(attr_.sum()), float(lg[0] - lg[1])

def occlusion(model, mean, std, x, target, fill):
    m = (CELL_ID[None] == torch.arange(GRID * GRID, device=DEVICE).view(-1, 1, 1))   # (196, 224, 224)
    p0 = prob_true(model, mean, std, x[None], target)[0]
    p = prob_true(model, mean, std, torch.where(m, fill[None], x[None]), target)
    return (p0 - p).view(GRID, GRID).cpu().numpy()

SOBEL = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=DEVICE)
def edge_map(x):
    gx = F.conv2d(F.pad(x[None, None], (1, 1, 1, 1), mode="reflect"), SOBEL.view(1, 1, 3, 3))
    gy = F.conv2d(F.pad(x[None, None], (1, 1, 1, 1), mode="reflect"), SOBEL.t().reshape(1, 1, 3, 3))
    return to_grid(torch.sqrt(gx ** 2 + gy ** 2)[:, 0])[0].cpu().numpy()

def rollout(model, mean, std, x01):
    """Attention rollout (Abnar and Zuidema 2020) for DeiT: head-averaged attention plus identity, row-normalised,
    multiplied over the blocks; the class-token row over the 196 patches. Descriptive only (plan section 5)."""
    att, hooks = [], []
    fused = [blk.attn.fused_attn for blk in model.blocks]
    for blk in model.blocks:
        blk.attn.fused_attn = False
        hooks.append(blk.attn.attn_drop.register_forward_hook(lambda mod, i, o: att.append(o.detach().float())))
    try:
        out = []
        for i in range(0, len(x01), 32):
            att.clear()
            with torch.no_grad():
                model(norm(x01[i:i + 32], mean, std))
            R = None
            for a in att:
                a = a.mean(1); a = a + torch.eye(a.shape[-1], device=a.device); a = a / a.sum(-1, keepdim=True)
                R = a if R is None else a @ R
            out.append(to_grid(R[:, 0, 1:].reshape(-1, GRID, GRID)).cpu())
    finally:
        for h in hooks:
            h.remove()
        for blk, f0 in zip(model.blocks, fused):
            blk.attn.fused_attn = f0
    return torch.cat(out).numpy()

# %%
# Grad-CAM validity (plan section 5): per model, on validation images only, using the seed-11 checkpoint
# One layer per model, valid on every seed's checkpoint (Codex review 004, point 3); judged on the scored 14 x 14 grid.
validity_rows, GC_CHOICE = [], {}
xval = torch.from_numpy(X["val"]).to(DEVICE).float() / 255
for name, _ in MODELS:
    if name not in eligible:
        continue
    GC_CHOICE[name] = None
    for layer, kind in GC_LAYERS[name]:
        shares = {}
        for seed in SEEDS:
            model, mean, std = load_model(f"{name}_s{seed}")
            _, ok = gradcam(model, mean, std, xval, Y["val"], layer, kind)
            shares[seed] = float(ok.mean())
            validity_rows.append({"model": name, "seed": seed, "layer": layer, "kind": kind, "valid_share": shares[seed], "n": len(ok),
                                  "valid": shares[seed] >= GC_VALID_SHARE})
            del model
        print(f"Grad-CAM validity {name} {layer}: " + ", ".join(f"seed {k} {v:.3f}" for k, v in shares.items()))
        if min(shares.values()) >= GC_VALID_SHARE:
            GC_CHOICE[name] = (layer, kind)
            break
validity = pd.DataFrame(validity_rows)
print("Grad-CAM layers:", GC_CHOICE)

# %%
# Per run: maps, curves and AUCs for every image; per-run results saved locally with a marker (resumable)
SIGNATURE = hashlib.sha256(json.dumps({"code": sha256_file(os.path.join(ROOT, "notebooks", "src", "03_attribution_GPU.py")),
    "split": SPLIT_SHA256, "octdl": OCTDL_INVENTORY_SHA256, "gc": GC_CHOICE, "packages": VERSIONS, "device": DEVICE,
    "members": [attr["member"].tolist(), octdl_t["member"].tolist()]}, sort_keys=True).encode()).hexdigest()

def run_signature(run):   # bound to the verified checkpoint, its normalisation and the execution settings (Codex review 004, point 1)
    d = markers[run]
    return hashlib.sha256(json.dumps({"base": SIGNATURE, "weights": d["weights_sha256"], "mean": d["mean"], "std": d["std"], "img": d["img"],
        "chunk": CHUNK, "order_chunk": ORDER_CHUNK, "gc": GC_CHOICE.get(d["model"]), "tf32": True}, sort_keys=True).encode()).hexdigest()

def run_attribution(run):
    marker_path = os.path.join(ATTR, run + ".json")
    need = {f"{run}_scores.csv.gz", f"{run}_curves.csv.gz", f"{run}_maps.npz"}
    try:
        mk = json.load(open(marker_path))
        if (mk["signature"] == run_signature(run) and set(mk["files"]) == need
                and all(sha256_file(os.path.join(ATTR, f)) == h for f, h in mk["files"].items())):
            print("resume: verified, skipped", run); return mk
    except (OSError, ValueError, KeyError):
        pass
    d = markers[run]; name = d["model"]
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, mean, std = load_model(run)
    rows, curves, maps_out, t_method = [], [], {}, {}
    for dataset, df in (("kermany", attr), ("octdl", octdl_t)):
        x_all = torch.from_numpy(X[dataset]).to(DEVICE).float() / 255
        tg = Y[dataset]
        t0 = tick()
        gc_maps = gradcam(model, mean, std, x_all, tg, *GC_CHOICE[name])[0] if GC_CHOICE[name] else None
        t_method["gradcam"] = t_method.get("gradcam", 0) + tick() - t0
        ro_maps = None
        if name in ROLLOUT_MODELS:
            t0 = tick(); ro_maps = rollout(model, mean, std, x_all); t_method["rollout"] = t_method.get("rollout", 0) + tick() - t0
        computed = ["ig", "occlusion", "edge"] + (["gradcam"] if gc_maps is not None else []) + (["rollout"] if ro_maps is not None else [])
        fill_names = ["blur", "mean"] if dataset == "kermany" else ["blur"]    # D8: OCTDL with the primary fill only
        M = {k: np.zeros((len(df), GRID, GRID), np.float32) for k in ("gradcam", "ig", "occlusion", "edge", "rollout")}
        for i in range(len(df)):
            x, t, member = x_all[i], int(tg[i]), df["member"].iat[i]
            fm = fills(x[None]); fm = {k: v[0] for k, v in fm.items() if k in fill_names}
            rng = image_rng(dataset, member)
            tiebreak = rng.permutation(GRID * GRID)
            rand_orders = np.stack([rng.permutation(GRID * GRID) for _ in range(N_RANDOM)])
            t0 = tick(); ig, ig_sum, ig_diff = integrated_gradients(model, mean, std, x, t); t_method["ig"] = t_method.get("ig", 0) + tick() - t0
            t0 = tick(); oc = occlusion(model, mean, std, x, t, fills(x[None])["blur"][0]); t_method["occlusion"] = t_method.get("occlusion", 0) + tick() - t0
            maps = {"ig": ig, "occlusion": oc, "edge": edge_map(x)}
            if gc_maps is not None:
                maps["gradcam"] = gc_maps[i]
            if ro_maps is not None:
                maps["rollout"] = ro_maps[i]
            for k, v in maps.items():
                assert np.isfinite(v).all(), f"{run} {member}: non-finite {k} map"
                M[k][i] = v
            methods = list(maps)
            orders = np.stack([order_from(maps[k], tiebreak) for k in methods] + list(rand_orders))
            t0 = tick()
            res = curve_aucs(model, mean, std, x, t, orders, fm)
            t_method["curves"] = t_method.get("curves", 0) + tick() - t0
            with torch.no_grad():
                probs = model(norm(x[None], mean, std)).float().softmax(1)[0]
            p_orig, pred = float(probs[t]), CLASSES[int(probs.argmax())]     # same forward: for the all-correct sensitivity subset
            for fname, (dA, iA, dC, iC) in res.items():
                rd, ri = dA[len(methods):], iA[len(methods):]
                for j, meth in enumerate(methods + ["random"]):
                    if meth == "random":
                        del_auc, ins_auc, sd_d, sd_i = float(rd.mean()), float(ri.mean()), float(rd.std()), float(ri.std())
                        dc, ic = dC[len(methods):].mean(0), iC[len(methods):].mean(0)
                    else:
                        del_auc, ins_auc, sd_d, sd_i = float(dA[j]), float(iA[j]), np.nan, np.nan
                        dc, ic = dC[j], iC[j]
                    rows.append({"dataset": dataset, "member": member, "class": df["class"].iat[i], "group": df["group"].iat[i],
                                 "patient_id": df["patient_id"].iat[i], "model": name, "family": d["family"], "seed": d["seed"],
                                 "method": meth, "fill": fname, "deletion_auc": del_auc, "insertion_auc": ins_auc,
                                 "random_deletion_sd": sd_d, "random_insertion_sd": sd_i, "p_true_original": p_orig, "pred_class_original": pred,
                                 "ig_attribution_sum": ig_sum if meth == "ig" else np.nan, "ig_logit_difference": ig_diff if meth == "ig" else np.nan})
                    curves.append([dataset, member, name, d["seed"], meth, fname] + list(np.round(dc, 6)) + list(np.round(ic, 6)))
        maps_out[dataset] = {k: M[k] for k in computed}
    sc = pd.DataFrame(rows)
    base = sc[sc["method"] == "random"].set_index(["dataset", "member", "fill"])
    key = list(zip(sc["dataset"], sc["member"], sc["fill"]))
    sc["faithfulness_gap"] = (sc["insertion_auc"] - sc["deletion_auc"]) - (base.loc[key, "insertion_auc"].to_numpy() - base.loc[key, "deletion_auc"].to_numpy())
    cv = pd.DataFrame(curves, columns=["dataset", "member", "model", "seed", "method", "fill"] + [f"del_{k}" for k in KS] + [f"ins_{k}" for k in KS])
    files = {}
    for fn, obj in ((f"{run}_scores.csv.gz", sc), (f"{run}_curves.csv.gz", cv)):
        atomic(os.path.join(ATTR, fn), lambda t, o=obj: o.to_csv(t, index=False, lineterminator="\n", compression="gzip"))
        files[fn] = sha256_file(os.path.join(ATTR, fn))
    def w(t):
        with open(t, "wb") as f:
            np.savez_compressed(f, **{f"{ds}_{k}": v.astype(np.float32) for ds, mm in maps_out.items() for k, v in mm.items()})
    atomic(os.path.join(ATTR, f"{run}_maps.npz"), w)
    files[f"{run}_maps.npz"] = sha256_file(os.path.join(ATTR, f"{run}_maps.npz"))
    n_img = len(attr) + len(octdl_t)
    mk = {"run": run, "signature": run_signature(run), "files": files, "seconds_by_part": t_method,
          "peak_gpu_mib_allocated": torch.cuda.max_memory_allocated() // 2**20 if DEVICE == "cuda" else None,
          "peak_gpu_mib_reserved": torch.cuda.max_memory_reserved() // 2**20 if DEVICE == "cuda" else None,
          "seconds_per_image": {k: v / n_img for k, v in t_method.items()}, "n_images": n_img}
    write_json(marker_path, mk)                       # marker last
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return mk

t_all = time.time()
done = {}
for name, _ in MODELS:
    if name not in eligible:
        continue
    for seed in SEEDS:
        run = f"{name}_s{seed}"
        t0 = time.time()
        done[run] = run_attribution(run)
        print(f"=== {run}: {time.time() - t0:.0f} s | per image: " + ", ".join(f"{k} {v:.2f} s" for k, v in done[run]["seconds_per_image"].items()))
print(f"attribution done in {(time.time() - t_all) / 60:.1f} min")

# %%
# Sanity check (plan section 6): seed-11 checkpoint, classifier head(s) re-initialised with a fixed seed; Grad-CAM and
# Integrated Gradients recomputed on the attribution images; Spearman correlation with the original maps per image.
sanity_rows = []
xk = torch.from_numpy(X["kermany"]).to(DEVICE).float() / 255
for name, _ in MODELS:
    if name not in eligible:
        continue
    run = f"{name}_s{SANITY_SEED}"
    orig = np.load(os.path.join(ATTR, f"{run}_maps.npz"))
    model, mean, std = load_model(run)
    torch.manual_seed(SEED)
    for h in HEADS[name]:
        get_module(model, h).reset_parameters()
    rnd = {"ig": np.stack([integrated_gradients(model, mean, std, xk[i], int(Y["kermany"][i]))[0] for i in range(len(xk))])}
    if GC_CHOICE[name]:
        rnd["gradcam"] = gradcam(model, mean, std, xk, Y["kermany"], *GC_CHOICE[name])[0]
    for meth, R in rnd.items():
        O = orig[f"kermany_{meth}"].astype(np.float64)
        R = R.astype(np.float64)
        def _rho(a, b):          # Spearman; undefined (NaN) when either map is constant
            ra, rb = rankdata(a.ravel()), rankdata(b.ravel())
            return np.nan if ra.std() == 0 or rb.std() == 0 else float(np.corrcoef(ra, rb)[0, 1])
        rho = np.array([_rho(O[i], R[i]) for i in range(len(O))])
        defined = rho[~np.isnan(rho)]
        status = "indeterminate" if len(defined) == 0 else ("insensitive" if np.median(defined) > SANITY_RHO else "sensitive")
        sanity_rows.append({"model": name, "seed": SANITY_SEED, "method": meth, "median_spearman": float(np.median(defined)) if len(defined) else np.nan,
                            "share_above_0.9_of_defined": float(np.mean(defined > SANITY_RHO)) if len(defined) else np.nan,
                            "n": len(rho), "n_undefined": int(np.isnan(rho).sum()), "status": status,
                            "note": "tests sensitivity to the classifier head only; edge maps and rollout do not depend on the head"})
    del model
sanity = pd.DataFrame(sanity_rows)
print(sanity.to_string(index=False))

# %%
# Assemble outputs (full run) or report the smoke timing
if not done:
    if not SMOKE:
        save_output({**ENV, "eligible_models": [], "note": "no model met the validation eligibility rule; no attribution computed"},
                    "attribution_environment.json", "Attribution environment: no eligible model, nothing computed", "assemble")
    raise SystemExit("no eligible model")
scores = pd.concat([pd.read_csv(os.path.join(ATTR, f"{r}_scores.csv.gz"), dtype={"patient_id": str, "group": str}) for r in done], ignore_index=True)
_key = ["dataset", "member", "model", "seed", "method", "fill"]
assert not scores.duplicated(_key).any(), "duplicate score keys"
_n = {"kermany": len(attr), "octdl": len(octdl_t)}
_cnt = scores.groupby(["dataset", "model", "seed", "method", "fill"]).size()
assert all(v == _n[k[0]] for k, v in _cnt.items()), "a model, seed, method and fill does not cover every image"
curves_all = pd.concat([pd.read_csv(os.path.join(ATTR, f"{r}_curves.csv.gz")) for r in done], ignore_index=True)
assert np.isfinite(scores[["deletion_auc", "insertion_auc", "faithfulness_gap"]]).all().all()
timing = pd.DataFrame([{"run": r, **{f"sec_per_image_{k}": v for k, v in mk["seconds_per_image"].items()}} for r, mk in done.items()])
print(scores.groupby(["dataset", "fill", "method"])["faithfulness_gap"].agg(["count", "mean"]).round(4).to_string())
print(timing.round(3).to_string(index=False))
if SMOKE:
    per_model = timing.set_index("run").sum(1)
    n_full = 820 + 217
    print("SMOKE TEST OK. Estimated full-run hours (3 seeds, %d images): %.1f" % (n_full, per_model.sum() * 3 * n_full / 3600))
else:
    save_output(scores, "faithfulness_scores.csv.gz", "Per image, model, seed, method and fill: deletion and insertion AUC (true-class probability), random-control SD, faithfulness gap, true-class probability and predicted class on the original image", "assemble")
    save_output(curves_all, "faithfulness_curves.csv.gz", "Deletion and insertion curves (15 points, 0-196 cells) per image, model, seed, method and fill; random = mean of 5 orders", "assemble")
    save_output(validity, "gradcam_validity.csv", "Grad-CAM layer validity on 200 validation images per model (seed-11 checkpoint): share of non-degenerate maps, chosen layer", "validity")
    save_output(sanity, "sanity_check_head_randomisation.csv", "Head-randomisation sanity check (seed 11): median Spearman correlation between original and randomised-head maps, flag above 0.9", "sanity")
    save_output(pd.DataFrame([{"run": r, "file": f, "sha256": h} for r, mk in done.items() for f, h in mk["files"].items()]),
                "attribution_files.csv", "SHA-256 of the per-run score, curve and map files (maps kept in checkpoints/attr/, not published)", "assemble")
    save_output({**ENV, "signature": SIGNATURE, "gradcam_layers": GC_CHOICE, "eligible_models": eligible, "timing_seconds_per_image": timing.to_dict("records"),
                 "n_images": {"kermany": len(attr), "octdl": len(octdl_t), "validation_for_gradcam_validity": len(val)},
                 "total_minutes": (time.time() - t_all) / 60, "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
                "attribution_environment.json", "Attribution environment, run signature, Grad-CAM layers, eligible models, timing per image and method", "assemble")
