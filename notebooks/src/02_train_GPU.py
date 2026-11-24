# %% [markdown]
# # RUNTIME: local workstation GPU (NVIDIA RTX 3060 12 GB), `.venv-local` (deviation D4)
#
# Project: `qpain/paper-01` · Stage: `02 train` · Estimated time: 3-5 h on the RTX 3060 (to be measured)
#
# Trains the six planned classifiers (analysis_plan.md section 4) with one fixed protocol, three seeds each (18 runs),
# on the frozen split from notebook 01 (SHA-256 asserted). Saves the best-validation checkpoint of every run (not
# published), per-epoch logs, validation/test/OCTDL predictions and the eligibility table (validation only).
# Test predictions are saved for notebook 04; no decision in this notebook uses test data.
#
# Run: `PAPER_ROOT=<project> python -m jupyter nbconvert --to notebook --execute notebooks/02_train_GPU.ipynb ...`
# Smoke test (plan rule: 1% of the data first): set `SMOKE=1` (1% per class and partition, at most 5 OCTDL images per
# class, 1 epoch, seed 11, results to checkpoints/smoke/, nothing to outputs/). `DEVICE=cpu` forces the CPU.
# Source of this notebook: `notebooks/src/02_train_GPU.py` (edit the source, then rebuild with scripts/py-to-ipynb.py).
# Revised after Codex review 003 (results/003-review-notebook-02.md; decisions/005-notebook-02-review.md).

# %%
# Setup: paths, parameters fixed by analysis_plan.md section 4 (and clarifications D6), determinism
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # needed for deterministic cuBLAS; set before CUDA starts
import sys, io, json, time, hashlib, zipfile, random, platform, datetime, subprocess
import importlib.metadata as ilmd
import numpy as np, pandas as pd, requests
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import timm

ROOT = os.environ.get("PAPER_ROOT", "/content/paper-01")
RAW, OUT, CKPT = (os.path.join(ROOT, d) for d in ("data/raw", "outputs", "checkpoints"))
DERIVED = os.path.join(ROOT, "data", "derived")
SMOKE = os.environ.get("SMOKE", "0") == "1"
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
NOTEBOOK = "02_train_GPU"
RUN_DIR = os.path.join(CKPT, "smoke" if SMOKE else "runs")
for d in (OUT, CKPT, DERIVED, RUN_DIR, os.path.join(CKPT, "pretrained")):
    os.makedirs(d, exist_ok=True)

SPLIT_SHA256 = "655b1785339d81d629ec8794e7698a6ada55161994a7fe02fab20a9073a506a6"          # frozen in analysis_plan.md D3e
OCTDL_INVENTORY_SHA256 = "357e26a6a263009ac160d161d0661bb956251b405505751fe3ec0e32348197cb" # frozen here (D7)
RAW_SHA256 = {  # publisher hashes verified by notebook 01 (outputs/raw_data_manifest.csv)
    "ZhangLabData.zip": "acbc42869440eec719361942d0ba560f132742230262f22e78bfb93eb5ea4563",
    "OCTDL.zip": "637a72ff3319645aac30408d4d838189337ac0b8b0f9faabf7c8156d271f46f7",
}
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
MODELS = [  # (timm name, family, plan name, Hub repo, pinned commit, SHA-256 of model.safetensors); section 4, D6g
    ("efficientnet_b0", "convolutional", "EfficientNet-B0", "timm/efficientnet_b0.ra_in1k",
     "1b5383e5f79cc0f7fc067e372f8f26a5fa73f26a", "d569899762ea9b1384ee07f4af64805cf8caa1c55f9253ebb1080dc40e87a2cd"),
    ("mobilenetv3_large_100", "convolutional", "MobileNetV3-Large", "timm/mobilenetv3_large_100.ra_in1k",
     "96f46a1c52932f27492dff66c72378eb99b443a7", "f425af34cc1cead2b5d6211f789a1f30b94835dc32f9c0fcc5a916e4fd2dde85"),
    ("mobilevit_s", "hybrid", "MobileViT-S", "timm/mobilevit_s.cvnets_in1k",
     "b531fe564ef88515b1aa1743ad4c40452b9f80ee", "87f5585cb61814a5e18920d556772ed9c3af6c9ad72b6cf798c654021c306f81"),
    ("efficientformer_l1", "hybrid", "EfficientFormer-L1", "timm/efficientformer_l1.snap_dist_in1k",
     "8738a58190800aaf5995f453b05f643df453a565", "dab9ad0e328fe94a9c155497dc3f46cb456ae0223e19184d377e03adc3e3e08e"),
    ("deit_tiny_patch16_224", "transformer", "DeiT-Tiny", "timm/deit_tiny_patch16_224.fb_in1k",
     "80e968688553f219e4a86f940ed945a23709c16f", "21d4764d94f6c3ffdb6da3581115a0a1ee2d505537d96883b540e54766407c9e"),
    ("deit_small_patch16_224", "transformer", "DeiT-Small", "timm/deit_small_patch16_224.fb_in1k",
     "91327a9c99f98fe6b524cd4d397b7226b80e1365", "1e747b4a8d0df2cfbd3c450e8c97685d867448ab0c2ddbfb34b6885f5cb23e5b"),
]
SEEDS = [11, 22, 33]
IMG = 224
EPOCHS, BATCH, LR, WD = 10, 64, 3e-4, 0.05
EVAL_BATCH = 64             # Codex review 003, point 3: evaluation batches no larger than training batches
CROP_MIN = 0.875            # D6c: "small random crop" = square crop of 87.5-100% of the side, resized back to 224
ELIGIBLE_MIN_VAL_BACC = 0.85
CACHE_VERSION = "2"         # bump when the image cache code changes
MAX_GPU_SHARE = 0.90        # memory probe: stop if any model needs more than 90% of the GPU's memory
SMOKE_FRACTION, SMOKE_SEED = 0.01, 20261119
if SMOKE:
    SEEDS, EPOCHS = [11], 1

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
VERSIONS = {m: ilmd.version(m) for m in ("torch", "torchvision", "timm", "numpy", "pandas", "Pillow", "safetensors", "requests")}
ENV = {"python": platform.python_version(), "platform": platform.platform(), "device": DEVICE, "packages": VERSIONS,
       "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
       "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None, "smoke": SMOKE}
if DEVICE == "cuda":
    ENV["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    ENV["gpu_total_mib"] = torch.cuda.get_device_properties(0).total_memory // 2**20
print(json.dumps(ENV, indent=1))

# %%
# Helpers: atomic writes, manifest-recorded outputs (same format as notebook 01), hashing, metrics
def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def atomic(path, write):
    """write(tmp_path) then rename: an interrupted write never leaves a partial file under the final name."""
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
    head = "| file | notebook | cell | description |\n|---|---|---|---|\n"
    rows = [l for l in (open(man).read() if os.path.exists(man) else head).splitlines() if not l.startswith(f"| {filename} |")]
    rows.append(f"| {filename} | {NOTEBOOK} | {cell} | {description} |")
    def w(t):
        with open(t, "w", newline="\n") as f:
            f.write("\n".join(rows) + "\n")
    atomic(man, w)
    print("saved", filename)
    return path

def metrics(y, p):
    """Balanced accuracy (mean recall over classes present in y) and macro F1 over the four classes."""
    pred = p.argmax(1)
    rec, f1 = [], []
    for c in range(len(CLASSES)):
        tp = int(((pred == c) & (y == c)).sum()); fn = int(((pred != c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum())
        if tp + fn:
            rec.append(tp / (tp + fn))
        f1.append(2 * tp / (2 * tp + fn + fp) if (2 * tp + fn + fp) else 0.0)
    return float(np.mean(rec)), float(np.mean(f1))

_y = np.array([0, 0, 1, 2, 3, 3]); _p = np.eye(4)[[0, 1, 1, 2, 3, 0]]
assert np.isclose(metrics(_y, _p)[0], (0.5 + 1 + 1 + 0.5) / 4)

# %%
# Frozen inputs: split and OCTDL inventory hashes, raw archive hashes; then the images this notebook uses
split_path, octdl_path = os.path.join(OUT, "kermany_split.csv"), os.path.join(OUT, "octdl_inventory.csv")
assert sha256_file(split_path) == SPLIT_SHA256, "kermany_split.csv is not the frozen split (analysis_plan.md D3e)"
assert sha256_file(octdl_path) == OCTDL_INVENTORY_SHA256, "octdl_inventory.csv is not the frozen inventory (D7)"
t0 = time.time()
for f, h in RAW_SHA256.items():
    assert sha256_file(os.path.join(RAW, f)) == h, f"{f} does not match the publisher hash verified by notebook 01"
print(f"raw archives verified in {time.time() - t0:.0f} s")
ker = pd.read_csv(split_path, dtype={"patient_id": str, "group": str})
assert len(ker) == 109309 and set(ker["class"]) == set(CLASSES)
octdl = pd.read_csv(octdl_path, dtype={"patient_id": str})
octdl_test = octdl[octdl["in_exploratory_test"]].sort_values("member").reset_index(drop=True)
assert set(octdl_test["class"]) == {"DME", "NORMAL"}

sets = {"train": ker[ker["in_train_capped"]], "val": ker[ker["partition"] == "val"], "test": ker[ker["partition"] == "test"]}
sets = {k: v.sort_values("member").reset_index(drop=True) for k, v in sets.items()}
assert set(sets["train"]["partition"]) == {"train"}
if SMOKE:   # 1% per class in each partition, seeded; at least 2 images per class
    def _per_class(d, n):   # no groupby().apply: pandas >= 2.2 drops the grouping column there
        return pd.concat([g.sample(n(len(g)), random_state=SMOKE_SEED) for _, g in d.groupby("class")]).sort_values("member").reset_index(drop=True)
    sets = {k: _per_class(v, lambda m: max(2, int(round(m * SMOKE_FRACTION)))) for k, v in sets.items()}
    octdl_test = _per_class(octdl_test, lambda m: min(m, 5))
for k, v in sets.items():
    print(k, len(v), v["class"].value_counts().reindex(CLASSES).to_dict())
print("octdl_test", len(octdl_test), octdl_test["class"].value_counts().to_dict())

# Class weights: inverse class frequency of the capped training set, normalised to mean 1 (plan section 4)
freq = sets["train"]["class"].value_counts().reindex(CLASSES).to_numpy(dtype=float)
CLASS_WEIGHTS = (1.0 / freq) / (1.0 / freq).mean()
print("class weights", dict(zip(CLASSES, CLASS_WEIGHTS.round(4))))

# %%
# Image cache: every used image decoded once from the verified raw zips, greyscale, square-resized to 224 (bicubic),
# uint8 (D6a). Stored in data/derived/ (not in git, never published). The fingerprint binds the archive hash, the
# member list and the preprocessing; the cache file's own hash is checked on reuse.
def build_cache(name, zip_name, members):
    path, fp_path = os.path.join(DERIVED, f"{name}.npy"), os.path.join(DERIVED, f"{name}.fingerprint.json")
    fp = hashlib.sha256(json.dumps([CACHE_VERSION, zip_name, RAW_SHA256[zip_name], list(members), IMG, "L", "bicubic"]).encode()).hexdigest()
    if os.path.exists(path) and os.path.exists(fp_path):
        try:
            meta = json.load(open(fp_path))
        except (ValueError, OSError):
            meta = {}
        if meta.get("fingerprint") == fp and meta.get("npy_sha256") == sha256_file(path):
            arr = np.load(path)
            assert arr.dtype == np.uint8 and arr.shape == (len(members), IMG, IMG)
            print("cache reused (hash verified):", name, arr.shape)
            return arr, meta
    t0 = time.time()
    arr = np.empty((len(members), IMG, IMG), dtype=np.uint8)
    with zipfile.ZipFile(os.path.join(RAW, zip_name)) as z:
        for i, m in enumerate(members):
            with Image.open(io.BytesIO(z.read(m))) as im:
                arr[i] = np.asarray(im.convert("L").resize((IMG, IMG), Image.BICUBIC))
            if (i + 1) % 10000 == 0:
                print(f"  {name}: {i + 1}/{len(members)} images, {time.time() - t0:.0f} s")
    def w(t):
        with open(t, "wb") as f:
            np.save(f, arr)
    atomic(path, w)
    meta = {"fingerprint": fp, "npy_sha256": sha256_file(path), "shape": list(arr.shape)}
    write_json(fp_path, meta)
    print(f"cache built: {name} {arr.shape} in {time.time() - t0:.0f} s")
    return arr, meta

tag = "smoke_" if SMOKE else ""
images, CACHE_META = {}, {}
for k, v in sets.items():
    images[k], CACHE_META[k] = build_cache(f"{tag}kermany_{k}", "ZhangLabData.zip", v["member"].tolist())
images["octdl"], CACHE_META["octdl"] = build_cache(f"{tag}octdl_test", "OCTDL.zip", octdl_test["member"].tolist())
labels = {k: torch.tensor(v["class"].map(CLASSES.index).to_numpy()) for k, v in sets.items()}
labels["octdl"] = torch.tensor(octdl_test["class"].map(CLASSES.index).to_numpy())

# %%
# Pretrained weights: downloaded with requests from the timm repositories on the Hugging Face Hub at the pinned
# commits above (the Hub client fails on this PC's certificate setup), checked against the pinned SHA-256, loaded from file.
def pretrained_file(name, repo, commit, sha):
    path = os.path.join(CKPT, "pretrained", f"{name}.safetensors")
    if not (os.path.exists(path) and sha256_file(path) == sha):
        r = requests.get(f"https://huggingface.co/{repo}/resolve/{commit}/model.safetensors", timeout=600)
        r.raise_for_status()
        def w(t):
            with open(t, "wb") as f:
                f.write(r.content)
        atomic(path, w)
    assert sha256_file(path) == sha, f"{name}: weight file does not match the pinned SHA-256"
    cfg = timm.create_model(name, pretrained=False).pretrained_cfg
    assert repo.endswith("." + cfg["tag"]), f"{name}: timm's default tag {cfg['tag']} differs from the pinned repository {repo}"
    return path, {"model": name, "repo": repo, "commit": commit, "sha256": sha, "mean": list(cfg["mean"]), "std": list(cfg["std"])}

WEIGHTS = {name: pretrained_file(name, repo, commit, sha) for name, _, _, repo, commit, sha in MODELS}
for name in WEIGHTS:
    print(name, WEIGHTS[name][1]["repo"], WEIGHTS[name][1]["commit"][:10], WEIGHTS[name][1]["sha256"][:12])

def make_model(name, seed):
    torch.manual_seed(seed)                     # the new 4-class head(s) are initialised from the run's seed
    path, meta = WEIGHTS[name]
    m = timm.create_model(name, pretrained=True, num_classes=len(CLASSES), pretrained_cfg_overlay=dict(file=path))
    return m, torch.tensor(meta["mean"]).view(1, 3, 1, 1), torch.tensor(meta["std"]).view(1, 3, 1, 1)

# Run signature: a finished run is reused only if all of these are unchanged (Codex review 003, point 1)
CODE_SHA = sha256_file(os.path.join(ROOT, "notebooks", "src", "02_train_GPU.py"))
CONFIG = {"img": IMG, "epochs": EPOCHS, "batch": BATCH, "eval_batch": EVAL_BATCH, "lr": LR, "wd": WD, "crop_min": CROP_MIN,
          "smoke": SMOKE, "split_sha256": SPLIT_SHA256, "octdl_inventory_sha256": OCTDL_INVENTORY_SHA256, "raw": RAW_SHA256,
          "caches": {k: v["npy_sha256"] for k, v in CACHE_META.items()}, "weights": {n: m[1]["sha256"] for n, m in WEIGHTS.items()},
          "class_weights": CLASS_WEIGHTS.round(8).tolist(), "precision": "bf16 autocast" if DEVICE == "cuda" else "fp32",
          "device": DEVICE, "packages": VERSIONS, "code_sha256": CODE_SHA}
CONFIG_SHA = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
print("run signature", CONFIG_SHA[:16], "| code", CODE_SHA[:16])

# %%
# Training and evaluation (plan section 4; clarifications D6)
def to_input(u8, mean, std):
    """uint8 (N, 224, 224) on the device -> normalised float (N, 3, 224, 224) with the model's pretrained mean/std (D6b)."""
    x = u8.float().div_(255).unsqueeze(1).expand(-1, 3, -1, -1)
    return (x - mean) / std

def augment(x, gen):
    """Per image: square crop of CROP_MIN-100% of the side at a random position, resized back (bilinear), and a
    horizontal flip with probability 0.5 (D6c). Done on the device with one affine grid per batch."""
    n = x.shape[0]
    s = CROP_MIN + (1 - CROP_MIN) * torch.rand(n, generator=gen, device=x.device)
    tx = (torch.rand(n, generator=gen, device=x.device) * 2 - 1) * (1 - s)
    ty = (torch.rand(n, generator=gen, device=x.device) * 2 - 1) * (1 - s)
    flip = torch.where(torch.rand(n, generator=gen, device=x.device) < 0.5, -1.0, 1.0)
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 0, 2], theta[:, 1, 1], theta[:, 1, 2] = s * flip, tx, s, ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

@torch.no_grad()
def predict(model, u8_cpu, mean, std, bs=EVAL_BATCH):
    model.eval()
    out = []
    for i in range(0, len(u8_cpu), bs):
        xb = to_input(torch.from_numpy(np.ascontiguousarray(u8_cpu[i:i + bs])).to(DEVICE), mean, std)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            out.append(model(xb).float().softmax(1).cpu())
    p = torch.cat(out).numpy()
    assert np.isfinite(p).all(), "non-finite probabilities"
    return p

def finished(run):
    """A run counts as finished only if its marker is readable, its signature matches and both artefacts match their hashes."""
    done_path = os.path.join(RUN_DIR, run + ".json")
    try:
        done = json.load(open(done_path))
    except (OSError, ValueError):
        return None
    for key in ("weights", "preds"):
        f = os.path.join(RUN_DIR, run + (".pt" if key == "weights" else "_preds.npz"))
        if not os.path.exists(f) or sha256_file(f) != done.get(f"{key}_sha256"):
            return None
    return done if done.get("config_sha") == CONFIG_SHA else None

def run_one(name, family, seed):
    run = f"{name}_s{seed}"
    done = finished(run)
    if done:
        print("resume: finished run found and verified, skipped:", run)
        return done
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model, mean, std = make_model(name, seed)
    model.to(DEVICE); mean, std = mean.to(DEVICE), std.to(DEVICE)
    xtr = torch.from_numpy(images["train"]).to(DEVICE)          # uint8 on the device (about 1.8 GB for the full set)
    ytr = labels["train"].to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps_per_epoch = (len(xtr) + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)
    wts = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=DEVICE)
    lossf = nn.CrossEntropyLoss(weight=wts)
    g_cpu = torch.Generator().manual_seed(seed)
    g_dev = torch.Generator(device=DEVICE).manual_seed(seed)
    weights_path = os.path.join(RUN_DIR, run + ".pt")
    best, best_val, log = None, None, []
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_run = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train(); t0 = time.time(); loss_sum, w_sum = 0.0, 0.0
        perm = torch.randperm(len(xtr), generator=g_cpu).to(DEVICE)
        for b in range(steps_per_epoch):
            idx = perm[b * BATCH:(b + 1) * BATCH]
            xb = augment(to_input(xtr[idx], mean, std), g_dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                loss = lossf(model(xb).float(), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            bw = float(wts[ytr[idx]].sum())          # weighted-mean loss x its weight total = the batch's weighted loss sum
            loss_sum += float(loss) * bw; w_sum += bw
        train_loss = loss_sum / w_sum
        assert np.isfinite(train_loss), f"{run}: training loss is not finite in epoch {epoch}"
        pv = predict(model, images["val"], mean, std)
        bacc, f1 = metrics(labels["val"].numpy(), pv)
        row = {"model": name, "family": family, "seed": seed, "epoch": epoch, "train_loss_weighted": train_loss, "val_balanced_accuracy": bacc,
               "val_macro_f1": f1, "lr_end": sched.get_last_lr()[0], "seconds": time.time() - t0}
        log.append(row); print(f"  {run} epoch {epoch}: loss {train_loss:.4f} | val bacc {bacc:.4f} | f1 {f1:.4f} | {row['seconds']:.0f} s")
        if best is None or bacc > best["val_balanced_accuracy"]:     # ties keep the earlier epoch (D6f)
            best, best_val = dict(row), pv
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            atomic(weights_path, lambda t: torch.save(state, t))
    peak = torch.cuda.max_memory_allocated() // 2**20 if DEVICE == "cuda" else None
    del xtr, opt
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True), strict=True)
    preds = {k: predict(model, images[k], mean, std) for k in ("val", "test", "octdl")}
    # Reload check (Codex review 003, point 5): the reloaded checkpoint must reproduce the best epoch's validation probabilities
    diff = float(np.abs(preds["val"] - best_val).max())
    assert diff <= 1e-3, f"{run}: reloaded checkpoint differs from the best epoch by {diff:.2e}"
    preds_path = os.path.join(RUN_DIR, run + "_preds.npz")
    def w(t):
        with open(t, "wb") as f:
            np.savez_compressed(f, **preds)
    atomic(preds_path, w)
    done = {"run": run, "model": name, "family": family, "seed": seed, "best_epoch": best["epoch"],
            "val_balanced_accuracy": best["val_balanced_accuracy"], "val_macro_f1": best["val_macro_f1"], "reload_max_abs_diff": diff,
            "params": sum(p.numel() for p in model.parameters()), "train_seconds": time.time() - t_run, "peak_gpu_mib": peak,
            # what notebook 03 needs to rebuild the model exactly
            "classes": CLASSES, "img": IMG, "mean": WEIGHTS[name][1]["mean"], "std": WEIGHTS[name][1]["std"],
            "pretrained_repo": WEIGHTS[name][1]["repo"], "pretrained_commit": WEIGHTS[name][1]["commit"],
            "checkpoint_format": "plain state_dict of timm.create_model(name, num_classes=4); load with strict=True",
            "weights_sha256": sha256_file(weights_path), "preds_sha256": sha256_file(preds_path), "config_sha": CONFIG_SHA, "log": log}
    write_json(os.path.join(RUN_DIR, run + ".json"), done)          # the completion marker is written last
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return done

# %%
# GPU memory probe (Codex review 003, point 3): each model, with the full training set resident, two training steps
# and validation-sized evaluation batches. Stops before any full run if a model needs more than 90% of the GPU.
if DEVICE == "cuda" and not SMOKE:
    xtr = torch.from_numpy(images["train"]).to(DEVICE)
    total = torch.cuda.get_device_properties(0).total_memory
    probe = {}
    for name, family, *_ in MODELS:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model, mean, std = make_model(name, 0); model.to(DEVICE); mean, std = mean.to(DEVICE), std.to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        g = torch.Generator(device=DEVICE).manual_seed(0)
        for _ in range(2):
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(augment(to_input(xtr[:BATCH], mean, std), g)).float(), labels["train"][:BATCH].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        predict(model, images["val"][:4 * EVAL_BATCH], mean, std)
        probe[name] = torch.cuda.max_memory_allocated() / total
        del model, opt
        print(f"memory probe {name}: peak {probe[name]:.1%} of {total / 2**30:.1f} GiB")
    del xtr; torch.cuda.empty_cache()
    assert max(probe.values()) <= MAX_GPU_SHARE, f"a model needs more than {MAX_GPU_SHARE:.0%} of GPU memory: {probe}"

# %%
# The 18 runs, one model at a time (a stop loses at most the run in progress; verified finished runs are skipped on rerun)
t_all = time.time()
results = []
for name, family, *_ in MODELS:
    for seed in SEEDS:
        print(f"=== {name} seed {seed} ({len(results) + 1}/{len(MODELS) * len(SEEDS)}) ===")
        results.append(run_one(name, family, seed))
print(f"all runs done in {(time.time() - t_all) / 60:.1f} min")

# %%
# Assemble outputs: logs, run summary, eligibility (validation only), predictions, checkpoint hashes, environment
log = pd.DataFrame([r for d in results for r in d["log"]])
summary = pd.DataFrame([{k: v for k, v in d.items() if k not in ("log", "classes", "mean", "std")} for d in results])
elig = (summary.groupby(["model", "family"], sort=False)["val_balanced_accuracy"].agg(["min", "mean", "std"]).reset_index()
        .rename(columns={"min": "val_bacc_min_over_seeds", "mean": "val_bacc_mean", "std": "val_bacc_sd"}))
elig["eligible"] = elig["val_bacc_min_over_seeds"] >= ELIGIBLE_MIN_VAL_BACC
print(summary[["run", "best_epoch", "val_balanced_accuracy", "val_macro_f1", "train_seconds", "peak_gpu_mib"]].to_string(index=False))
print(elig.to_string(index=False))

def pred_frame(split, df):
    rows = []
    for d in results:
        p = np.load(os.path.join(RUN_DIR, d["run"] + "_preds.npz"))[split]
        assert len(p) == len(df) and np.isfinite(p).all()
        f = pd.DataFrame(p.astype(np.float64), columns=[f"p_{c}" for c in CLASSES])     # full float32 precision
        f.insert(0, "seed", d["seed"]); f.insert(0, "model", d["model"]); f.insert(0, "member", df["member"].to_numpy())
        f["pred_class"] = np.array(CLASSES)[p.argmax(1)]                                    # argmax before any rounding
        f["true_class"] = df["class"].to_numpy()
        rows.append(f)
    return pd.concat(rows, ignore_index=True)

pv, pt, po = pred_frame("val", sets["val"]), pred_frame("test", sets["test"]), pred_frame("octdl", octdl_test)
weights_tab = pd.DataFrame([{"run": d["run"], "weights_sha256": d["weights_sha256"], "preds_sha256": d["preds_sha256"], "best_epoch": d["best_epoch"]} for d in results])
pre_tab = pd.DataFrame([WEIGHTS[n][1] for n, *_ in MODELS])

if SMOKE:
    print("SMOKE TEST OK:", len(results), "runs; nothing written to outputs/. Rows:", len(pv), len(pt), len(po))
else:
    save_output(log, "train_log.csv", "Per-epoch class-weighted training loss, validation balanced accuracy and macro F1, learning rate and time for all 18 runs", "assemble")
    save_output(summary, "train_runs.csv", "One row per run: best epoch (by validation balanced accuracy), validation scores, reload check, parameters, training time, peak GPU memory, checkpoint and prediction hashes", "assemble")
    save_output(elig, "model_eligibility.csv", "Eligibility for the faithfulness comparison: validation balanced accuracy >= 0.85 on every seed (plan section 4; validation only)", "assemble")
    save_output(pv, "predictions_val.csv.gz", "Validation-set class probabilities (full precision) and predicted class, every run (best checkpoint)", "assemble")
    save_output(pt, "predictions_test.csv.gz", "Test-set class probabilities (full precision) and predicted class, every run (best checkpoint); used only in notebook 04", "assemble")
    save_output(po, "predictions_octdl.csv.gz", "OCTDL exploratory test (DME, NORMAL) class probabilities and predicted class, every run (exploratory, plan section 8)", "assemble")
    save_output(pre_tab, "pretrained_weights.csv", "Pretrained weight files: Hugging Face repository, pinned commit, SHA-256, normalisation constants", "assemble")
    save_output(weights_tab, "model_checkpoints.csv", "SHA-256 of each run's best checkpoint and prediction archive (kept in checkpoints/runs/, not published)", "assemble")
    save_output({**ENV, "config": CONFIG, "config_sha": CONFIG_SHA, "class_weights": dict(zip(CLASSES, CLASS_WEIGHTS.tolist())),
                 "memory_probe_peak_share": probe if DEVICE == "cuda" else None,
                 "n_images": {k: int(len(v)) for k, v in sets.items()} | {"octdl": int(len(octdl_test))},
                 "total_minutes": (time.time() - t_all) / 60, "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
                "train_environment.json", "Training environment, run signature (configuration, input, weight, code and package fingerprints), class weights, memory probe, image counts and total time", "assemble")
