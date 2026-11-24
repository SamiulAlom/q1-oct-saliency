# %% [markdown]
# # RUNTIME: local workstation CPU, `.venv-local` (deviation D4)
#
# Project: `qpain/paper-01` · Stage: `05 post hoc exploratory (D10)` · Estimated time: under 5 minutes
#
# POST HOC, proposed after the registered results were seen (analysis_plan.md section 10, D10). Not confirmatory: no
# p-values. Edge-adjusted faithfulness gap = method gap minus the model-independent edge-map gap on the same image,
# model, seed and fill, averaged over seeds per image. Reported per model, per family (with leave-one-model-out and
# model-pair breakdowns of each family contrast) and per class, with pointwise 95% cluster-bootstrap percentile
# intervals from the same resampling matrix as notebook 04 (seed 20261119, 10,000 resamples). The bootstrap functions
# are taken unchanged from notebook 04's source, whose hash is recorded.
#
# Run: `PAPER_ROOT=<project> python -m jupyter nbconvert --to notebook --execute notebooks/05_posthoc_edge_adjusted_CPU.ipynb ...`
# Smoke test: `SMOKE=1` reads checkpoints/attr_smoke/ and writes to checkpoints/posthoc_smoke/ only (200 resamples).
# Source of this notebook: `notebooks/src/05_posthoc_edge_adjusted_CPU.py` (rebuild with scripts/py-to-ipynb.py).
# Revised after Codex review 006 (results/006-review-notebook-05.md; decisions/008-posthoc-and-step5-review.md).

# %%
# Setup; the bootstrap functions come from notebook 04's source, unchanged
import os, json, glob, hashlib, datetime, platform
import importlib.metadata as ilmd
import numpy as np, pandas as pd
from scipy.stats import norm as _norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("PAPER_ROOT", "/content/paper-01")
OUT, CKPT = os.path.join(ROOT, "outputs"), os.path.join(ROOT, "checkpoints")
SMOKE = os.environ.get("SMOKE", "0") == "1"
NOTEBOOK = "05_posthoc_edge_adjusted_CPU"
DEST = os.path.join(CKPT, "posthoc_smoke") if SMOKE else OUT
os.makedirs(DEST, exist_ok=True)
SEED, B = 20261119, (200 if SMOKE else 10_000)
SEEDS = [11] if SMOKE else [11, 22, 33]
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
FAMILIES = ["convolutional", "hybrid", "transformer"]
METHODS = ["gradcam", "ig", "occlusion"]
FAMILY_PAIRS = [("convolutional", "hybrid"), ("convolutional", "transformer"), ("hybrid", "transformer")]
CLASS_PAIRS = [("CNV", "DME"), ("CNV", "DRUSEN"), ("DME", "DRUSEN")]
LABEL = {"efficientnet_b0": "EfficientNet-B0", "mobilenetv3_large_100": "MobileNetV3-L", "mobilevit_s": "MobileViT-S",
         "efficientformer_l1": "EfficientFormer-L1", "deit_tiny_patch16_224": "DeiT-Tiny", "deit_small_patch16_224": "DeiT-Small"}
METHOD_LABEL = {"gradcam": "Grad-CAM", "ig": "Integrated Gradients", "occlusion": "Occlusion"}
OKABE_ITO = {"convolutional": "#0072B2", "hybrid": "#E69F00", "transformer": "#009E73"}
MARKERS = {"convolutional": "o", "hybrid": "s", "transformer": "^"}
TAG = "POST HOC (D10), exploratory, no tests"

def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

NB04_SRC = os.path.join(ROOT, "notebooks", "src", "04_statistics_figures_CPU.py")
import ast
_WANT = {"resampler", "gsums", "n_groups", "mean_stat", "diff_stat", "summarise", "holm"}
_tree = ast.parse(open(NB04_SRC, encoding="utf-8").read())
_defs = [n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name in _WANT]   # function definitions only, unchanged
assert {n.name for n in _defs} == _WANT, "notebook 04 no longer defines every bootstrap function"
_g = {"np": np, "B": B, "_norm": _norm}
exec(compile(ast.Module(body=_defs, type_ignores=[]), NB04_SRC, "exec"), _g)
resampler, gsums, n_groups, mean_stat, diff_stat, summarise = (_g[k] for k in ("resampler", "gsums", "n_groups", "mean_stat", "diff_stat", "summarise"))
VERSIONS = {m: ilmd.version(m) for m in ("numpy", "pandas", "scipy", "matplotlib")}
print("SMOKE" if SMOKE else "FULL", "| B =", B, "| notebook 04 source", sha256_file(NB04_SRC)[:16], "|", VERSIONS)

def save(obj, filename, description, cell=""):
    path = os.path.join(DEST, filename); tmp = path + ".tmp"
    if hasattr(obj, "to_csv"):
        obj.to_csv(tmp, index=False, lineterminator="\n")
    elif hasattr(obj, "savefig"):
        obj.savefig(tmp, format=filename.rsplit(".", 1)[1], dpi=600)
    else:
        with open(tmp, "w", newline="\n") as f:
            json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)
    if not SMOKE:
        man = os.path.join(OUT, "MANIFEST.md")
        rows = [l for l in open(man).read().splitlines() if not l.startswith(f"| {filename} |")]
        rows.append(f"| {filename} | {NOTEBOOK} | {cell} | {description} |")
        with open(man + ".tmp", "w", newline="\n") as f:
            f.write("\n".join(rows) + "\n")
        os.replace(man + ".tmp", man)
    print("saved", filename)

# %%
# Inputs: notebook 03's scores (Kermany attribution list); edge-adjusted gap per image, model, seed and fill
if SMOKE:
    scores = pd.concat([pd.read_csv(f, dtype={"patient_id": str, "group": str}) for f in sorted(glob.glob(os.path.join(CKPT, "attr_smoke", "*_scores.csv.gz")))], ignore_index=True)
    INPUT_HASHES = {}
else:
    scores = pd.read_csv(os.path.join(OUT, "faithfulness_scores.csv.gz"), dtype={"patient_id": str, "group": str})
    INPUT_HASHES = {f: sha256_file(os.path.join(OUT, f)) for f in ("faithfulness_scores.csv.gz", "model_eligibility.csv", "h1_family_contrasts.csv",
                                                                    "family_method_means.csv", "gradcam_validity.csv", "kermany_split.csv")}
    elig = pd.read_csv(os.path.join(OUT, "model_eligibility.csv"))
    assert set(scores["model"]) == set(elig.loc[elig["eligible"], "model"])
    # Same inputs and NumPy as the registered statistics run (Codex review 006, point 2)
    reg = json.load(open(os.path.join(OUT, "statistics_environment.json")))
    assert reg["input_sha256"]["faithfulness_scores.csv.gz"] == INPUT_HASHES["faithfulness_scores.csv.gz"], "scores differ from notebook 04's input"
    assert reg["packages"]["numpy"] == VERSIONS["numpy"], "NumPy differs from notebook 04's run; resampling could differ"
    assert reg["seed"] == SEED and reg["resamples"] == B
s = scores[scores["dataset"] == "kermany"]
assert not s.duplicated(["member", "model", "seed", "method", "fill"]).any()
assert np.isfinite(s["faithfulness_gap"]).all()
# Exact coverage (Codex review 006, point 1): every frozen attribution image x model x seed x applicable method x fill
if SMOKE:
    MEMBERS, GC_UNAVAILABLE = set(s["member"]), set()
else:
    _split = pd.read_csv(os.path.join(OUT, "kermany_split.csv"), usecols=["member", "class", "group", "in_attribution_test"], dtype={"group": str})
    _att = _split[_split["in_attribution_test"]]
    MEMBERS = set(_att["member"])
    _gv = pd.read_csv(os.path.join(OUT, "gradcam_validity.csv"))
    GC_UNAVAILABLE = {mo for mo in s["model"].unique() if not _gv[_gv["model"] == mo].groupby("layer")["valid"].all().any()}
    _ref = _att.set_index("member")[["class", "group"]]
    _chk = s.drop_duplicates("member").set_index("member")[["class", "group"]].sort_index()
    assert _chk.equals(_ref.loc[_chk.index]), "class or group differs from the frozen split"
for mo, d in s.groupby("model"):
    want = {"ig", "occlusion", "edge", "random"} | ({"gradcam"} if mo not in GC_UNAVAILABLE else set()) | ({"rollout"} if mo.startswith("deit") else set())
    assert set(d["method"]) == want and set(d["fill"]) == {"blur", "mean"} and set(d["seed"]) == set(SEEDS), f"{mo}: methods, fills or seeds differ"
    assert set(d["member"]) == MEMBERS and len(d) == len(MEMBERS) * len(SEEDS) * len(want) * 2, f"{mo}: incomplete coverage"
    assert d.groupby("member")["family"].nunique().max() == 1 and d["family"].nunique() == 1
key = ["member", "model", "seed", "fill"]
edge = s[s["method"] == "edge"][key + ["faithfulness_gap"]].rename(columns={"faithfulness_gap": "edge_gap"})
m = s[s["method"].isin(METHODS + ["rollout"])].merge(edge, on=key, how="left", validate="many_to_one")
assert m["edge_gap"].notna().all(), "an image, model, seed or fill lacks its edge-map control"
m["edge_adjusted_gap"] = m["faithfulness_gap"] - m["edge_gap"]
ikey = ["member", "class", "group", "model", "family", "method", "fill"]
assert (m.groupby(ikey)["seed"].nunique() == len(SEEDS)).all(), "an image lacks a seed"
img = m.groupby(ikey, as_index=False)[["edge_adjusted_gap", "faithfulness_gap", "edge_gap"]].mean()     # seeds averaged per image
edge_img = s[s["method"] == "edge"].groupby(["member", "class", "group", "model", "family", "fill"], as_index=False)["faithfulness_gap"].mean()
FAMILY_OF = s.drop_duplicates("model").set_index("model")["family"].to_dict()
GMAP, C = resampler(s["group"], SEED)                # notebook 04: resampler(img.loc[kermany, "group"], SEED); same group set (asserted below)
print("images", s["member"].nunique(), "| groups", len(GMAP), "| models", sorted(FAMILY_OF))

def wide(method, fill, value="edge_adjusted_gap"):
    d = img[(img["method"] == method) & (img["fill"] == fill)]
    w = d.pivot(index=["member", "class", "group"], columns="model", values=value).reset_index()
    assert not w.drop(columns=["member", "class", "group"]).isna().any().any()
    return w

def interval(groups, v):
    S, N = gsums(groups, v, GMAP)
    r = summarise(*mean_stat(S, N, C), test=False)
    return {"n_images": int(N.sum()), "n_groups": n_groups(groups, v), **r}

if not SMOKE:   # the resampling matrix must reproduce notebook 04's published interval exactly (Codex review 006, point 2)
    _w = wide("gradcam", "blur", value="faithfulness_gap")
    _conv = _w[[mo for mo in _w.columns if FAMILY_OF.get(mo) == "convolutional"]].mean(axis=1).to_numpy()
    _r = interval(_w["group"], _conv)
    _pub = pd.read_csv(os.path.join(OUT, "family_method_means.csv")).query("method == 'gradcam' and family == 'convolutional'").iloc[0]
    assert np.allclose([_r["estimate"], _r["ci95_low"], _r["ci95_high"]], [_pub["estimate"], _pub["ci95_low"], _pub["ci95_high"]], rtol=0, atol=1e-12), \
        f"resampling differs from notebook 04: {_r} vs {_pub.to_dict()}"
    print("resampling matrix reproduces notebook 04's published interval exactly")

# %%
# (a) per model and method; (b) per family and family contrasts; (c) model pairs and leave-one-model-out
rows_model, rows_family, rows_contrast, rows_pairs, rows_loo = [], [], [], [], []
for fill in ("blur", "mean"):
    for method in METHODS + ["rollout"]:
        if not ((img["method"] == method) & (img["fill"] == fill)).any():
            continue
        w = wide(method, fill)
        models = [c for c in w.columns if c in FAMILY_OF]
        for mo in models:
            rows_model.append({"fill": fill, "method": method, "model": mo, "family": FAMILY_OF[mo], **interval(w["group"], w[mo].to_numpy())})
        if method == "rollout":
            continue
        fam = {f: [mo for mo in models if FAMILY_OF[mo] == f] for f in FAMILIES}
        fval = {f: w[ms].mean(axis=1).to_numpy() for f, ms in fam.items() if ms}
        for f, v in fval.items():
            rows_family.append({"fill": fill, "method": method, "family": f, "models": "+".join(fam[f]), **interval(w["group"], v)})
        for a, b in FAMILY_PAIRS:
            if a in fval and b in fval:
                rows_contrast.append({"fill": fill, "method": method, "contrast": f"{a} - {b}", **interval(w["group"], fval[a] - fval[b])})
                for ma in fam[a]:
                    for mb in fam[b]:
                        rows_pairs.append({"fill": fill, "method": method, "contrast": f"{a} - {b}", "model_a": ma, "model_b": mb,
                                           **interval(w["group"], w[ma].to_numpy() - w[mb].to_numpy())})
        for drop in models:                           # leave one model out: its family is represented by the other model
            fam_l = {f: [x for x in ms if x != drop] for f, ms in fam.items()}
            fv = {f: w[ms].mean(axis=1).to_numpy() for f, ms in fam_l.items() if ms}
            for a, b in FAMILY_PAIRS:
                if FAMILY_OF[drop] in (a, b) and a in fv and b in fv:
                    rows_loo.append({"fill": fill, "method": method, "contrast": f"{a} - {b}", "left_out": drop, **interval(w["group"], fv[a] - fv[b])})

# (d) classes, pooled over models: edge-adjusted gap, the edge map's own gap, and H2-style contrasts (intervals only)
rows_class, rows_class_contrast = [], []
for fill, method in [(f, mm) for f in ("blur", "mean") for mm in METHODS + ["edge"]]:
    if method == "edge":
        e = edge_img[edge_img["fill"] == fill].pivot(index=["member", "class", "group"], columns="model", values="faithfulness_gap").reset_index()
        assert not e.isna().any().any()
        w, label = e, "edge map gap (descriptive control)"
    else:
        w, label = wide(method, fill), "edge-adjusted gap"
    models = [c for c in w.columns if c in FAMILY_OF]
    pooled = w[models].mean(axis=1).to_numpy()
    for c in CLASSES:
        msk = (w["class"] == c).to_numpy()
        if msk.any():
            rows_class.append({"fill": fill, "method": method, "outcome": label, "class": c, **interval(w["group"], np.where(msk, pooled, np.nan))})
    for a, b in CLASS_PAIRS:
        ma, mb = (w["class"] == a).to_numpy(), (w["class"] == b).to_numpy()
        if ma.any() and mb.any():
            SA, NA = gsums(w["group"], np.where(ma, pooled, np.nan), GMAP)
            SB, NB = gsums(w["group"], np.where(mb, pooled, np.nan), GMAP)
            rows_class_contrast.append({"fill": fill, "method": method, "outcome": label, "contrast": f"{a} - {b}", "n_a": int(ma.sum()), "n_b": int(mb.sum()),
                                        **summarise(*diff_stat(SA, NA, SB, NB, C), test=False)})

per_model, per_family, contrasts = pd.DataFrame(rows_model), pd.DataFrame(rows_family), pd.DataFrame(rows_contrast)
pairs, loo, per_class, class_contrasts = pd.DataFrame(rows_pairs), pd.DataFrame(rows_loo), pd.DataFrame(rows_class), pd.DataFrame(rows_class_contrast)
for df in (per_model, per_family, contrasts, pairs, loo, per_class, class_contrasts):
    df.insert(0, "analysis", TAG)
pd.set_option("display.width", 220)
print("=== " + TAG + " ===")
print(per_model[per_model["fill"] == "blur"][["method", "model", "estimate", "ci95_low", "ci95_high"]].round(3).to_string(index=False))
print(contrasts[contrasts["fill"] == "blur"][["method", "contrast", "estimate", "ci95_low", "ci95_high"]].round(3).to_string(index=False))
print(loo[loo["fill"] == "blur"][["method", "contrast", "left_out", "estimate", "ci95_low", "ci95_high"]].round(3).to_string(index=False))
print(per_class[per_class["fill"] == "blur"][["method", "outcome", "class", "estimate", "ci95_low", "ci95_high"]].round(3).to_string(index=False))

# %%
# Save tables before the figure; figure: per-model edge-adjusted gap by method (blurred fill)
CI = "pointwise 95% percentile interval from 10,000 cluster-bootstrap resamples of the 680 groups of 820 images (same resamples as notebook 04); zero = parity with the edge-map control on this perturbation metric"
for fn, df, desc in (("posthoc_edge_adjusted_by_model.csv", per_model, "edge-adjusted gap (method minus edge map) per model, method and fill"),
                     ("posthoc_edge_adjusted_by_family.csv", per_family, "edge-adjusted gap per family and method"),
                     ("posthoc_edge_adjusted_family_contrasts.csv", contrasts, "family contrasts of the edge-adjusted gap"),
                     ("posthoc_edge_adjusted_model_pairs.csv", pairs, "each family contrast split into its model-pair differences"),
                     ("posthoc_edge_adjusted_leave_one_out.csv", loo, "family contrasts recomputed leaving out each model in turn"),
                     ("posthoc_class_edge.csv", per_class, "per class, pooled over models: edge-adjusted gap and the edge map's own gap"),
                     ("posthoc_class_contrasts_edge_adjusted.csv", class_contrasts, "class contrasts (CNV, DME, DRUSEN) of the edge-adjusted gap and of the edge map's gap")):
    save(df, fn, f"{TAG}: {desc}, {CI}", "save")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.5, 2.4), layout="constrained")
pm = per_model[(per_model["fill"] == "blur") & per_model["method"].isin(METHODS)]
order = [mo for f in FAMILIES for mo in LABEL if FAMILY_OF.get(mo) == f]
for i, method in enumerate(METHODS):
    for j, mo in enumerate(order):
        d = pm[(pm["method"] == method) & (pm["model"] == mo)]
        if d.empty:
            continue
        x = i + (j - (len(order) - 1) / 2) * 0.12
        f = FAMILY_OF[mo]
        ax.vlines(x, d["ci95_low"], d["ci95_high"], color=OKABE_ITO[f], lw=0.8)
        ax.plot(x, d["estimate"], MARKERS[f], color=OKABE_ITO[f], ms=3.5, mfc="white" if j % 2 else OKABE_ITO[f],
                label=LABEL[mo] if i == 0 else None)
ax.axhline(0, color="0.4", lw=0.6, ls="--")
ax.set_xticks(range(len(METHODS)), [METHOD_LABEL[mm] for mm in METHODS])
ax.set_ylabel("Gap minus edge-map gap (post hoc;\nmean, pointwise 95% bootstrap interval)")
fig.legend(loc="outside upper center", ncol=3, frameon=False, fontsize=6)
for ext in ("pdf", "png"):
    save(fig, f"fig_posthoc_edge_adjusted.{ext}", f"{TAG}: edge-adjusted gap per model and method (blurred fill), mean and {CI}; values in posthoc_edge_adjusted_by_model.csv", "figure")
plt.close(fig)
save({"python": platform.python_version(), "packages": VERSIONS, "seed": SEED, "resamples": B, "smoke": SMOKE, "plan_deviation": "D10",
      "notebook04_source_sha256": sha256_file(NB04_SRC), "input_sha256": INPUT_HASHES,
      "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
     "posthoc_environment.json", f"{TAG}: run record (seed, resamples, input hashes, notebook 04 source hash)", "save")
print("SMOKE TEST OK" if SMOKE else "post hoc analysis done")
