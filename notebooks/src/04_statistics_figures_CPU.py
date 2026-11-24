# %% [markdown]
# # RUNTIME: local workstation CPU, `.venv-local` (deviation D4)
#
# Project: `qpain/paper-01` · Stage: `04 statistics and figures` · Estimated time: under 10 minutes
#
# Implements analysis_plan.md section 7 (and the exploratory analyses of section 8) on the outputs of notebooks 02
# and 03: seed-averaged faithfulness gaps per image, one cluster bootstrap over groups (10,000 resamples, seed
# 20261119) shared by every estimate, studentised (bootstrap-t) tests with Holm correction for H1 (9 family contrasts)
# and H2 (9 class contrasts), pointwise 95% percentile intervals, seed spread, minimum detectable differences,
# sensitivity analyses (all-correct subset, mean fill), controls, OCTDL, test accuracy, and the result figures.
#
# Run: `PAPER_ROOT=<project> python -m jupyter nbconvert --to notebook --execute notebooks/04_statistics_figures_CPU.ipynb ...`
# Smoke test: `SMOKE=1` reads the smoke attribution files of notebook 03 (checkpoints/attr_smoke/, seed 11 only, a few
# images), uses 200 resamples and writes only to checkpoints/stats_smoke/ (nothing to outputs/).
# Source of this notebook: `notebooks/src/04_statistics_figures_CPU.py` (edit the source, then rebuild with scripts/py-to-ipynb.py).
# Revised after Codex review 005 (results/005-review-notebook-04.md; decisions/007-notebook-04-review.md).
# 2026-11-21: two input assertions added after Codex's step-5 check (D11); rerun confirmed unchanged result tables.

# %%
# Setup
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
NOTEBOOK = "04_statistics_figures_CPU"
DEST = os.path.join(CKPT, "stats_smoke") if SMOKE else OUT
os.makedirs(DEST, exist_ok=True)

SEED, B = 20261119, (200 if SMOKE else 10_000)       # plan section 7
ALPHA = 0.05
SEEDS = [11] if SMOKE else [11, 22, 33]
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
FAMILIES = ["convolutional", "hybrid", "transformer"]
PRIMARY_METHODS = ["gradcam", "ig", "occlusion"]          # the family comparison (section 7); edge, random, rollout are controls/extras
FAMILY_PAIRS = [("convolutional", "hybrid"), ("convolutional", "transformer"), ("hybrid", "transformer")]
CLASS_PAIRS = [("CNV", "DME"), ("CNV", "DRUSEN"), ("DME", "DRUSEN")]   # H2; NORMAL descriptive only
MODEL_LABEL = {"efficientnet_b0": "EfficientNet-B0", "mobilenetv3_large_100": "MobileNetV3-L", "mobilevit_s": "MobileViT-S",
               "efficientformer_l1": "EfficientFormer-L1", "deit_tiny_patch16_224": "DeiT-Tiny", "deit_small_patch16_224": "DeiT-Small"}
METHOD_LABEL = {"gradcam": "Grad-CAM", "ig": "Integrated Gradients", "occlusion": "Occlusion", "edge": "Edge map (control)",
                "rollout": "Attention rollout", "random": "Random (control)"}
OKABE_ITO = {"convolutional": "#0072B2", "hybrid": "#E69F00", "transformer": "#009E73"}
MARKERS = {"convolutional": "o", "hybrid": "s", "transformer": "^"}
VERSIONS = {m: ilmd.version(m) for m in ("numpy", "pandas", "scipy", "matplotlib")}
print("SMOKE" if SMOKE else "FULL", "| B =", B, "|", VERSIONS)

def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def save(obj, filename, description, cell=""):
    """Write to outputs/ with a manifest row (full run) or to checkpoints/stats_smoke/ (smoke, no manifest)."""
    path = os.path.join(DEST, filename)
    tmp = path + ".tmp"
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
# Inputs: faithfulness scores (notebook 03), eligibility and test predictions (notebook 02)
if SMOKE:
    files = sorted(glob.glob(os.path.join(CKPT, "attr_smoke", "*_scores.csv.gz")))
    scores = pd.concat([pd.read_csv(f, dtype={"patient_id": str, "group": str}) for f in files], ignore_index=True)
    eligible = sorted(scores["model"].unique())
    test_pred = None
else:
    scores = pd.read_csv(os.path.join(OUT, "faithfulness_scores.csv.gz"), dtype={"patient_id": str, "group": str})
    elig = pd.read_csv(os.path.join(OUT, "model_eligibility.csv"))
    eligible = elig.loc[elig["eligible"], "model"].tolist()
    test_pred = pd.read_csv(os.path.join(OUT, "predictions_test.csv.gz"))
    INPUT_HASHES = {f: sha256_file(os.path.join(OUT, f)) for f in ("faithfulness_scores.csv.gz", "faithfulness_curves.csv.gz", "model_eligibility.csv",
                                                                    "predictions_test.csv.gz", "predictions_octdl.csv.gz", "kermany_split.csv",
                                                                    "octdl_inventory.csv", "attribution_files.csv", "gradcam_validity.csv")}
assert set(scores["model"]) <= set(eligible), "scores contain a model that is not eligible"
assert np.isfinite(scores[["deletion_auc", "insertion_auc", "faithfulness_gap"]]).all().all()
FAMILY_OF = scores.drop_duplicates("model").set_index("model")["family"].to_dict()
print("eligible models:", eligible, "| families:", FAMILY_OF)
print(scores.groupby(["dataset", "fill", "method"]).size().to_string())

# Seeds are replicates of a model: per image, average each outcome over the seeds (section 7), after checking that
# every image has every seed
# Exact coverage (Codex review 005, point 3): every image x eligible model x seed x applicable method x fill exactly once.
# Grad-CAM may be missing only for a whole model whose layers all failed the validity check (notebook 03).
assert not scores.duplicated(["dataset", "member", "model", "seed", "method", "fill"]).any(), "duplicate score rows"
assert set(scores["seed"]) == set(SEEDS), f"unexpected seeds: {sorted(set(scores['seed']))}"
if SMOKE:
    MEMBERS = {d: set(g["member"]) for d, g in scores.groupby("dataset")}
    GC_UNAVAILABLE = set()
else:
    # frozen inputs (D3e, D7) and the class and group of every score row bound to them (Codex step-5 check, point 4; D11)
    assert INPUT_HASHES["kermany_split.csv"] == "655b1785339d81d629ec8794e7698a6ada55161994a7fe02fab20a9073a506a6", "not the frozen split"
    assert INPUT_HASHES["octdl_inventory.csv"] == "357e26a6a263009ac160d161d0661bb956251b405505751fe3ec0e32348197cb", "not the frozen OCTDL inventory"
    _split = pd.read_csv(os.path.join(OUT, "kermany_split.csv"), usecols=["member", "class", "group", "in_attribution_test"], dtype={"group": str})
    _oct = pd.read_csv(os.path.join(OUT, "octdl_inventory.csv"), usecols=["member", "class", "group", "in_exploratory_test"], dtype={"group": str})
    _ref = pd.concat([_split.assign(dataset="kermany"), _oct.assign(dataset="octdl", group=lambda d: "octdl_" + d["group"])])[["dataset", "member", "class", "group"]]
    _rows = scores.drop_duplicates(["dataset", "member"])[["dataset", "member", "class", "group"]]
    _j = _rows.merge(_ref, on=["dataset", "member"], how="left", suffixes=("", "_frozen"), validate="one_to_one")
    assert (_j["class"] == _j["class_frozen"]).all() and (_j["group"] == _j["group_frozen"]).all(), "class or group of a score row differs from the frozen inventory"
    MEMBERS = {"kermany": set(_split.loc[_split["in_attribution_test"], "member"]), "octdl": set(_oct.loc[_oct["in_exploratory_test"], "member"])}
    _gv = pd.read_csv(os.path.join(OUT, "gradcam_validity.csv"))
    GC_UNAVAILABLE = {m for m in eligible if not _gv[(_gv["model"] == m)].groupby("layer")["valid"].all().any()}
    assert set(scores["model"]) == set(eligible), "scores do not cover every eligible model"
FILLS = {"kermany": {"blur", "mean"}, "octdl": {"blur"}}
for (ds, m), d in scores.groupby(["dataset", "model"]):
    want = {"ig", "occlusion", "edge", "random"} | ({"gradcam"} if m not in GC_UNAVAILABLE else set()) | ({"rollout"} if m.startswith("deit") else set())
    assert set(d["method"]) == want, f"{ds} {m}: methods {sorted(set(d['method']))}, expected {sorted(want)}"
    assert set(d["fill"]) == FILLS[ds], f"{ds} {m}: fills {sorted(set(d['fill']))}"
    assert len(d) == len(MEMBERS[ds]) * len(SEEDS) * len(want) * len(FILLS[ds]), f"{ds} {m}: incomplete coverage"
    assert set(d["member"]) == MEMBERS[ds], f"{ds} {m}: members differ from the frozen lists"
_pc = scores.groupby(["dataset", "member", "model", "seed"])["pred_class_original"].nunique()
assert (_pc == 1).all() and scores["pred_class_original"].notna().all(), "inconsistent or missing pred_class_original"
print("coverage checks passed | Grad-CAM unavailable for:", sorted(GC_UNAVAILABLE))
key = ["dataset", "member", "class", "group", "model", "family", "method", "fill"]
img = scores.groupby(key, as_index=False)[["deletion_auc", "insertion_auc", "faithfulness_gap"]].mean()

# %%
# One cluster bootstrap over groups, shared by every estimate (section 7). Kermany and OCTDL have separate groups and
# are resampled separately (OCTDL is exploratory); the Kermany resampling matrix is the same for every analysis.
def resampler(groups, seed):
    G = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    C = rng.multinomial(len(G), np.full(len(G), 1 / len(G)), size=B).astype(np.float64)   # (B, G) copies of each group
    return {g: i for i, g in enumerate(G)}, C

KER_GROUPS, C_KER = resampler(img.loc[img["dataset"] == "kermany", "group"], SEED)
OCT_GROUPS, C_OCT = resampler(img.loc[img["dataset"] == "octdl", "group"], SEED + 1) if (img["dataset"] == "octdl").any() else ({}, None)
print("Kermany groups:", len(KER_GROUPS), "| OCTDL groups:", len(OCT_GROUPS), "| resamples:", B)

def n_groups(groups, values):
    return int(len(set(np.asarray(groups)[~np.isnan(values)])))

def gsums(groups, values, gmap):
    """Per-group sums of values and counts (values with NaN rows dropped)."""
    ok = ~np.isnan(values)
    gi = np.array([gmap[g] for g in np.asarray(groups)[ok]], dtype=int)
    G = len(gmap)
    return np.bincount(gi, weights=values[ok], minlength=G), np.bincount(gi, minlength=G).astype(float)

def mean_stat(S, N, C):
    """Weighted mean over groups and its cluster-robust SE, observed (all groups once) and per resample."""
    th = S.sum() / N.sum()
    se = np.sqrt(((S - th * N) ** 2).sum()) / N.sum()
    CS, CN = C @ S, C @ N
    thb = CS / CN
    ssq = C @ (S ** 2) - 2 * thb * (C @ (S * N)) + thb ** 2 * (C @ (N ** 2))
    seb = np.sqrt(np.maximum(ssq, 0)) / CN
    return th, se, thb, seb

def diff_stat(SA, NA, SB, NB, C):
    """Difference of two weighted means (A minus B) with a cluster-robust (linearised) SE, observed and per resample."""
    def parts(SA, NA, SB, NB, c):
        nA, nB = c @ NA, c @ NB
        mA, mB = (c @ SA) / nA, (c @ SB) / nB
        # sum over (resampled) groups of the squared influence a/nA - b/nB, with a = SA - mA NA, b = SB - mB NB
        aa = c @ SA ** 2 - 2 * mA * (c @ (SA * NA)) + mA ** 2 * (c @ NA ** 2)
        bb = c @ SB ** 2 - 2 * mB * (c @ (SB * NB)) + mB ** 2 * (c @ NB ** 2)
        ab = c @ (SA * SB) - mB * (c @ (SA * NB)) - mA * (c @ (NA * SB)) + mA * mB * (c @ (NA * NB))
        var = aa / nA ** 2 - 2 * ab / (nA * nB) + bb / nB ** 2
        return mA - mB, np.sqrt(np.maximum(var, 0))
    th, se = parts(SA, NA, SB, NB, np.ones((1, len(SA))))
    thb, seb = parts(SA, NA, SB, NB, C)
    return float(th[0]), float(se[0]), thb, seb

def summarise(th, se, thb, seb, test=True):
    """Pointwise 95% percentile interval; for tests, the centred studentised bootstrap p-value of 'difference = 0'.
    Pre-declared handling (Codex review 005, point 1): an observed SE of 0 or a non-finite estimate makes inference
    'undefined' (no interval, no p-value). A resample in which the statistic or its SE is undefined (e.g. a class absent
    from the resampled groups) is counted as invalid and left out of both the interval and the test; the count is reported.
    With all resamples valid the p-value is (1 + #{|t*| >= |t|}) / (1 + B), as in D9."""
    thb, seb = np.asarray(thb, float), np.asarray(seb, float)
    out = {"estimate": th, "se_cluster": se}
    if not (np.isfinite(th) and np.isfinite(se) and se > 0):
        out.update({"inference": "undefined (observed SE is 0 or not finite)", "ci95_low": np.nan, "ci95_high": np.nan, "n_resamples_invalid": int(len(thb))})
        if test:
            out.update({"t": np.nan, "p_bootstrap_t": np.nan})
        return out
    valid = np.isfinite(thb) & np.logical_or(not test, np.isfinite(seb) & (seb > 0))
    lo, hi = np.percentile(thb[valid], [2.5, 97.5])
    out.update({"inference": "defined", "ci95_low": lo, "ci95_high": hi, "n_resamples_invalid": int((~valid).sum())})
    if test:
        t_obs = th / se
        tb = (thb[valid] - th) / seb[valid]
        out.update({"t": t_obs, "p_bootstrap_t": (1 + np.sum(np.abs(tb) >= abs(t_obs))) / (1 + len(tb))})
    return out

def holm(p):
    p = np.asarray(p, dtype=float); m = len(p)
    order = np.argsort(p)
    adj = np.empty(m); running = 0.0
    for r, i in enumerate(order):
        running = max(running, (m - r) * p[i])
        adj[i] = min(1.0, running)
    return adj

assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

# %%
# Per-image values for a method and fill: each model's seed-averaged gap; a family's value is the mean of its
# eligible models' values for that method (section 7); the pooled value (H2) is the mean over eligible models.
def wide(dataset, method, fill, outcome="faithfulness_gap", members=None):
    d = img[(img["dataset"] == dataset) & (img["method"] == method) & (img["fill"] == fill)]
    if members is not None:
        d = d[d["member"].isin(members)]
    w = d.pivot(index=["member", "class", "group"], columns="model", values=outcome).reset_index()   # raises on duplicates
    assert not w.drop(columns=["member", "class", "group"]).isna().any().any(), f"{dataset} {method} {fill}: a model lacks some images"
    return w

def family_values(w):
    fam = {}
    for f in FAMILIES:
        cols = [m for m in w.columns if FAMILY_OF.get(m) == f]
        fam[f] = w[cols].mean(axis=1).to_numpy() if cols else np.full(len(w), np.nan)
    return fam, {f: [m for m in w.columns if FAMILY_OF.get(m) == f] for f in FAMILIES}

def family_contrasts(dataset, fill, members=None, test=True, tag=""):
    rows = []
    gmap, C = (KER_GROUPS, C_KER) if dataset == "kermany" else (OCT_GROUPS, C_OCT)
    for method in PRIMARY_METHODS:
        w = wide(dataset, method, fill, members=members)
        fam, used = family_values(w)
        for a, b in FAMILY_PAIRS:
            row = {"analysis": tag, "dataset": dataset, "fill": fill, "method": method, "contrast": f"{a} - {b}",
                   "models_a": "+".join(used[a]), "models_b": "+".join(used[b]), "n_images": int(len(w))}
            if not used[a] or not used[b]:
                row["status"] = "not estimable (no eligible model with this method in a family)"
            else:
                S, N = gsums(w["group"], fam[a] - fam[b], gmap)
                row.update(summarise(*mean_stat(S, N, C), test=test)); row["status"] = "estimated"
                row["n_groups"] = n_groups(w["group"], fam[a] - fam[b])
            rows.append(row)
    return pd.DataFrame(rows)

def class_contrasts(dataset, fill, members=None, test=True, tag=""):
    rows = []
    gmap, C = (KER_GROUPS, C_KER) if dataset == "kermany" else (OCT_GROUPS, C_OCT)
    for method in PRIMARY_METHODS:
        w = wide(dataset, method, fill, members=members)
        models = [m for m in w.columns if m in FAMILY_OF]
        pooled = w[models].mean(axis=1).to_numpy()
        for a, b in CLASS_PAIRS:
            ma, mb = (w["class"] == a).to_numpy(), (w["class"] == b).to_numpy()
            row = {"analysis": tag, "dataset": dataset, "fill": fill, "method": method, "contrast": f"{a} - {b}",
                   "n_a": int(ma.sum()), "n_b": int(mb.sum()), "models": "+".join(models)}
            if ma.sum() == 0 or mb.sum() == 0 or not models:
                row["status"] = "not estimable"
            else:
                SA, NA = gsums(w["group"], np.where(ma, pooled, np.nan), gmap)
                SB, NB = gsums(w["group"], np.where(mb, pooled, np.nan), gmap)
                row.update(summarise(*diff_stat(SA, NA, SB, NB, C), test=test)); row["status"] = "estimated"
                row["n_groups_a"], row["n_groups_b"] = n_groups(w["group"], np.where(ma, pooled, np.nan)), n_groups(w["group"], np.where(mb, pooled, np.nan))
            rows.append(row)
    return pd.DataFrame(rows)

def add_holm(df):
    """Holm over the registered nine slots (Codex review 005, point 2): an unavailable or undefined test enters as p = 1,
    and its adjusted p-value and decision are reported as missing."""
    df = df.copy()
    if "p_bootstrap_t" not in df:
        df["p_bootstrap_t"] = np.nan
    avail = (df["status"] == "estimated") & df["p_bootstrap_t"].notna()
    adj = holm(np.where(avail, df["p_bootstrap_t"], 1.0))
    df["p_holm"] = np.where(avail, adj, np.nan)
    df["difference_at_0.05_holm"] = np.where(avail, df["p_holm"] < ALPHA, None)
    df["holm_family_size"] = len(df)
    return df

# %%
# Primary (H1) and secondary (H2) analyses: Kermany attribution list, blurred fill (section 7)
h1 = add_holm(family_contrasts("kermany", "blur", tag="H1 primary"))
h2 = add_holm(class_contrasts("kermany", "blur", tag="H2 secondary"))
# Minimum detectable difference for each contrast (section 7: what the final sample could detect): two-sided
# alpha 0.05 / 9 (the strictest Holm step) and 80% power, from the observed cluster-robust SE
z = _norm.ppf(1 - ALPHA / 2 / 9) + _norm.ppf(0.80)
for df in (h1, h2):
    if "se_cluster" in df:
        df["min_detectable_difference_80pct"] = z * df["se_cluster"]
print(h1.drop(columns=["models_a", "models_b"], errors="ignore").round(4).to_string(index=False))
print(h2.drop(columns=["models"], errors="ignore").round(4).to_string(index=False))

# Seed spread (section 7): SD over seeds of each model's mean gap; a family difference smaller than the largest seed SD
# of the models involved is described as not robust
seed_means = (scores[(scores["dataset"] == "kermany") & (scores["fill"] == "blur")]
              .groupby(["model", "family", "method", "seed"])["faithfulness_gap"].mean().reset_index())
seed_sd = seed_means.groupby(["model", "family", "method"])["faithfulness_gap"].agg(["mean", "std"]).reset_index().rename(
    columns={"mean": "mean_gap_over_seeds", "std": "sd_over_seeds"})
def robust(row):
    if row["status"] != "estimated":
        return np.nan
    models = row["models_a"].split("+") + row["models_b"].split("+")
    sd = seed_sd[(seed_sd["method"] == row["method"]) & (seed_sd["model"].isin(models))]["sd_over_seeds"].max()
    return bool(abs(row["estimate"]) > sd) if np.isfinite(sd) else np.nan
h1["larger_than_max_seed_sd"] = h1.apply(robust, axis=1)

# %%
# Descriptive estimates with pointwise percentile intervals from the same resamples: per family, model, class
def mean_rows(dataset, fill, by, methods, members=None, outcome="faithfulness_gap"):
    gmap, C = (KER_GROUPS, C_KER) if dataset == "kermany" else (OCT_GROUPS, C_OCT)
    rows = []
    for method in methods:
        w = wide(dataset, method, fill, members=members, outcome=outcome)
        if w.empty:
            continue
        models = [m for m in w.columns if m in FAMILY_OF]
        if by == "model":
            units = {m: w[m].to_numpy() for m in models}
        elif by == "family":
            units = {f: v for f, v in family_values(w)[0].items() if not np.isnan(v).all()}
        else:   # class, pooled over models
            pooled = w[models].mean(axis=1).to_numpy()
            units = {c: np.where(w["class"] == c, pooled, np.nan) for c in CLASSES if (w["class"] == c).any()}
        for u, v in units.items():
            S, N = gsums(w["group"], v, gmap)
            if N.sum() == 0:
                continue
            r = summarise(*mean_stat(S, N, C), test=False)
            rows.append({"dataset": dataset, "fill": fill, "method": method, "outcome": outcome, by: u, "n_images": int(N.sum()),
                         "n_groups": n_groups(w["group"], v), **r})
    return pd.DataFrame(rows)

ALL_METHODS = sorted(img["method"].unique())
fam_means = mean_rows("kermany", "blur", "family", PRIMARY_METHODS)
model_means = pd.concat([mean_rows("kermany", f, "model", ALL_METHODS, outcome=o) for f in ("blur", "mean")
                         for o in ("faithfulness_gap", "deletion_auc", "insertion_auc")], ignore_index=True)
fam_auc = pd.concat([mean_rows("kermany", "blur", "family", PRIMARY_METHODS + ["edge", "random"], outcome=o)
                     for o in ("deletion_auc", "insertion_auc")], ignore_index=True)
class_means = mean_rows("kermany", "blur", "class", PRIMARY_METHODS)

# %%
# Sensitivity analyses (labelled as such): images classified correctly by every eligible model and seed; mean fill
correct = (scores[(scores["dataset"] == "kermany")].drop_duplicates(["member", "model", "seed"])
           .assign(ok=lambda d: d["pred_class_original"] == d["class"]).groupby("member")["ok"].all())
correct_members = correct[correct].index
n_correct = scores[scores["member"].isin(correct_members)].drop_duplicates("member")["class"].value_counts().reindex(CLASSES, fill_value=0)
print("all-correct subset per class:", n_correct.to_dict())
sens_correct = add_holm(family_contrasts("kermany", "blur", members=correct_members, tag="sensitivity: all-correct subset"))
sens_fill = family_contrasts("kermany", "mean", test=False, tag="sensitivity: mean fill (exploratory, no test)")

# Exploratory: OCTDL scanner shift (blurred fill), family means and contrasts with intervals only
octdl_rows = pd.DataFrame()
if C_OCT is not None:
    # matched-label comparator (Codex review 005, point 4): Kermany DME and NORMAL images only; a dataset comparison, not a test
    ker_dn = img.loc[(img["dataset"] == "kermany") & img["class"].isin(["DME", "NORMAL"]), "member"].unique()
    octdl_rows = pd.concat([mean_rows("octdl", "blur", "family", PRIMARY_METHODS).assign(kind="family mean, OCTDL DME+NORMAL"),
                            mean_rows("kermany", "blur", "family", PRIMARY_METHODS, members=ker_dn).assign(kind="family mean, Kermany DME+NORMAL (comparator)"),
                            family_contrasts("octdl", "blur", test=False, tag="exploratory: OCTDL").assign(kind="family contrast, OCTDL")], ignore_index=True)

# Accuracy on the Kermany test partition and on OCTDL (per model: mean and SD over seeds); not used for any decision
def acc_table(pred, label):
    rows = []
    for (m, s), d in pred.groupby(["model", "seed"]):
        y = d["true_class"].to_numpy(); p = d["pred_class"].to_numpy()       # four-way predictions kept as they are
        present = [c for c in CLASSES if (y == c).any()]
        rec = [np.mean(p[y == c] == c) for c in present]
        f1 = []
        for c in present:                                                     # macro F1 over the labels present in the set
            tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
            f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        rows.append({"set": label, "model": m, "seed": s, "balanced_accuracy": np.mean(rec), "macro_f1_4class": np.mean(f1),
                     "accuracy": np.mean(p == y), "n": len(y), "label_set": "+".join(present),
                     "share_predicted_outside_label_set": float(np.mean(~np.isin(p, present)))})
    t = pd.DataFrame(rows)
    return t.groupby(["set", "model"]).agg(balanced_accuracy_mean=("balanced_accuracy", "mean"), balanced_accuracy_sd=("balanced_accuracy", "std"),
                                            macro_f1_mean=("macro_f1_4class", "mean"), macro_f1_sd=("macro_f1_4class", "std"),
                                            accuracy_mean=("accuracy", "mean"), n=("n", "first"), label_set=("label_set", "first"),
                                            share_predicted_outside_label_set=("share_predicted_outside_label_set", "mean")).reset_index().rename(
                                            columns={"macro_f1_mean": "macro_f1_over_label_set_mean", "macro_f1_sd": "macro_f1_over_label_set_sd"})
acc = pd.DataFrame()
if test_pred is not None:
    acc = pd.concat([acc_table(test_pred, "kermany_test"), acc_table(pd.read_csv(os.path.join(OUT, "predictions_octdl.csv.gz")), "octdl_dme_normal")], ignore_index=True)
    print(acc.round(4).to_string(index=False))

# IG completeness diagnostic (from notebook 03): median relative residual |sum(attr) - logit difference| / |logit difference|
igc = scores[(scores["method"] == "ig") & (scores["fill"] == "blur")].copy()
igc["rel_residual"] = (igc["ig_attribution_sum"] - igc["ig_logit_difference"]).abs() / igc["ig_logit_difference"].abs().clip(lower=1e-6)
ig_check = igc.groupby(["dataset", "model"])["rel_residual"].describe(percentiles=[0.5, 0.9])[["count", "50%", "90%"]].reset_index()

# %%
# Figures (IEEE column width 3.5 in; vector PDF and 600 dpi PNG); every value from the tables above
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})

def fig_family(fm):
    fig, ax = plt.subplots(figsize=(3.5, 2.2), layout="constrained")
    for j, f in enumerate(FAMILIES):
        d = fm[fm["family"] == f].set_index("method").reindex(PRIMARY_METHODS)
        x = np.arange(len(PRIMARY_METHODS)) + (j - 1) * 0.22
        ax.vlines(x, d["ci95_low"], d["ci95_high"], color=OKABE_ITO[f], lw=0.8)       # interval endpoints drawn directly
        ax.plot(x, d["estimate"], MARKERS[f], color=OKABE_ITO[f], ms=4, label=f.capitalize())
    ax.axhline(0, color="0.4", lw=0.6, ls="--")
    ax.set_xticks(range(len(PRIMARY_METHODS)), [METHOD_LABEL[m] for m in PRIMARY_METHODS])
    ax.set_ylabel("Faithfulness gap (mean, pointwise\n95% cluster-bootstrap interval)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    return fig

def fig_class(cm):
    fig, ax = plt.subplots(figsize=(3.5, 2.2), layout="constrained")
    cols = {"CNV": "#CC79A7", "DME": "#56B4E9", "DRUSEN": "#D55E00", "NORMAL": "#999999"}
    mk = {"CNV": "o", "DME": "s", "DRUSEN": "^", "NORMAL": "D"}
    present = [c for c in CLASSES if c in set(cm["class"])]
    for j, c in enumerate(present):
        d = cm[cm["class"] == c].set_index("method").reindex(PRIMARY_METHODS)
        x = np.arange(len(PRIMARY_METHODS)) + (j - (len(present) - 1) / 2) * 0.18
        ax.vlines(x, d["ci95_low"], d["ci95_high"], color=cols[c], lw=0.8)
        ax.plot(x, d["estimate"], mk[c], color=cols[c], ms=4, label=c + (" (descriptive)" if c == "NORMAL" else ""))
    ax.axhline(0, color="0.4", lw=0.6, ls="--")
    ax.set_xticks(range(len(PRIMARY_METHODS)), [METHOD_LABEL[m] for m in PRIMARY_METHODS])
    ax.set_ylabel("Gap pooled over models (mean,\npointwise 95% cluster-bootstrap interval)")
    ax.legend(frameon=False, ncol=2, fontsize=6)
    return fig

def fig_curves():
    cv_path = os.path.join(CKPT, "attr_smoke") if SMOKE else None
    if SMOKE:
        cv = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(cv_path, "*_curves.csv.gz"))], ignore_index=True)
    else:
        cv = pd.read_csv(os.path.join(OUT, "faithfulness_curves.csv.gz"))
    cv = cv[(cv["dataset"] == "kermany") & (cv["fill"] == "blur")]
    cv["family"] = cv["model"].map(FAMILY_OF)
    ks = [int(c[4:]) for c in cv.columns if c.startswith("del_")]
    # per family and method, the random-order curve comes from the same models (Codex review 005, point 6); means saved
    means = []
    for method in PRIMARY_METHODS:
        for f in FAMILIES:
            d = cv[(cv["method"] == method) & (cv["family"] == f)]
            if not len(d):
                continue
            r = cv[(cv["method"] == "random") & cv["model"].isin(d["model"].unique())]
            for kind in ("del", "ins"):
                for src, dd in (("method", d), ("random_matched", r)):
                    means.append({"method": method, "family": f, "curve": kind, "order": src, "models": "+".join(sorted(d["model"].unique())),
                                  "n_rows": len(dd), **{f"k{k}": dd[f"{kind}_{k}"].mean() for k in ks}})
    means = pd.DataFrame(means)
    save(means, "curve_means.csv", "Mean deletion and insertion curves per family and method, and the random-order curve of the same models (values plotted in fig_curves)", "figures")
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.6), layout="constrained", sharex=True, sharey="row")
    for col, method in enumerate(PRIMARY_METHODS):
        for row, kind in enumerate(("del", "ins")):
            ax = axes[row, col]
            for f in FAMILIES:
                for src, ls, lab in (("method", "-", f.capitalize()), ("random_matched", ":", None)):
                    d = means[(means["method"] == method) & (means["family"] == f) & (means["curve"] == kind) & (means["order"] == src)]
                    if len(d):
                        ax.plot(np.array(ks) / ks[-1], d[[f"k{k}" for k in ks]].to_numpy()[0], color=OKABE_ITO[f], ls=ls, lw=0.9,
                                marker=MARKERS[f] if src == "method" else None, ms=2.5, label=lab)
            if row == 0:
                ax.set_title(METHOD_LABEL[method], fontsize=7)
            if col == 0:
                ax.set_ylabel(("Deletion" if kind == "del" else "Insertion") + ": true-class\nprobability (mean)")
            if row == 1:
                ax.set_xlabel("Fraction of cells perturbed")
    axes[0, 0].plot([], [], color="0.4", ls=":", lw=0.9, label="Random order, same models")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=4, frameon=False, fontsize=6)    # above the panels, never over data
    return fig

# Tables are saved before any figure is drawn (Codex review 005, point 5)
CI = "pointwise 95% cluster-bootstrap percentile interval over groups (n images and groups in each row)"
tables = {"h1_family_contrasts.csv": (h1, "H1 (primary): 9 family contrasts of the faithfulness gap, bootstrap-t p-values, Holm over 9 slots, " + CI + ", minimum detectable difference, robustness to seed spread"),
          "h2_class_contrasts.csv": (h2, "H2 (secondary): 9 class contrasts (CNV, DME, DRUSEN) of the pooled faithfulness gap, Holm over 9 slots as a second family, " + CI),
          "family_method_means.csv": (fam_means, "Mean faithfulness gap per family (its eligible models named in h1) and method, Kermany, blurred fill, " + CI),
          "family_method_auc_means.csv": (fam_auc, "Mean deletion and insertion AUC per family and method, including edge and random controls, Kermany, blurred fill, " + CI),
          "model_method_means.csv": (model_means, "Mean faithfulness gap, deletion AUC and insertion AUC per model, method (including controls and rollout) and fill, " + CI + " (exploratory)"),
          "class_method_means.csv": (class_means, "Mean faithfulness gap per class and method pooled over eligible models, " + CI),
          "seed_spread.csv": (seed_sd, "Per model and method: mean faithfulness gap over seeds and its SD over the 3 seeds (Kermany, blurred fill)"),
          "sensitivity_all_correct.csv": (sens_correct.assign(n_images_per_class=json.dumps(n_correct.to_dict())), "Sensitivity: H1 contrasts on images classified correctly by every eligible model and seed (own Holm over 9 slots)"),
          "sensitivity_mean_fill.csv": (sens_fill, "Sensitivity: family contrasts with the mean-intensity fill, " + CI + " (no tests)"),
          "octdl_exploratory.csv": (octdl_rows, "Exploratory dataset comparison: OCTDL DME/NORMAL family means and contrasts next to Kermany DME/NORMAL family means (blurred fill), " + CI + " (no tests)"),
          "test_accuracy.csv": (acc, "Balanced accuracy and macro F1 over the label set present (four labels on Kermany; DME and NORMAL on OCTDL, four-way predictions kept), accuracy, share predicted outside the label set; mean and SD over seeds"),
          "ig_completeness.csv": (ig_check, "Integrated Gradients completeness diagnostic: relative residual of attribution sum versus logit difference (median, 90th percentile)")}
for fn, (df, desc) in tables.items():
    save(df, fn, desc, "save")

figs = {"fig_family_gap": (fig_family(fam_means), "Faithfulness gap by architecture family and method (Kermany, blurred fill): mean and " + CI + "; values in family_method_means.csv"),
        "fig_class_gap": (fig_class(class_means), "Faithfulness gap by disease class and method, pooled over eligible models (NORMAL descriptive): mean and " + CI + "; values in class_method_means.csv"),
        "fig_curves": (fig_curves(), "Mean deletion and insertion curves (true-class probability) by family and method, with the random-order curve of the same models (Kermany, blurred fill); values in curve_means.csv")}

# %%
# Save the figures and the run record
for name, (fig, desc) in figs.items():
    for ext in ("pdf", "png"):
        save(fig, f"{name}.{ext}", desc, "figures")
    plt.close(fig)
record = {"python": platform.python_version(), "packages": VERSIONS, "seed": SEED, "resamples": B, "eligible_models": eligible,
          "kermany_groups": len(KER_GROUPS), "octdl_groups": len(OCT_GROUPS), "smoke": SMOKE,
          "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
if not SMOKE:
    record["input_sha256"] = INPUT_HASHES
save(record, "statistics_environment.json", "Statistics run record: seed, resamples, eligible models, group counts, input file hashes, versions", "save")
print("SMOKE TEST OK" if SMOKE else "statistics done")
