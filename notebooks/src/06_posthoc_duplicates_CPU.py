# %% [markdown]
# # RUNTIME: local workstation CPU, `.venv-local` (deviation D4)
#
# Project: `qpain/paper-01` · Stage: `06 post hoc sensitivity (D11)` · Estimated time: under 5 minutes
#
# POST HOC sensitivity analyses after Codex's step-5 check (analysis_plan.md section 10, D11). (a) H1 and H2 tests of
# notebook 04 repeated, with notebook 04's own functions and resampling matrix, on the attribution list without the
# image with conflicting labels and with one image per same-label exact duplicate; test balanced accuracy without the
# pixel-identical test images that carry more than one class. (b) OCTDL and Kermany family means per class.
#
# Run: `PAPER_ROOT=<project> python -m jupyter nbconvert --to notebook --execute notebooks/06_posthoc_duplicates_CPU.ipynb ...`
# Source of this notebook: `notebooks/src/06_posthoc_duplicates_CPU.py` (rebuild with scripts/py-to-ipynb.py).
# Revised after Codex review 008 (two minor points); rerun.

# %%
# Setup: notebook 04's functions and constants, executed unchanged from its source
import os, ast, json, hashlib, datetime, platform
import importlib.metadata as ilmd
import numpy as np, pandas as pd
from scipy.stats import norm as _norm

ROOT = os.environ.get("PAPER_ROOT", "/content/paper-01")
OUT = os.path.join(ROOT, "outputs")
NOTEBOOK = "06_posthoc_duplicates_CPU"
TAG = "POST HOC sensitivity (D11)"

def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

NB04_SRC = os.path.join(ROOT, "notebooks", "src", "04_statistics_figures_CPU.py")
tree = ast.parse(open(NB04_SRC, encoding="utf-8").read())
FUNCS = {"resampler", "gsums", "n_groups", "mean_stat", "diff_stat", "summarise", "holm", "wide", "family_values", "family_contrasts",
         "class_contrasts", "add_holm", "mean_rows", "acc_table"}
CONSTS = {"ALPHA", "CLASSES", "FAMILIES", "PRIMARY_METHODS", "FAMILY_PAIRS", "CLASS_PAIRS"}
body = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in FUNCS)
        or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in CONSTS for t in n.targets))]
names = {n.name for n in body if isinstance(n, ast.FunctionDef)} | {t.id for n in body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
assert FUNCS | CONSTS <= names, f"missing from notebook 04: {sorted((FUNCS | CONSTS) - names)}"
_src = open(NB04_SRC, encoding="utf-8").read()
assert "SEED, B = 20261119, (200 if SMOKE else 10_000)" in _src, "notebook 04's seed or resample count changed"
G = {"np": np, "pd": pd, "_norm": _norm, "B": 10_000, "SEED": 20261119}
exec(compile(ast.Module(body=body, type_ignores=[]), NB04_SRC, "exec"), G)
assert G["SEED"] == 20261119
VERSIONS = {m: ilmd.version(m) for m in ("numpy", "pandas", "scipy")}
reg = json.load(open(os.path.join(OUT, "statistics_environment.json")))
INPUT_HASHES = {f: sha256_file(os.path.join(OUT, f)) for f in ("faithfulness_scores.csv.gz", "kermany_split.csv", "predictions_test.csv.gz", "family_method_means.csv")}
assert reg["input_sha256"]["faithfulness_scores.csv.gz"] == INPUT_HASHES["faithfulness_scores.csv.gz"] and reg["packages"]["numpy"] == VERSIONS["numpy"]
print(TAG, "| notebook 04 source", sha256_file(NB04_SRC)[:16], "|", VERSIONS)

def save(df, filename, description):
    path = os.path.join(OUT, filename)
    df.to_csv(path + ".tmp", index=False, lineterminator="\n"); os.replace(path + ".tmp", path)
    man = os.path.join(OUT, "MANIFEST.md")
    rows = [l for l in open(man).read().splitlines() if not l.startswith(f"| {filename} |")]
    rows.append(f"| {filename} | {NOTEBOOK} | save | {description} |")
    with open(man + ".tmp", "w", newline="\n") as f:
        f.write("\n".join(rows) + "\n")
    os.replace(man + ".tmp", man)
    print("saved", filename)

# %%
# Inputs prepared exactly as in notebook 04 (seed-averaged per image), and its resampling matrices
scores = pd.read_csv(os.path.join(OUT, "faithfulness_scores.csv.gz"), dtype={"patient_id": str, "group": str})
key = ["dataset", "member", "class", "group", "model", "family", "method", "fill"]
img = scores.groupby(key, as_index=False)[["deletion_auc", "insertion_auc", "faithfulness_gap"]].mean()
G["img"] = img
G["FAMILY_OF"] = scores.drop_duplicates("model").set_index("model")["family"].to_dict()
G["KER_GROUPS"], G["C_KER"] = G["resampler"](img.loc[img["dataset"] == "kermany", "group"], G["SEED"])
G["OCT_GROUPS"], G["C_OCT"] = G["resampler"](img.loc[img["dataset"] == "octdl", "group"], G["SEED"] + 1)
# the matrix must reproduce a published notebook-04 interval exactly
_fm = G["mean_rows"]("kermany", "blur", "family", ["gradcam"])
_pub = pd.read_csv(os.path.join(OUT, "family_method_means.csv")).query("method == 'gradcam'").set_index("family")
assert all(np.isclose(_fm.set_index("family").loc[f, c], _pub.loc[f, c], rtol=0, atol=1e-12) for f in _pub.index for c in ("estimate", "ci95_low", "ci95_high"))
print("resampling reproduces notebook 04's published Kermany Grad-CAM family estimates and interval ends (absolute tolerance 1e-12)")

# %%
# (a) Exact duplicates in the attribution list: drop label conflicts entirely, keep the first (by name) of same-label copies
split = pd.read_csv(os.path.join(OUT, "kermany_split.csv"), dtype={"patient_id": str, "group": str})
att = split[split["in_attribution_test"]].sort_values("member")
dup = att.groupby("pixel_sha256").agg(n=("member", "size"), n_class=("class", "nunique"))
conflict_px = dup.index[dup["n_class"] > 1]
drop_conflict = set(att.loc[att["pixel_sha256"].isin(conflict_px), "member"])
same = att[~att["pixel_sha256"].isin(conflict_px)]
drop_copy = set(same[same.duplicated("pixel_sha256", keep="first")]["member"])
clean = sorted(set(att["member"]) - drop_conflict - drop_copy)
dup_record = pd.DataFrame([{"member": m, "class": att.set_index("member").loc[m, "class"], "excluded_because": "label conflict (same pixels, different class)"} for m in sorted(drop_conflict)]
                          + [{"member": m, "class": att.set_index("member").loc[m, "class"], "excluded_because": "same-label exact copy (first by name kept)"} for m in sorted(drop_copy)])
print(dup_record.to_string(index=False)); print("attribution images kept:", len(clean), "of", len(att))

h1s = G["add_holm"](G["family_contrasts"]("kermany", "blur", members=clean, tag=TAG + ": H1 without label conflicts and exact copies"))
h2s = G["add_holm"](G["class_contrasts"]("kermany", "blur", members=clean, tag=TAG + ": H2 without label conflicts and exact copies"))
h1 = pd.read_csv(os.path.join(OUT, "h1_family_contrasts.csv")); h2 = pd.read_csv(os.path.join(OUT, "h2_class_contrasts.csv"))
def with_registered(new, old):   # one-to-one join on the contrast's identity, not row position (Codex review 008)
    k = ["dataset", "fill", "method", "contrast"]
    o = old[k + ["estimate", "p_holm", "difference_at_0.05_holm"]].rename(columns={"estimate": "registered_estimate", "p_holm": "registered_p_holm",
                                                                                 "difference_at_0.05_holm": "registered_decision"})
    j = new.merge(o, on=k, how="left", validate="one_to_one")
    assert j["registered_estimate"].notna().all() and len(j) == len(new) == len(old)
    j["same_decision_as_registered"] = j["difference_at_0.05_holm"].astype(str) == j["registered_decision"].astype(str)
    return j
h1s, h2s = with_registered(h1s, h1), with_registered(h2s, h2)
pd.set_option("display.width", 220)
print(h1s[["method", "contrast", "estimate", "registered_estimate", "p_holm", "same_decision_as_registered", "n_images"]].round(4).to_string(index=False))
print(h2s[["method", "contrast", "estimate", "registered_estimate", "p_holm", "same_decision_as_registered", "n_a", "n_b"]].round(4).to_string(index=False))

# Test accuracy without pixel-identical test images that carry more than one class
test = split[split["partition"] == "test"]
nc = test.groupby("pixel_sha256")["class"].nunique()
conflict_test = set(test.loc[test["pixel_sha256"].isin(nc.index[nc > 1]), "member"])
pt = pd.read_csv(os.path.join(OUT, "predictions_test.csv.gz"))
acc_all = G["acc_table"](pt, "kermany_test (registered)")
acc_clean = G["acc_table"](pt[~pt["member"].isin(conflict_test)], "kermany_test without label-conflict copies")
acc = pd.concat([acc_all, acc_clean], ignore_index=True)
acc["n_label_conflict_images_removed"] = np.where(acc["set"].str.contains("without"), len(conflict_test), 0)
print("test images in label-conflict sets:", len(conflict_test)); print(acc.round(4).to_string(index=False))

# %%
# (b) OCTDL and Kermany family means per class (the two lists have different class mixes)
octdl_members = {c: img.loc[(img["dataset"] == "octdl") & (img["class"] == c), "member"].unique() for c in ("DME", "NORMAL")}
ker_members = {c: img.loc[(img["dataset"] == "kermany") & (img["class"] == c), "member"].unique() for c in ("DME", "NORMAL")}
per_class = pd.concat([G["mean_rows"](ds, "blur", "family", G["PRIMARY_METHODS"], members=mem[c]).assign(dataset_label=ds, class_=c)
                       for ds, mem in (("octdl", octdl_members), ("kermany", ker_members)) for c in ("DME", "NORMAL")], ignore_index=True)
per_class = per_class.rename(columns={"class_": "class"})
print(per_class[["dataset", "class", "method", "family", "estimate", "ci95_low", "ci95_high", "n_images"]].round(3).to_string(index=False))

# %%
CI = "pointwise 95% cluster-bootstrap percentile interval (same resamples as notebook 04)"
save(dup_record, "posthoc_duplicates_excluded.csv", f"{TAG}: attribution images excluded in the duplicate sensitivity analysis and why")
save(h1s, "posthoc_h1_without_duplicates.csv", f"{TAG}: H1 family contrasts without label conflicts and exact copies; bootstrap-t, Holm over 9 slots, {CI}; registered values alongside")
save(h2s, "posthoc_h2_without_duplicates.csv", f"{TAG}: H2 class contrasts without label conflicts and exact copies; bootstrap-t, Holm over 9 slots, {CI}; registered values alongside")
save(acc, "posthoc_test_accuracy_without_conflicts.csv", f"{TAG}: test balanced accuracy and macro F1 with and without the pixel-identical test images that carry more than one class")
save(per_class, "posthoc_octdl_kermany_per_class.csv", f"{TAG}: OCTDL and Kermany family means per class (DME, NORMAL), blurred fill, {CI}; a dataset comparison, no tests")
save(pd.DataFrame([{"python": platform.python_version(), **VERSIONS, "notebook04_source_sha256": sha256_file(NB04_SRC), **{f"sha256_{k}": v for k, v in INPUT_HASHES.items()},
                    "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}]),
     "posthoc_duplicates_environment.csv", f"{TAG}: run record (versions, input hashes, notebook 04 source hash)")
print("post hoc sensitivity done")
