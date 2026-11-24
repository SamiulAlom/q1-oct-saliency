# %% [markdown]
# # RUNTIME: local workstation CPU, `.venv-local` (deviation D4)
#
# Project: `qpain/paper-01` · Stage: `07 paper figure and tables` · Estimated time: under 1 minute
#
# Builds the manuscript's Fig. 2 and Tables I-II (Jon's ruling, decisions/009) from existing outputs only. No new
# analysis: every value is copied from a file in outputs/, and each table row names its source file.
# Fig. 2 (a): per-model faithfulness gap of Grad-CAM, IG and occlusion beside the edge-map control (blurred fill;
#   model_method_means.csv). (b): per-class gap pooled over models beside the edge-map gap per class (blurred fill;
#   class_method_means.csv for the methods, posthoc_class_edge.csv for the edge map: that per-class edge value is a post
#   hoc output of notebook 05 (D10) and is labelled so).
# Table I: the 18 pre-specified contrasts (h1_family_contrasts.csv, h2_class_contrasts.csv).
# Table II (post hoc, D10): edge-adjusted gap per model, both fills, with test balanced accuracy
#   (posthoc_edge_adjusted_by_model.csv, test_accuracy.csv).
#
# Source of this notebook: `notebooks/src/07_paper_figures_tables_CPU.py` (rebuild with scripts/py-to-ipynb.py).
# Revised after Codex review 009 (post hoc label in panel b, exact-key checks, caption and table notes).

# %%
import os, json, hashlib, datetime
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("PAPER_ROOT", "/content/paper-01")
OUT = os.path.join(ROOT, "outputs")
NOTEBOOK = "07_paper_figures_tables_CPU"
MODELS = ["efficientnet_b0", "mobilenetv3_large_100", "mobilevit_s", "efficientformer_l1", "deit_tiny_patch16_224", "deit_small_patch16_224"]
LABEL = {"efficientnet_b0": "EffNet-B0", "mobilenetv3_large_100": "MbNetV3-L", "mobilevit_s": "MobileViT-S",
         "efficientformer_l1": "EffFormer-L1", "deit_tiny_patch16_224": "DeiT-Ti", "deit_small_patch16_224": "DeiT-S"}
FAMILY = {"efficientnet_b0": "Conv.", "mobilenetv3_large_100": "Conv.", "mobilevit_s": "Hybrid", "efficientformer_l1": "Hybrid",
          "deit_tiny_patch16_224": "Transf.", "deit_small_patch16_224": "Transf."}
METHODS = ["gradcam", "ig", "occlusion", "edge"]
MLABEL = {"gradcam": "Grad-CAM", "ig": "Integrated Gradients", "occlusion": "Occlusion (perturbation-matched)", "edge": "Edge map (control)"}
COLOR = {"gradcam": "#0072B2", "ig": "#D55E00", "occlusion": "#009E73", "edge": "#555555"}   # Okabe-Ito plus grey for the control
MARK = {"gradcam": "o", "ig": "s", "occlusion": "^", "edge": "D"}
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]

def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def save(obj, filename, description):
    path = os.path.join(OUT, filename); tmp = path + ".tmp"
    if hasattr(obj, "to_csv"):
        obj.to_csv(tmp, index=False, lineterminator="\n")
    elif hasattr(obj, "savefig"):
        obj.savefig(tmp, format=filename.rsplit(".", 1)[1], dpi=600)
    else:
        with open(tmp, "w", newline="\n") as f:
            json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)
    man = os.path.join(OUT, "MANIFEST.md")
    rows = [l for l in open(man).read().splitlines() if not l.startswith(f"| {filename} |")]
    rows.append(f"| {filename} | {NOTEBOOK} | build | {description} |")
    with open(man + ".tmp", "w", newline="\n") as f:
        f.write("\n".join(rows) + "\n")
    os.replace(man + ".tmp", man)
    print("saved", filename)

INPUTS = ["model_method_means.csv", "class_method_means.csv", "posthoc_class_edge.csv", "h1_family_contrasts.csv", "h2_class_contrasts.csv",
          "posthoc_edge_adjusted_by_model.csv", "test_accuracy.csv"]
INPUT_HASHES = {f: sha256_file(os.path.join(OUT, f)) for f in INPUTS}

# %%
# Fig. 2 data, copied from the outputs (blurred fill, Kermany)
mm = pd.read_csv(os.path.join(OUT, "model_method_means.csv"))
a = mm[(mm["dataset"] == "kermany") & (mm["fill"] == "blur") & (mm["outcome"] == "faithfulness_gap") & mm["method"].isin(METHODS)]
assert len(a) == len(MODELS) * len(METHODS), "panel (a) does not have one value per model and method"
cm = pd.read_csv(os.path.join(OUT, "class_method_means.csv"))
b_methods = cm[(cm["dataset"] == "kermany") & (cm["fill"] == "blur") & (cm["outcome"] == "faithfulness_gap") & cm["method"].isin(METHODS[:3])][["method", "class", "estimate", "ci95_low", "ci95_high", "n_images"]]
ce = pd.read_csv(os.path.join(OUT, "posthoc_class_edge.csv"))
b_edge = ce[(ce["fill"] == "blur") & (ce["method"] == "edge") & (ce["outcome"] == "edge map gap (descriptive control)")][["method", "class", "estimate", "ci95_low", "ci95_high", "n_images"]]
b = pd.concat([b_methods, b_edge], ignore_index=True)
assert len(b) == len(CLASSES) * len(METHODS), "panel (b) does not have one value per class and method"
fig_data = pd.concat([a.assign(panel="a", unit=a["model"], source="model_method_means.csv")[["panel", "unit", "method", "estimate", "ci95_low", "ci95_high", "n_images", "source"]],
                      b.assign(panel="b", unit=b["class"], source=np.where(b["method"] == "edge", "posthoc_class_edge.csv (post hoc, D10)", "class_method_means.csv"))
                      [["panel", "unit", "method", "estimate", "ci95_low", "ci95_high", "n_images", "source"]]], ignore_index=True)
assert not fig_data.duplicated(["panel", "unit", "method"]).any()
assert set(map(tuple, fig_data.loc[fig_data["panel"] == "a", ["unit", "method"]].values)) == {(u, m) for u in MODELS for m in METHODS}
assert set(map(tuple, fig_data.loc[fig_data["panel"] == "b", ["unit", "method"]].values)) == {(u, m) for u in CLASSES for m in METHODS}

# %%
# Fig. 2: two panels, IEEE double-column width (7.16 in); interval ends drawn directly; colour plus marker per method
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5), layout="constrained", sharey=True, gridspec_kw={"width_ratios": [6, 4]})
for ax, panel, units, ulabel in ((axes[0], "a", MODELS, lambda u: f"{LABEL[u]}\n({FAMILY[u]})"), (axes[1], "b", CLASSES, lambda u: u)):
    d = fig_data[fig_data["panel"] == panel]
    for k, meth in enumerate(METHODS):
        dm = d[d["method"] == meth].set_index("unit").reindex(units)
        x = np.arange(len(units)) + (k - 1.5) * 0.18
        ax.vlines(x, dm["ci95_low"], dm["ci95_high"], color=COLOR[meth], lw=0.8)
        ax.plot(x, dm["estimate"], MARK[meth], color=COLOR[meth], ms=3.5, mfc="white" if meth == "edge" else COLOR[meth],
                label=MLABEL[meth] if panel == "a" else None)
    ax.axhline(0, color="0.5", lw=0.5, ls="--")
    ax.set_xticks(range(len(units)), [ulabel(u) for u in units])
    ax.set_title("(a) Per model (820 images, 680 groups)" if panel == "a" else "(b) Per class, six models pooled", fontsize=7)
axes[1].text(0.98, 0.10, "NORMAL: descriptive\nedge-map values: post hoc", transform=axes[1].transAxes, ha="right", va="bottom", fontsize=6, color=COLOR["edge"])
axes[0].set_ylabel("Faithfulness gap\n(mean, 95% interval)")
fig.legend(loc="outside upper center", ncol=4, frameon=False, fontsize=6.5)

# %%
# Table I: the 18 pre-specified contrasts (values copied; formatting only)
def fmt_ci(r):
    return f"{r['estimate']:+.3f} [{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]"
def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"
h1 = pd.read_csv(os.path.join(OUT, "h1_family_contrasts.csv")); h2 = pd.read_csv(os.path.join(OUT, "h2_class_contrasts.csv"))
assert len(h1) == 9 and len(h2) == 9 and (h1["status"] == "estimated").all() and (h2["status"] == "estimated").all()
assert not h1.duplicated(["method", "contrast"]).any() and not h2.duplicated(["method", "contrast"]).any()
t1 = pd.concat([pd.DataFrame({"hypothesis": "H1 (family)", "method": d["method"], "contrast": d["contrast"], "estimate_ci95": d.apply(fmt_ci, axis=1),
                              "p_holm": d["p_holm"].map(fmt_p), "seed_robust": d["larger_than_max_seed_sd"].map({True: "yes", False: "no"}),
                              "source": "h1_family_contrasts.csv"}) for d in [h1]] +
               [pd.DataFrame({"hypothesis": "H2 (class)", "method": h2["method"], "contrast": h2["contrast"], "estimate_ci95": h2.apply(fmt_ci, axis=1),
                              "p_holm": h2["p_holm"].map(fmt_p), "seed_robust": "n/a", "source": "h2_class_contrasts.csv"})], ignore_index=True)
print(t1.to_string(index=False))

# Table II (post hoc, D10): edge-adjusted gap per model for both fills, with test balanced accuracy (mean over seeds)
pa = pd.read_csv(os.path.join(OUT, "posthoc_edge_adjusted_by_model.csv"))
pa = pa[pa["method"].isin(["gradcam", "ig", "occlusion"])]
acc = pd.read_csv(os.path.join(OUT, "test_accuracy.csv")).query("set == 'kermany_test'")
assert acc["model"].is_unique and set(acc["model"]) == set(MODELS) and (acc["label_set"] == "CNV+DME+DRUSEN+NORMAL").all()
acc = acc.set_index("model")
rows = []
for mo in MODELS:
    r = {"model": LABEL[mo], "family": FAMILY[mo], "test_balanced_accuracy": f"{acc.loc[mo, 'balanced_accuracy_mean']:.3f}"}
    for fill in ("blur", "mean"):
        for meth in ("gradcam", "ig", "occlusion"):
            q = pa[(pa["model"] == mo) & (pa["fill"] == fill) & (pa["method"] == meth)]
            assert len(q) == 1
            r[f"{meth}_{fill}"] = fmt_ci(q.iloc[0])
    rows.append(r)
t2 = pd.DataFrame(rows); t2["source"] = "posthoc_edge_adjusted_by_model.csv; test_accuracy.csv"
print(t2.to_string(index=False))

# %%
NOTES = {
    "fig2_caption": ("Faithfulness gap (insertion AUC minus deletion AUC, minus the same for random orders) of Grad-CAM, Integrated Gradients and occlusion "
                     "(a perturbation-matched reference) beside the model-independent edge-map control, blurred fill, Kermany attribution list "
                     "(820 images in 680 patient/duplicate groups). (a) Per model; (b) per class, pooled over the six models; the method means are "
                     "pre-specified outputs, the per-class edge-map values are post hoc (D10); NORMAL is descriptive. Points are means; bars are pointwise "
                     "95% percentile intervals from 10,000 cluster-bootstrap resamples of the groups."),
    "table1_note": ("Pre-specified contrasts (analysis plan committed before data access). Estimate with pointwise 95% cluster-bootstrap interval; "
                    "p from a studentised bootstrap test, Holm-adjusted within each family of nine; <0.001 means the smallest attainable value "
                    "with 10,000 resamples. Seed robust (H1 only): yes if the absolute contrast exceeds the largest seed SD of the models involved."),
    "table2_note": ("POST HOC (D10): edge-adjusted gap = method gap minus edge-map gap on the same image, model, seed and fill, averaged over seeds; "
                    "mean with pointwise 95% cluster-bootstrap interval, no tests. Test balanced accuracy: mean over three seeds on the full Kermany "
                    "test partition (22,537 images; CNV, DME, DRUSEN, NORMAL)."),
}
save(NOTES, "paper_captions_notes.json", "Caption of Fig. 2 and notes for Tables I-II (definitions of intervals, tests, seed robustness, accuracy population)")
save(fig_data, "fig2_data.csv", "Values plotted in Fig. 2 (a: per model, b: per class; blurred fill), each with its source file; panel (b) edge values are post hoc (D10)")
for ext in ("pdf", "png"):
    save(fig, f"fig2_gap_vs_edge.{ext}", "Fig. 2 (caption in paper_captions_notes.json): gap of Grad-CAM, IG and occlusion beside the edge-map control, (a) per model, (b) per class pooled over models (edge values in b post hoc); blurred fill; values in fig2_data.csv")
plt.close(fig)
save(t1, "table1_prespecified_contrasts.csv", "Table I: the 18 pre-specified contrasts (H1 family, H2 class): estimate, pointwise 95% interval, Holm-adjusted p, seed robustness (H1); formatted copies of h1/h2 outputs")
save(t2, "table2_posthoc_edge_adjusted.csv", "Table II (POST HOC, D10): edge-adjusted gap per model and method for both fills, with test balanced accuracy; formatted copies of posthoc_edge_adjusted_by_model.csv and test_accuracy.csv")
save({"inputs_sha256": INPUT_HASHES, "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
     "paper_figures_environment.json", "Run record of notebook 07: input file hashes")
print("figures and tables done")
