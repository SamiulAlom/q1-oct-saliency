# %% [markdown]
# # RUNTIME: CPU (standard) — do NOT pick a GPU, do NOT pick High-RAM
#
# Project: `qpain/paper-01` · Stage: `01 audit and split` · Estimated units: 0.15
#
# Downloads Kermany OCT v3 and OCTDL v4 from Mendeley Data, verifies publisher SHA-256 hashes, audits patient
# identifiers and duplicate images, and writes the frozen split and the fixed attribution test list
# (analysis_plan.md section 3 and deviations D1-D3). No model is trained here. Only derived metadata is published;
# no image leaves the VM.
#
# Run unattended with `node scripts/colab-run.mjs projects/qpain/paper-01 notebooks/01_audit_split_CPU.ipynb`.
# Source of this notebook: `notebooks/src/01_audit_split_CPU.py` (edit the source, then rebuild with scripts/py-to-ipynb.py).

# %%
# Setup: paths from PAPER_ROOT, seed, parameters fixed by analysis_plan.md section 3
import os, sys, json, re, io, time, hashlib, zipfile, random, platform, datetime, subprocess, collections
import importlib.metadata as ilmd
ROOT = os.environ.get("PAPER_ROOT", "/content/paper-01")
RAW, OUT, CKPT = (os.path.join(ROOT, d) for d in ("data/raw", "outputs", "checkpoints"))
for d in (RAW, OUT, CKPT):
    os.makedirs(d, exist_ok=True)
NOTEBOOK = "01_audit_split_CPU"
PARSER_VERSION = "3"         # bump when the inventory code changes: old inventory checkpoints are then ignored

SEED = 20261119              # plan section 3, item 5
HAMMING_MAX = 4              # pairs up to this distance are searched, counted and profiled (plan section 3, item 3)
GROUP_HAMMING = 0            # deviation D5: only an identical pHash links two images; distances 1-4 are reported, not used
FRACTIONS = (0.70, 0.10, 0.20)   # train, val, test (by groups)
TRAIN_CAP = 12               # images per patient-class pair in the capped training set
ATTR_PER_CLASS = 250         # patients per class in the attribution test list
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
MAX_NEAR_DUP_PAIRS = 20_000_000   # safety stop on memory
MAX_GROUP_SHARE = 0.05       # deviation D2: stop if one group holds more than 5% of images or of patients

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imagehash==4.3.1"], check=True)
import numpy as np, pandas as pd, requests, imagehash
from PIL import Image
random.seed(SEED); np.random.seed(SEED)
VERSIONS = {m: ilmd.version(m) for m in ("numpy", "pandas", "Pillow", "ImageHash", "scipy", "PyWavelets", "requests")}
print("Run at", datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "| Python", platform.python_version())
print("SEED", SEED, "| HAMMING_MAX", HAMMING_MAX, "| packages", VERSIONS)

def popcount64(x):
    """Number of set bits in each uint64 (numpy >= 2 has bitwise_count; older numpy uses a byte table)."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return table[np.ascontiguousarray(x).view(np.uint8).reshape(x.shape + (8,))].sum(axis=-1, dtype=np.uint8)

def phash64(img):
    return int(str(imagehash.phash(img, hash_size=8)), 16)

# pHash fingerprint of a fixed synthetic image: lets a later run prove its hashing stack gives the same hashes
_yy, _xx = np.mgrid[0:256, 0:256]
_probe = Image.fromarray(((np.sin(_xx / 9.0) + np.cos(_yy / 5.0) + 2) * 63).astype(np.uint8), mode="L")
PHASH_PROBE = f"{phash64(_probe):016x}"
print("pHash probe:", PHASH_PROBE)

# %%
# Helpers: manifest-recorded outputs, hashed downloads
def save_output(obj, filename, description, cell=""):
    path = os.path.join(OUT, filename)
    if hasattr(obj, "to_csv"):
        obj.to_csv(path, index=False)
    else:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)
    man = os.path.join(OUT, "MANIFEST.md")
    if not os.path.exists(man):
        with open(man, "w") as f:
            f.write("| file | notebook | cell | description |\n|---|---|---|---|\n")
    rows = [l for l in open(man).read().splitlines() if not l.startswith(f"| {filename} |")]
    rows.append(f"| {filename} | {NOTEBOOK} | {cell} | {description} |")
    with open(man, "w") as f:
        f.write("\n".join(rows) + "\n")
    print("saved", filename)
    return path

def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

API = "https://data.mendeley.com/public-api/datasets/{id}/files?folder_id=root&version={v}"
HEAD = {"Accept": "application/vnd.mendeley-public-dataset.1+json"}

def mendeley_listing(dataset_id, version):
    r = requests.get(API.format(id=dataset_id, v=version), headers=HEAD, timeout=60)
    r.raise_for_status()
    return {f["filename"]: {"sha256": f["content_details"]["sha256_hash"], "size": f["content_details"]["size"],
                            "url": f["content_details"]["download_url"]} for f in r.json()}

def download_verified(url, name, expected_sha, expected_size, tries=3):
    path = os.path.join(RAW, name)
    if os.path.exists(path) and os.path.getsize(path) == expected_size and sha256_file(path) == expected_sha:
        print("already present and verified:", name)
        return path
    tmp, last = path + ".part", None
    for attempt in range(1, tries + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for b in r.iter_content(1 << 22):
                        f.write(b)
            size = os.path.getsize(tmp)
            assert size == expected_size, f"{name}: {size} bytes received, {expected_size} expected"
            got = sha256_file(tmp)
            assert got == expected_sha, f"{name}: SHA-256 {got} differs from the publisher's {expected_sha}"
            os.replace(tmp, path)             # atomic: a killed download never leaves a file that looks complete
            print("downloaded and verified:", name, size, "bytes")
            return path
        except (requests.RequestException, AssertionError) as e:
            last = e
            print(f"download attempt {attempt}/{tries} failed for {name}: {e}")
            time.sleep(20 * attempt)
    raise RuntimeError(f"{name}: download failed after {tries} attempts: {last}")

# %%
# Download. Publisher hashes below were read from the Mendeley Data public API on 2026-11-19 (data_sources.md).
# The listing is read again now and after the download; any difference stops the run.
PINNED = {
    ("rscbjbr9sj", 3): {"ZhangLabData.zip": "acbc42869440eec719361942d0ba560f132742230262f22e78bfb93eb5ea4563"},
    ("sncdhf53xc", 4): {"OCTDL_labels.csv": "b7db437cfe67741623f26e8bb2cef0f4d6828d6d86150f752f412da1c4ee8b3a",
                        "OCTDL.zip": "637a72ff3319645aac30408d4d838189337ac0b8b0f9faabf7c8156d271f46f7"},
}
raw_rows, paths = [], {}
for (ds, ver), files in PINNED.items():
    before = mendeley_listing(ds, ver)
    for name, sha in files.items():
        assert name in before, f"{name} is not in the version-{ver} listing of {ds}"
        assert before[name]["sha256"] == sha, f"{name}: listing hash changed since 2026-11-19"
        paths[name] = download_verified(before[name]["url"], name, sha, before[name]["size"])
    after = mendeley_listing(ds, ver)
    assert {k: v["sha256"] for k, v in after.items()} == {k: v["sha256"] for k, v in before.items()}, "listing changed during download"
    for name, sha in files.items():
        raw_rows.append({"file": name, "dataset_doi": f"10.17632/{ds}.{ver}", "url": before[name]["url"],
                         "downloaded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                         "size_bytes": before[name]["size"], "sha256": sha})
save_output(pd.DataFrame(raw_rows), "raw_data_manifest.csv", "Raw files: DOI, URL, download time, size, verified SHA-256", "download")

# %%
# Inventory: one row per OCT image, read straight from the zip (nothing is extracted to disk).
NAME_RE = re.compile(r"^(CNV|DME|DRUSEN|NORMAL)-(\d+)-(\d+)\.jpe?g$", re.IGNORECASE)
IMG_EXT = (".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff")
# Parser fixtures: the documented name pattern, tested before any archive is read
for _name, _want in (("CNV-1016042-1.jpeg", ("CNV", "1016042", "1")), ("DME-15307-3.jpeg", ("DME", "15307", "3")),
                     ("DRUSEN-228939-12.jpeg", ("DRUSEN", "228939", "12")), ("NORMAL-9251-2.jpeg", ("NORMAL", "9251", "2"))):
    assert NAME_RE.match(_name).groups() == _want
assert NAME_RE.match("NORMAL2-IM-0001-0001.jpeg") is None and NAME_RE.match("person1_bacteria_1.jpeg") is None

def archive_overview(zip_path):
    """Member names, a path/extension overview, and a stop on duplicate member names."""
    with zipfile.ZipFile(zip_path) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
    dup = [n for n, c in collections.Counter(names).items() if c > 1]
    assert not dup, f"{len(dup)} duplicate member names in {os.path.basename(zip_path)}: {dup[:3]}"
    tree = collections.Counter(("/".join(n.replace("\\", "/").split("/")[:-1]), os.path.splitext(n)[1].lower()) for n in names)
    return sorted(names), [{"folder": k[0], "extension": k[1], "files": v} for k, v in sorted(tree.items())]

def pixel_sha(im):
    """Hash of decoded grayscale pixels and size: equal for pixel-identical images saved with different bytes."""
    g = im.convert("L")
    return hashlib.sha256(f"{g.width}x{g.height}:".encode() + g.tobytes()).hexdigest()

def inventory(zip_path, todo, ckpt_name):
    """todo: list of (member, extra columns). Returns one row per image; checkpointed with a fingerprint."""
    fp = hashlib.sha256(json.dumps([sha256_file(zip_path), PARSER_VERSION, VERSIONS, PHASH_PROBE, len(todo)], sort_keys=True).encode()).hexdigest()
    ck, ckfp = os.path.join(CKPT, ckpt_name), os.path.join(CKPT, ckpt_name + ".fingerprint")
    if os.path.exists(ck) and os.path.exists(ckfp) and open(ckfp).read().strip() == fp:
        df = pd.read_csv(ck, dtype={"patient_id": str, "phash": str})
        if len(df) == len(todo) and list(df["member"]) == [m for m, _ in todo]:
            print("resuming inventory from a checkpoint with a matching fingerprint:", ckpt_name)
            return df
    rows, t0 = [], time.time()
    with zipfile.ZipFile(zip_path) as z:
        for i, (m, extra) in enumerate(todo):
            try:
                data = z.read(m)
                with Image.open(io.BytesIO(data)) as im:
                    im.load()
                    row = {"member": m, **extra, "sha256": hashlib.sha256(data).hexdigest(), "pixel_sha256": pixel_sha(im),
                           "phash": f"{phash64(im.convert('L')):016x}", "width": im.width, "height": im.height, "mode": im.mode}
            except Exception as e:
                raise RuntimeError(f"cannot read image {m} in {os.path.basename(zip_path)}: {e}") from e
            rows.append(row)
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{len(todo)} images, {time.time() - t0:.0f} s")
    df = pd.DataFrame(rows)
    tmp = ck + ".tmp"; df.to_csv(tmp, index=False); os.replace(tmp, ck)
    with open(ckfp, "w") as f: f.write(fp)
    return df

ker_names, ker_tree = archive_overview(paths["ZhangLabData.zip"])
print(pd.DataFrame(ker_tree).to_string(index=False))
ker_todo, pattern_outside, images_unparsed, xray_named_like_oct = [], [], [], []
for m in ker_names:
    parts = m.replace("\\", "/").split("/")
    base = parts[-1]
    if base.startswith("._") or "__MACOSX" in parts or not base.lower().endswith(IMG_EXT):
        continue
    in_oct = any(p.upper() == "OCT" for p in parts[:-1])
    mt = NAME_RE.match(base)
    if in_oct and not mt:
        images_unparsed.append(m)
    elif mt and not in_oct:
        # The archive's chest X-ray NORMAL files are named NORMAL-<id>-0001.jpeg, like OCT scans (seen in the first run, 2026-11-19).
        # They are another modality in a folder named chest_xray: counted and skipped. Anything else with an OCT-like name still stops the run.
        (xray_named_like_oct if any(p.lower() == "chest_xray" for p in parts[:-1]) else pattern_outside).append(m)
    elif mt:
        ker_todo.append((m, {"file_name": base, "class": mt.group(1).upper(), "folder_class": parts[-2].upper(), "patient_id": mt.group(2),
                             "image_no": int(mt.group(3)), "provided_split": next((p.lower() for p in parts if p.lower() in ("train", "test", "val")), "")}))
assert ker_todo, "no OCT images found: the archive layout is not what the parser expects (see the folder overview above)"
assert not images_unparsed, f"{len(images_unparsed)} images under an OCT folder do not match (disease)-(patient)-(number): {images_unparsed[:5]}"
assert not pattern_outside, f"{len(pattern_outside)} images named like OCT scans sit outside an OCT folder: {pattern_outside[:5]}"
print("chest X-ray files with an OCT-like name, skipped:", len(xray_named_like_oct), "| OCT images to read:", len(ker_todo))
ker = inventory(paths["ZhangLabData.zip"], ker_todo, "kermany_inventory.csv")
assert set(ker["class"]) == set(CLASSES), f"classes found: {sorted(set(ker['class']))}"
mismatch = ker[ker["class"] != ker["folder_class"]]
assert mismatch.empty, f"{len(mismatch)} files whose name class differs from the folder class: {mismatch['member'].head(3).tolist()}"
for c in CLASSES:   # parser shown on one real file from every class folder (plan section 3, item 8)
    r = ker[ker["class"] == c].iloc[0]
    assert NAME_RE.match(r["file_name"]).group(2) == r["patient_id"]
    print("parser check:", r["member"], "-> class", r["class"], "patient", r["patient_id"], "image", r["image_no"])
print("Kermany OCT images inventoried:", len(ker))

# %%
# Grouping: connected components of (same patient) OR (identical file) OR (identical pixels) OR (near-duplicate).
class DSU:
    def __init__(self, n): self.p = np.arange(n)
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[max(ra, rb)] = min(ra, rb)

def near_duplicate_pairs(hashes_hex, max_d, block=128):
    """All index pairs i<j with Hamming distance <= max_d between 64-bit hashes (exact, brute force)."""
    h = np.array([int(x, 16) for x in hashes_hex], dtype=np.uint64)
    out_i, out_j, out_d, total = [], [], [], 0
    for s in range(0, len(h), block):
        d = popcount64(h[s:s + block, None] ^ h[None, :])
        ii, jj = np.nonzero(d <= max_d)
        keep = (ii + s) < jj
        ii, jj = ii[keep], jj[keep]
        total += len(ii)
        assert total <= MAX_NEAR_DUP_PAIRS, (
            f"more than {MAX_NEAR_DUP_PAIRS} near-duplicate pairs at distance <= {max_d}: the planned threshold links "
            "too many images; stop and record a deviation in analysis_plan.md before changing it")
        out_i.append((ii + s).astype(np.int64)); out_j.append(jj.astype(np.int64)); out_d.append(d[ii, jj].astype(np.int8))
    pi, pj, pd_ = np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_d)
    return pi, pj, pd_, np.bincount(pd_, minlength=max_d + 1)[:max_d + 1]

def components(df, patient_col, pairs, d):
    """Group label per row using identity links plus near-duplicate pairs with distance <= d (d = -1: identity links only)."""
    pi, pj, pd_ = pairs; n = len(df); dsu = DSU(n)
    for col in (patient_col, "sha256", "pixel_sha256"):
        for idx in df.groupby(col).indices.values():
            for k in idx[1:]: dsu.union(int(idx[0]), int(k))
    m = pd_ <= d
    for a, b in zip(pi[m].tolist(), pj[m].tolist()): dsu.union(a, b)
    return pd.factorize(np.array([dsu.find(i) for i in range(n)]))[0]

def group_profile(df, patient_col, groups, label, dataset):
    g = pd.DataFrame({"g": groups, "p": df[patient_col].to_numpy(), "c": df["class"].to_numpy()})
    img, pat, cls = g.groupby("g").size(), g.groupby("g")["p"].nunique(), g.groupby("g")["c"].nunique()
    # The image maximum and the patient maximum may come from different groups (Codex review 002, point 3)
    return {"dataset": dataset, "links": label, "images": int(len(g)), "patients": int(g["p"].nunique()), "groups": int(len(img)), "largest_group_images": int(img.max()),
            "share_of_images_in_largest_group": round(float(img.max()) / len(g), 4),
            "largest_group_patients": int(pat.max()), "share_of_patients_in_largest_group": round(float(pat.max()) / g["p"].nunique(), 4),
            "groups_with_more_than_one_patient": int((pat > 1).sum()), "groups_with_more_than_one_class": int((cls > 1).sum()),
            "group_image_size_quantiles_50_90_99_max": [int(x) for x in np.quantile(img, [0.5, 0.9, 0.99, 1.0])]}

def group_dataset(df, patient_col, dataset):
    pi, pj, pd_, hist = near_duplicate_pairs(df["phash"].tolist(), HAMMING_MAX)
    pairs = (pi, pj, pd_)
    prof = [group_profile(df, patient_col, components(df, patient_col, pairs, d),
                          "patient and identical images only" if d < 0 else f"plus pHash distance <= {d}", dataset) for d in range(-1, HAMMING_MAX + 1)]
    groups = components(df, patient_col, pairs, GROUP_HAMMING)
    final = prof[GROUP_HAMMING + 1]            # prof[0] is identity links only, prof[d + 1] is distance <= d
    print(pd.DataFrame(prof).to_string(index=False))
    # Deviation D2 (written before any run): the split is not frozen if the planned threshold chains patients together
    # exact counts against denominators, not the rounded shares (Codex review 002, point 3)
    assert final["largest_group_images"] <= MAX_GROUP_SHARE * final["images"] and final["largest_group_patients"] <= MAX_GROUP_SHARE * final["patients"], (
        f"{dataset}: at pHash distance <= {GROUP_HAMMING} one group holds {final['share_of_images_in_largest_group']:.1%} of images and "
        f"{final['share_of_patients_in_largest_group']:.1%} of patients (limit {MAX_GROUP_SHARE:.0%}). The split is NOT written. "
        "Record a deviation in analysis_plan.md section 10 before choosing another threshold.")
    return groups, pairs, hist, prof

ker = ker.sort_values("member").reset_index(drop=True)        # row order used by ker_pairs; not changed again
ker["group"], ker_pairs, ker_hist, profiles = group_dataset(ker, "patient_id", "kermany")
print("Kermany near-duplicate pairs by Hamming distance 0..%d:" % HAMMING_MAX, ker_hist.tolist())

# %%
# Audit numbers (reported in the paper whatever they are)
def audit(df, patient_col, pairs, hist, name):
    pi, pj, pd_ = pairs
    cls_per_patient = df.groupby(patient_col)["class"].nunique()
    pat = df[patient_col].to_numpy()
    def sets_with_other_patient(col):
        g = df.groupby(col)[patient_col].agg(["size", "nunique"])
        return int((g["size"] > 1).sum()), int((g["nunique"] > 1).sum())
    file_sets, file_sets_diff = sets_with_other_patient("sha256")
    pix_sets, pix_sets_diff = sets_with_other_patient("pixel_sha256")
    a = {"dataset": name, "images": int(len(df)), "patients": int(df[patient_col].nunique()),
         "images_per_class": df["class"].value_counts().to_dict(),
         "patients_per_class": df.groupby("class")[patient_col].nunique().to_dict(),
         "patient_ids_in_more_than_one_class": int((cls_per_patient > 1).sum()),
         "identical_file_sets": file_sets, "identical_file_sets_with_different_patient_ids": file_sets_diff,
         "identical_pixel_sets": pix_sets, "identical_pixel_sets_with_different_patient_ids": pix_sets_diff,
         "near_duplicate_pairs_by_hamming_distance": {str(k): int(v) for k, v in enumerate(hist)},
         "near_duplicate_pairs_between_different_patient_ids": int((pat[pi] != pat[pj]).sum()),
         "groups": int(df["group"].nunique()), "largest_group_images": int(df["group"].value_counts().max()),
         "image_sizes_top5": {f"{w}x{h}": int(c) for (w, h), c in df.groupby(["width", "height"]).size().sort_values(ascending=False).head(5).items()},
         "image_modes": df["mode"].value_counts().to_dict()}
    if "provided_split" in df:
        prov = df["provided_split"].to_numpy()
        a["provided_split_counts"] = df["provided_split"].value_counts().to_dict()
        a["patient_ids_in_more_than_one_provided_split"] = int((df.groupby(patient_col)["provided_split"].nunique() > 1).sum())
        a["near_duplicate_pairs_across_provided_splits"] = int((prov[pi] != prov[pj]).sum())
    return a

audit_ker = audit(ker, "patient_id", ker_pairs, ker_hist, "kermany_v3")
print(json.dumps(audit_ker, indent=2))

# %%
# Split by group (plan section 3, items 4-9). `ker` keeps the member-sorted row order throughout.
rng = np.random.default_rng(SEED)
maj = ker.groupby("group")["class"].agg(lambda s: s.value_counts().sort_index().idxmax())   # ties: alphabetical first
part_of = {}
for c in CLASSES:
    gs = np.array(sorted(maj.index[maj == c])); rng.shuffle(gs)
    n = len(gs)
    assert n >= 10, f"class {c} has only {n} groups: too few for a 70/10/20 split"
    n_te = max(1, int(round(FRACTIONS[2] * n))); n_va = max(1, int(round(FRACTIONS[1] * n))); n_tr = n - n_te - n_va
    for g in gs[:n_tr]: part_of[g] = "train"
    for g in gs[n_tr:n_tr + n_va]: part_of[g] = "val"
    for g in gs[n_tr + n_va:]: part_of[g] = "test"
ker["partition"] = ker["group"].map(part_of)
assert ker["partition"].notna().all()
cover = ker.groupby(["partition", "class"]).size().unstack(fill_value=0)
assert (cover.reindex(index=["train", "val", "test"], columns=CLASSES, fill_value=0) > 0).all().all(), f"a partition lacks a class:\n{cover}"

# Zero-crossing assertions (plan section 3, item 8)
for col in ("group", "patient_id", "sha256", "pixel_sha256"):
    assert (ker.groupby(col)["partition"].nunique() == 1).all(), f"a {col} value is in two partitions"
_p = ker["partition"].to_numpy(); _pi, _pj, _pd0 = ker_pairs
_link = _pd0 <= GROUP_HAMMING
assert (_p[_pi[_link]] == _p[_pj[_link]]).all(), "a pair with a linking pHash distance crosses partitions"

# Capped training set: at most TRAIN_CAP images per patient-class pair, by sorted file name (ties by member path)
order = ker.sort_values(["patient_id", "class", "file_name", "member"])
rank = order.groupby(["patient_id", "class"]).cumcount()
ker["in_train_capped"] = (ker["partition"] == "train") & (rank.reindex(ker.index) < TRAIN_CAP)

# Attribution test list (plan section 3, item 7): one image per test patient per class (first by sorted file name),
# then at most ATTR_PER_CLASS patients per class by seeded shuffle (the plan's seed). The group id is kept: the group
# is the resampling unit of the bootstrap (item 9).
first = (ker[ker["partition"] == "test"].sort_values(["patient_id", "class", "file_name", "member"])
         .groupby(["patient_id", "class"], as_index=False).first())
rng_attr = np.random.default_rng(SEED)
attr_parts, test_patients_per_class = [], {}
for c in CLASSES:
    fc = first[first["class"] == c]
    pats = np.array(sorted(fc["patient_id"])); test_patients_per_class[c] = int(len(pats))
    rng_attr.shuffle(pats)
    attr_parts.append(fc[fc["patient_id"].isin(pats[:ATTR_PER_CLASS])])
attr = pd.concat(attr_parts).sort_values(["class", "patient_id"]).reset_index(drop=True)
assert not attr.duplicated(["patient_id", "class"]).any()
ker["in_attribution_test"] = ker["member"].isin(attr["member"])
assert ker["in_attribution_test"].sum() == len(attr) and set(ker.loc[ker["in_attribution_test"], "partition"]) <= {"test"}

split_summary = {
    "images_per_partition_and_class": cover.to_dict(),
    "patients_per_partition_and_class": ker.groupby(["partition", "class"])["patient_id"].nunique().unstack(fill_value=0).to_dict(),
    "groups_per_partition": ker.groupby("partition")["group"].nunique().to_dict(),
    "capped_training_images_per_class": ker[ker["in_train_capped"]]["class"].value_counts().to_dict(),
    "test_patients_per_class_before_the_250_cap": test_patients_per_class,
    "attribution_test_images_per_class": attr["class"].value_counts().to_dict(),
    "attribution_test_patients": int(attr["patient_id"].nunique()), "attribution_test_groups": int(attr["group"].nunique()),
    "attribution_patients_contributing_more_than_one_class": int((attr.groupby("patient_id")["class"].nunique() > 1).sum()),
    "classes_with_fewer_than_100_attribution_patients": [c for c in CLASSES if (attr["class"] == c).sum() < 100],
}
print(json.dumps(split_summary, indent=2))

# %%
# OCTDL: inventory, duplicate audit, groups, and the exploratory DME/NORMAL test list (no training on OCTDL)
lab = pd.read_csv(paths["OCTDL_labels.csv"], dtype=str, keep_default_na=False)
print("OCTDL label columns:", list(lab.columns), "| rows:", len(lab))
assert {"file_name", "disease", "patient_id"} <= set(lab.columns)
for col in ("file_name", "disease", "patient_id"):
    blank = lab[col].str.strip().isin(["", "nan", "NaN", "None"])
    assert not blank.any(), f"OCTDL labels: {int(blank.sum())} blank values in {col}"
lab["stem"] = lab["file_name"].str.strip().str.replace(r"\.(jpe?g|png)$", "", regex=True, case=False).str.lower()
assert lab["stem"].is_unique, "OCTDL file_name values are not unique"
row_of = lab.set_index("stem").to_dict("index")

oct_names, oct_tree = archive_overview(paths["OCTDL.zip"])
print(pd.DataFrame(oct_tree).to_string(index=False))
oct_todo, seen, unmatched = [], collections.Counter(), []
for m in oct_names:
    base = m.replace("\\", "/").split("/")[-1]
    if base.startswith("._") or "__MACOSX" in m or not base.lower().endswith(IMG_EXT):
        continue
    stem = os.path.splitext(base)[0].lower()
    seen[stem] += 1
    r = row_of.get(stem)
    if r is None:
        unmatched.append(m); continue
    oct_todo.append((m, {"file_name": base, "class": r["disease"].strip().upper(), "patient_id": r["patient_id"].strip()}))
# The image-label join must be one-to-one in both directions
assert not unmatched, f"{len(unmatched)} OCTDL images have no label row: {unmatched[:5]}"
assert max(seen.values()) == 1, f"image stems that occur more than once: {[s for s, c in seen.items() if c > 1][:5]}"
missing = sorted(set(row_of) - set(seen))
assert not missing, f"{len(missing)} label rows have no image: {missing[:5]}"
assert len(oct_todo) == len(lab)

octdl = inventory(paths["OCTDL.zip"], oct_todo, "octdl_inventory.csv").sort_values("member").reset_index(drop=True)
# OCTDL labels normal scans "NO" (found in run 3, 2026-11-19: the exploratory test had no normal images). Map to the plan's NORMAL;
# any label outside the published seven classes stops the run.
OCTDL_LABELS = {"AMD": "AMD", "DME": "DME", "ERM": "ERM", "NO": "NORMAL", "RAO": "RAO", "RVO": "RVO", "VID": "VID"}
assert set(octdl["class"]) <= set(OCTDL_LABELS), f"unexpected OCTDL labels: {sorted(set(octdl['class']) - set(OCTDL_LABELS))}"
octdl["class"] = octdl["class"].map(OCTDL_LABELS)
assert {"DME", "NORMAL"} <= set(octdl["class"]), "the exploratory test needs both DME and NORMAL images"
octdl["group"], oct_pairs, oct_hist, prof_oct = group_dataset(octdl, "patient_id", "octdl")
profiles += prof_oct
audit_oct = audit(octdl, "patient_id", oct_pairs, oct_hist, "octdl_v4")
print(json.dumps(audit_oct, indent=2))

# Cross-source check, with identities saved. Deviation D1: an OCTDL image identical or near-identical to any Kermany
# image is excluded from the exploratory external test list.
hk = np.array([int(x, 16) for x in ker["phash"]], dtype=np.uint64); ho = np.array([int(x, 16) for x in octdl["phash"]], dtype=np.uint64)
ker_file, ker_pix = set(ker["sha256"]), set(ker["pixel_sha256"])
cross_rows = []
for s in range(0, len(ho), 64):
    d = popcount64(ho[s:s + 64, None] ^ hk[None, :])
    for a, b in zip(*np.nonzero(d <= HAMMING_MAX)):
        cross_rows.append({"octdl_member": octdl["member"].iat[s + a], "kermany_member": ker["member"].iat[b], "hamming": int(d[a, b]),
                           "kermany_partition": ker["partition"].iat[b]})
cross = pd.DataFrame(cross_rows, columns=["octdl_member", "kermany_member", "hamming", "kermany_partition"])
octdl["matches_kermany"] = octdl["member"].isin(cross.loc[cross["hamming"] <= GROUP_HAMMING, "octdl_member"]) | octdl["sha256"].isin(ker_file) | octdl["pixel_sha256"].isin(ker_pix)
print("OCTDL images that match a Kermany image (identical or near-duplicate):", int(octdl["matches_kermany"].sum()))

# Exploratory test list: DME and NORMAL, not matching Kermany, one image per patient per class (first by file name)
sub = octdl[octdl["class"].isin(["DME", "NORMAL"]) & ~octdl["matches_kermany"]]
octdl_test = (sub.sort_values(["patient_id", "class", "file_name", "member"]).groupby(["patient_id", "class"], as_index=False).first()
              .sort_values(["class", "patient_id"]).reset_index(drop=True))
assert not octdl_test.duplicated(["patient_id", "class"]).any()
octdl["in_exploratory_test"] = octdl["member"].isin(octdl_test["member"])

# %%
# Publish derived metadata (no images) and the frozen split with its hash
cols = ["member", "file_name", "class", "patient_id", "image_no", "provided_split", "sha256", "pixel_sha256", "phash", "width", "height",
        "group", "partition", "in_train_capped", "in_attribution_test"]
# Look-alike pairs that the grouping rule does NOT link (pHash distance 1-4, different groups): unverified candidates, reported
# by partition pair so the zero-crossing claim can be qualified (Codex review 002, point 1). Not used for any decision.
_pi, _pj, _pd = ker_pairs
_g, _part = ker["group"].to_numpy(), ker["partition"].to_numpy()
_res = (_pd > GROUP_HAMMING) & (_g[_pi] != _g[_pj])
residual = pd.DataFrame({"hamming": _pd[_res], "partitions": ["-".join(sorted(x)) for x in zip(_part[_pi[_res]], _part[_pj[_res]])]})
residual_tab = residual.groupby(["hamming", "partitions"]).size().rename("pairs").reset_index()
octdl_test_members = set(octdl_test["member"])
cr = cross[(cross["hamming"] > GROUP_HAMMING) & cross["octdl_member"].isin(octdl_test_members)]
octdl_residual = {"pairs_with_kermany_train_by_distance": cr[cr["kermany_partition"] == "train"]["hamming"].value_counts().sort_index().to_dict(),
                  "octdl_test_images_with_any_kermany_train_pair_at_distance_1_to_4": int(cr[cr["kermany_partition"] == "train"]["octdl_member"].nunique()),
                  "octdl_test_images": int(len(octdl_test_members))}
print("Unlinked look-alike pairs (unverified) by distance and partition pair:")
print(residual_tab.to_string(index=False))
print("OCTDL exploratory test vs Kermany train, unlinked look-alikes:", octdl_residual)
save_output(residual_tab, "kermany_unlinked_lookalike_pairs_by_partition.csv",
            "Kermany pairs at pHash distance 1-4 that are not in the same group, counted by distance and partition pair: unverified look-alike candidates, not confirmed duplicates (deviation D5)", "publish")
p = save_output(ker[cols], "kermany_split.csv", "Kermany v3: one row per OCT image with patient, hashes, group, partition, capped-train and attribution flags (the frozen split)", "publish")
split_sha = sha256_file(p)
save_output(attr[["member", "class", "patient_id", "group"]], "attribution_test_list.csv", "Fixed attribution test list: one image per test patient per class, at most 250 patients per class, with group id", "publish")
save_output(octdl[["member", "file_name", "class", "patient_id", "sha256", "pixel_sha256", "phash", "width", "height", "group", "matches_kermany", "in_exploratory_test"]],
            "octdl_inventory.csv", "OCTDL v4: one row per image with patient, hashes, group, Kermany-match flag and exploratory-test flag", "publish")
save_output(cross, "cross_source_matches.csv", "OCTDL-Kermany image pairs with pHash distance 0 to 4 (only distance 0, identical bytes or identical pixels exclude an OCTDL image; deviation D5), with the Kermany partition", "octdl")
save_output(pd.DataFrame(profiles), "group_threshold_sensitivity.csv", "Group counts, largest group (images, patients) and mixed groups for identity links only and for pHash thresholds 0 to 4", "grouping")
save_output(pd.DataFrame(ker_tree + oct_tree), "archive_folder_overview.csv", "Files per folder and extension in the two raw archives (evidence that no OCT folder was skipped)", "inventory")
_mem, _pat = ker["member"].to_numpy(), ker["patient_id"].to_numpy()
pairs_df = pd.DataFrame({"member_a": _mem[_pi], "member_b": _mem[_pj], "hamming": _pd, "different_patient": _pat[_pi] != _pat[_pj]})
between = pairs_df[pairs_df["different_patient"]].sort_values(["hamming", "member_a", "member_b"])
save_output(between.head(200000), "kermany_near_duplicate_pairs_between_patients.csv",
            f"Kermany near-duplicate pairs (pHash distance <= 4) whose images carry different patient IDs: first 200,000 of {len(between)}", "publish")
save_output({"seed": SEED, "hamming_max": HAMMING_MAX, "group_hamming": GROUP_HAMMING, "max_group_share": MAX_GROUP_SHARE, "parser_version": PARSER_VERSION,
             "fractions_train_val_test": FRACTIONS, "train_cap": TRAIN_CAP, "attribution_patients_per_class": ATTR_PER_CLASS,
             "kermany_split_csv_sha256": split_sha, "phash_probe": PHASH_PROBE, "kermany": audit_ker, "split": split_summary, "octdl": audit_oct,
             "octdl_images_matching_kermany": int(octdl["matches_kermany"].sum()),
             "kermany_unlinked_lookalike_pairs_by_partition_pair": residual.groupby("partitions").size().to_dict(),
             "octdl_test_unlinked_lookalikes_vs_kermany_train": octdl_residual,
             "octdl_exploratory_test_images_per_class": octdl_test["class"].value_counts().to_dict(),
             "raw_files": raw_rows, "python": platform.python_version(), "packages": VERSIONS,
             "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
            "audit_summary.json", "Audit and split summary: counts, leakage findings, partition sizes, split file hash, raw file hashes, versions", "publish")
print("kermany_split.csv sha256:", split_sha)
