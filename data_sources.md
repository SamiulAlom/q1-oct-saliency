# Data sources: paper-01 — checkpoint 3

Written 2026-11-19 by Taha. Nothing has been downloaded. Facts marked "verified" were read on the official Mendeley Data pages in the browser on 2026-11-19; everything else is marked.

## 1. Kermany OCT (primary: training, validation, test)
- **Name:** Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images, version 3.
- **Owner:** D. Kermany, K. Zhang, M. Goldbaum (University of California San Diego; Guangzhou Women and Children's Medical Center).
- **URL / DOI:** https://data.mendeley.com/datasets/rscbjbr9sj/3 ; doi:10.17632/rscbjbr9sj.3 (verified). Published 2 June 2018; versions 1 and 2 exist (verified). Version 3 is used and nothing else.
- **Licence:** CC BY 4.0 (verified). The description adds "made available for use in research only" (verified). Our use is research; the paper cites the dataset and the article.
- **Access level:** open, no account (verified: "Download All 7.85 GB").
- **Citation required:** Kermany et al., Cell 2018;172(5):1122-1131, doi:10.1016/j.cell.2018.02.010 (verified on the dataset page).
- **Content:** OCT B-scans in four class folders, CNV, DME, DRUSEN, NORMAL; file names are `(disease)-(randomized patient ID)-(image number by this patient)` (verified in the description). The archive also holds chest X-rays, which are not used.
- **What we use:** the OCT images, the class from the folder, the patient ID from the file name.
- **Not verified:** image count, number of patients, image sizes, scanner model, exact archive file names. Notebook 01 records them.
- **Known issues:**
  - The provided train/test folders overlap at subject and image level in version 2 (Tampu 2022, doi:10.1038/s41597-022-01618-6). Our split ignores the provided folders.
  - Patient IDs are reported to be shared across class folders (Jaman 2026, doi:10.5120/ijca8fee57096b16, abstract only). Notebook 01 tests this.
  - Single source and one scanner type, as far as described; adults; no demographic fields.
- **Changes over time:** versioned DOI; version 3 is fixed. The notebook records the version and file hashes at the start and end of the download.

## 2. OCTDL (external, exploratory scanner-shift analysis only; first to be cut if time is short)
- **Name:** OCTDL: Optical Coherence Tomography Dataset for Image-Based Deep Learning Methods, version 4.
- **Owner:** M. Kulyabin and others (Friedrich-Alexander-Universität Erlangen-Nürnberg; Ural Federal University; Ural State Medical University).
- **URL / DOI:** https://data.mendeley.com/datasets/sncdhf53xc/4 ; doi:10.17632/sncdhf53xc.4 (verified). Published 19 March 2024.
- **Licence:** CC BY 4.0 (verified). **Access:** open, 380 MB (verified).
- **Content (verified on the page):** AMD 1,231; DME 147; ERM 155; NORMAL 332 (labelled `NO` in the CSV, found in the run of 2026-11-19); RAO 22; RVO 101; VID 76 images. CSV columns: file_name, disease, subcategory, condition, patient_id, eye, sex, year, image_width, image_height. The page says the dataset "will be updated periodically": version 4 is pinned.
- **What we use:** DME and NORMAL images only, with patient_id, for testing without any training.
- **Citation:** the dataset page gives the preprint arXiv:2312.08255. A peer-reviewed data descriptor is believed to exist (Scientific Data, 2024) `[CHECK: DOI not yet verified]`; preprints may not be cited under the house citation rule, so the descriptor's DOI must be confirmed before drafting, or the dataset DOI is cited alone.
- **Known issues:** 47 pairs of pixel-identical images with different patient IDs are reported (Jaman 2026, abstract only). Notebook 01 tests this and groups duplicates. Different scanner from Kermany (reported as Optovue Avanti; not verified by us). Small DME sample (147 images before one-per-patient selection).

## How the data will be handled
- **Where:** downloaded inside the Colab runtime by notebook `01_audit_split_CPU`, never to this PC. Raw images stay on the runtime disk (and Drive only if a later run needs them cached). Raw data is never committed to git.
- **Download route:** Mendeley Data's public file links for the pinned versions `[CHECK: exact file URLs are resolved and written into the notebook, then into data/raw/MANIFEST.md]`. If Mendeley blocks automated download, I stop and ask you to fetch the archive in your browser; I do not work around a block.
- **Integrity:** SHA-256 of every downloaded archive and a per-image hash list go to `data/raw/MANIFEST.md` and `outputs/`; dataset version read at the start and end of the download.
- **Outputs of notebook 01 (only derived metadata, no images):** audit table (images, patients per class, IDs shared across classes, duplicate pairs), group membership file, split file with SHA-256, the fixed attribution test list.
- **Privacy and ethics:** both sets are public and de-identified, with randomised patient IDs. No new human data; no ethics approval needed for this secondary use. The paper states the licences and that data were used as released.

## One-time steps for the human
None: no account, terms or approval is needed for either dataset.

## Decision asked at checkpoint 3
Approve Kermany v3 as the primary dataset and OCTDL v4 as the exploratory external set, used as described.
