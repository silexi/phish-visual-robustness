# Robustness of Visual Representations for Phishing Webpages

Code, prompts, analysis scripts, experiment settings and result files for

> B. Kılıç and B. Çeliktaş, *Robustness of Visual Representations for Phishing
> Webpages: A Measurement Study from Classical Similarity Metrics to
> Vision-Language Models*, submitted to the Journal of Network and Computer
> Applications, 2026.

The study measures how well eight visual representations of a phishing page
(colour histogram, 8×8 thumbnail, global SSIM, multi-resolution perceptual
hash, LBP, HOG, a weighted fusion of the pixel metrics, and a CLIP embedding)
keep retrieving the page's own kit cluster after the page is re-rendered under
one of six CSS/DOM perturbations, and whether a vision-language model
(Qwen3-VL-8B) still recovers the brand, page type and credential-seeking
intent of the same renders. The unit of analysis is the phishing kit cluster,
not the page, and every number in the paper is reproducible from the files in
this repository plus the public corpus named below.

Everything is released under the MIT licence (see `LICENSE`). Please cite the
paper (see `CITATION.cff`).

## What is and is not in this repository

Included:

| directory / file | content |
|---|---|
| `*.py`, `*.sh` (repository root) | the complete pipeline, in the flat layout the scripts expect (see *Pipeline*) |
| `pixel/` | the pixel-tier comparator from the authors' earlier conference work, byte-identical (`config.py` holds the fusion weights and hash/LBP/HOG settings) |
| `vlm/PROMPT.txt` | the exact prompt given to Qwen3-VL (also embedded in `describe.py`) |
| `vlm/desc_*.jsonl` | raw model outputs, one JSON record per render, for the clean baseline (`A0`), every perturbation and the legitimate pages (`BENIGN`); e-mail addresses and URLs that the model transcribed from phishing pages are replaced by `[email redacted]` / `[url redacted]`, nothing else is altered |
| `results_*.json` | every result table in the paper (`EXPERIMENT_SETTINGS.md` maps tables and figures to files and lists every parameter) |
| `ids_*.txt`, `clusters_*.json` | the page ids that survive each gate and the kit-cluster assignment; ids are 16-hex-digit hashes of the archived HTML, they do not encode the source URL |
| `pert/manifest_*.jsonl`, `fidelity_f.jsonl` | per-page efficacy of each perturbation (`applied`) and the render-fidelity covariates |
| `benign_urls*.txt`, `benign/fetch_manifest.jsonl`, `benign_replay/manifest_replay.jsonl`, `benign_*_manifest.json` | the legitimate login pages: the URL list, the capture log, the replay log and the brand-disjoint reference / calibration split |
| `audit*.py`, `verify_*.py` | the audit scripts written while checking the numbers; they re-derive the tables from the raw files and are kept so a reader can repeat the checks |
| `figures/` | `make_figures.py` and the inputs it needs regenerate the four figures of the paper into `figures/out/`. `figures/input/panel/` holds the six renders (clean and five perturbations) of the single phishing page that Figures 1 and 4 of the paper show; they are the only phishing renders in the repository, the page carries no victim data, and its kit's servers have been offline since 2024 |
| `requirements.txt`, `requirements-freeze.txt` | pinned dependencies (direct / complete) |

Deliberately **not** included:

* **Phishing renders, phishing HTML and the parquet shards** (except the one
  figure page named above). The corpus is the
  public Hugging Face dataset `nyuuzyou/phishing-snapshots`; every page used
  here is identified by its `safe` id, which is derived from the archived HTML
  and can be recomputed by `render.py` from the shards. Redistributing the
  renders or the HTML would redistribute live credential-harvesting kits,
  including any victim data pre-filled in them.
* **Phishing URLs and domains.** The per-shard render manifests
  (`corpus/manifest_s*.jsonl`) contain the source domains and are therefore
  withheld; the counts derived from them are stated in
  `EXPERIMENT_SETTINGS.md` and in Table 1 of the paper. No phishing URL was
  fetched live at any point of the study; every phishing page was rendered
  from the archived HTML with all external subresources blocked.
* **Feature caches (`feat_*.npz`) and the VLM weights.** Both are regenerated
  by the pipeline; the model checkpoint is `mlx-community/Qwen3-VL-8B-Instruct-8bit`
  from the Hugging Face hub.

## Environment

The reported runs were made on one machine: macOS 26.2 (build 25C56), Apple
M4 Pro, 64 GB, Python 3.12.13 in a virtual environment at `./venv`,
Playwright 1.63.0 with its bundled Chromium (builds 1223 and 1243 were
installed; the renders used the `chromium` channel with
`--disable-lcd-text --hide-scrollbars`). The semantic rung needs Apple
Silicon because it runs through `mlx-vlm 0.7.2`; every other stage is plain
CPU Python and should run anywhere the pinned packages install.

```bash
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt        # or requirements-freeze.txt
./venv/bin/playwright install chromium
```

The shell drivers call `./venv/bin/python` relative to the repository root and
`cd` into the script's own directory, so the repository must be run from its
root with the environment at `./venv`.

## Pipeline

The scripts are kept in the flat working-directory layout they were written
for (they import each other with `sys.path.insert(0, <own dir>)` and read data
files relative to the repository root). In order:

| stage | script | what it does |
|---|---|---|
| 1 download | `dl_shards.py`, `dl_retry.py` | pulls parquet shards 1–19 of `nyuuzyou/phishing-snapshots` and keeps the records with `has_password_field ∧ http_status == 200 ∧ html_length > 3000`; shard 0 was downloaded whole and filtered by `render.py` under the same predicate |
| 2 render | `render.py` via `worker.sh` (`--limit 700`) | renders the first 700 candidates of each shard in file order under the fixed capture protocol (1366×768, DPR 1, light, en-US, UTC, `domcontentloaded` + 1200 ms settle, 15 s navigation timeout, external subresources aborted) and writes one manifest row per attempt with the content covariates |
| 3 content gate | `gate_content.py` → `ids_gated_all.txt` | keeps renders that succeeded, show a visible password field, have more than 60 characters of text and at least 10 quantised colours. The gate was run when 13 of the 20 shards had finished rendering and the other 7 were never gated (a timing accident, disclosed in the paper); the script defaults to those 13 shards so that it reproduces the released id list, `--shards all` gives the 20-shard counts, `--check ids_gated_all.txt` compares |
| 4 re-render + fidelity gate | `replay_ids.py`, `fidelity.py` → `ids_f.txt` (the same 834 ids as `ids_fid_full.txt`, in sorted order; the first 200 lines of the two files share only 51 ids) | re-renders the gated pages over `file://` (so both classes take the identical path) and keeps the 834 pages whose solid-colour block area is at least 2 % of the canvas. `ids_f.txt` has no trailing newline; the shell loop in `pipeline3.sh` stage 6 therefore skipped its last id, and every result table has denominator 833 (kept as it was; see `EXPERIMENT_SETTINGS.md`) |
| 5 kit clustering | `dedup.py --maxdist 16` → `clusters_f.json`, `clusters_flat.json` | clusters the surviving renders by 256-bit difference hash, Hamming ≤ 16; the cluster id is carried through every later step |
| 6 perturb | `perturb.py --cond {A0n,A1,A3,A4,S1,S3,S7}` → `pert/` | re-renders each page with one CSS/DOM injection and records per page whether the injection changed the pixels (`applied`) |
| 7 legitimate side | `benign_fetch.py`, `render_html.py`, `split_benign.py` | captures the legitimate login pages once each (declared research user agent, robots.txt honoured, no form submitted), freezes them to HTML, replays them under the same protocol, and splits them brand-disjointly into a reference and a calibration set |
| 8 features | `features.py` → `feat_*.npz` | pixel-tier features through `pixel/` (unmodified) plus CLIP ViT-B/32 (`open_clip`, `openai` weights) |
| 9 primary analysis | `retention2.py --clean feat_A0.npz --conds A0n=feat_A0n.npz … S7=feat_S7.npz --clusters clusters_flat.json --applied pert --eps 1e-4 --out results_retention_v2.json` | rank-1 retention of the own kit cluster per configuration and condition (tie-tolerant hit rule, eps 1e-4), cluster-weighted, applied-only and intent-to-treat, with a kit-level bootstrap (B = 4000, seed 20260922) |
| 10 exclusion analysis | `retention_excl.py` → `results_retention_excl.json` | the same rule on five galleries: unchanged (`full`), the query's own clean render removed (`self_excl`), every byte-identical clean render removed (`dup_excl`), and two sensitivity galleries (`rmse1_excl`, `dhash0_excl`); each variant is also scored against the full gallery on the surviving population, with paired exclusion-minus-full and HOG-minus-CLIP contrasts |
| 10b paired contrasts | `paired_contrasts.py` → `results_paired_contrasts.json` | the two paired differences quoted in the text on the full gallery (HOG − CLIP under A3, LBP − fusion under A4): per-page hit differences averaged per cluster and over clusters, applied-only, same hit rule and bootstrap as stage 9 |
| 11 perceptibility | `perceptibility.py` → `results_perceptibility.json` | SSIM, mean ΔE2000 and the fraction of pixels above one JND between each perturbed render and its clean counterpart |
| 12 semantic rung | `describe.py` via `vlm_batch.sh`, `rerun_a1s3.sh`, `run_s3.sh`, `run_benign_vlm.sh` → `vlm/desc_*.jsonl`; `vlm_retention.py` → `results_vlm_v2.json` | one structured description per render from Qwen3-VL, then brand / page-type / credential-intent retention against the clean description |
| 12b misattribution | `misattribution.py` → `results_misattribution.json` | the `y.atıf` column of Table 6 (a perturbed answer naming a brand other than the clean one, or any brand on a page unnamed in the clean run; denominator 196, 195 for A1/S3) with its named/unnamed split and the conditional count on the brand-eligible pages |
| 13 credential-intent detector | `f1_recheck.py` → `results_f1_recheck.json` | the confusion matrix of `asks_for_credentials` (positive class phishing) on clean phishing renders against the replayed legitimate pages, under four definitions of the negative set (model-labelled login pages; all parsed pages; visible password field in the replayed render; the same flag from the live capture) plus two parse-failure sensitivity checks, the always-positive baseline and cluster-weighted recall |
| 14 detector operating point | `score.py` → `results_pixel_clip.json` | every configuration read at a threshold calibrated on the legitimate calibration split at 5 % FPR (secondary analysis) |

`pipeline3.sh` chains stages 4–8 and 14 exactly as they were run for the
reported numbers; `pipeline.sh` and `pipeline2.sh` are the two earlier chains
and are kept for provenance. The corpus funnel (136,414 archive records →
100,000 downloaded → 25,646 after the record filter → 12,119 after the
per-shard cap → 10,520 rendered → 6,966 entering the content gate → 1,702 →
834 → 172 kit clusters) is tabulated with every predicate in
`EXPERIMENT_SETTINGS.md`.

## Reproducing the tables from the released files

The analysis stages (9–13) read only files that are in this repository except
for the feature caches, which stage 8 regenerates from the renders. Without
the renders the following still run as-is and reproduce the paper's numbers:

```bash
./venv/bin/python vlm_retention.py --clean vlm/desc_A0.jsonl \
    --conds A0n=vlm/desc_A0n.jsonl A1=vlm/desc_A1.jsonl A3=vlm/desc_A3.jsonl \
            A4=vlm/desc_A4.jsonl S1=vlm/desc_S1.jsonl S3=vlm/desc_S3.jsonl S7=vlm/desc_S7.jsonl \
    --clusters clusters_flat.json --out results_vlm_v2.json
./venv/bin/python f1_recheck.py
./venv/bin/python misattribution.py
./venv/bin/python figures/make_figures.py
```

Stages 9–11 need the feature caches and the renders; once stage 8 has been
re-run they reproduce `results_retention_v2.json`,
`results_retention_excl.json`, `results_paired_contrasts.json` and
`results_perceptibility.json` with the
commands listed in `EXPERIMENT_SETTINGS.md`.

## Ethics

Phishing pages were never contacted; they were rendered from archived HTML
with network access to anything but the local file blocked. Legitimate pages
were fetched once each with a declared user agent
(`AcademicPhishingRobustnessStudy/1.0`), robots.txt was honoured, requests
were rate-limited, and no form was ever submitted or credential entered. The
contact address for that user agent is the corresponding author's, given in
the paper.

## Citation

See `CITATION.cff`. Until the article is published, cite the repository
release (a Zenodo DOI for the tagged version is given in the paper's data
availability statement).
