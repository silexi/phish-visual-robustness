# Robustness of Visual Representations for Phishing Webpages

Code, prompts, analysis scripts, experiment settings and result files for

> B. Kılıç and B. Çeliktaş, *Robustness of Visual Representations for Phishing
> Webpages: A Measurement Study from Classical Similarity Metrics to
> Vision-Language Models*, submitted to the Journal of Network and Computer
> Applications, 2026.

The study measures how well eight visual representations of a phishing page
(colour histogram, 8×8 thumbnail, global SSIM, multi-resolution perceptual
hash, LBP, HOG, a weighted fusion of the classical metrics, and a CLIP embedding)
keep retrieving the page's own kit cluster after the page is re-rendered under
one of six CSS/DOM perturbations, and whether a vision-language model
(Qwen3-VL-8B) still recovers the brand, page type and credential-seeking
intent of the same renders. The unit of analysis is the phishing kit cluster,
not the page, and every number in the paper is reproducible from the files in
this repository plus the public corpus named below.

The reported measurement covers **1,206 pages in 248 kit clusters**. A smaller
measurement of the same protocol — 833 pages in 172 clusters — was released as
v1.0.0 and is kept unchanged in `run1_833/`; see *The first measurement* below.

Everything is released under the MIT licence (see `LICENSE`). Please cite the
paper (see `CITATION.cff`).

## What is and is not in this repository

Included:

| directory / file | content |
|---|---|
| `*.py`, `*.sh` (repository root) | the complete pipeline, in the flat layout the scripts expect (see *Pipeline*) |
| `pixel/` | the pixel-tier comparator from the authors' earlier conference work, byte-identical (`config.py` holds the fusion weights and hash/LBP/HOG settings) |
| `vlm/PROMPT.txt` | the exact prompt given to Qwen3-VL (also embedded in `describe.py`) |
| `vlm/desc_*.jsonl` | raw model outputs, one JSON record per render, for the clean baseline (`A0`), every perturbation and the legitimate pages (`BENIGN`); e-mail addresses and URLs that the model transcribed from phishing pages are replaced by `[email redacted]` / `[url redacted]`, nothing else is altered. The legitimate-page file is not redacted: what it transcribes is the brands' own public content. Replacing every address and URL in all nine files leaves `results_vlm_v2.json`, `results_misattribution.json` and `results_f1_recheck.json` bit-identical, so the redaction cannot move a published number |
| `results_*.json` | every result table in the paper (`EXPERIMENT_SETTINGS.md` maps tables and figures to files and lists every parameter) |
| `ids_*.txt`, `clusters_*.json` | the page ids that survive each gate and the kit-cluster assignment; ids are the first 16 hex digits of the SHA-256 of the dataset's `id` column (`safe` in `render.py`); they are one-way and do not encode the source URL or domain |
| `pert/manifest_*.jsonl`, `fidelity_u.jsonl` | per-page efficacy of each perturbation (`applied`, 1,206 rows per condition) and the render-fidelity covariates of all 2,359 re-rendered pages, the 1,153 the gate rejected included |
| `benign_urls*.txt`, `benign/fetch_manifest.jsonl`, `benign_replay/manifest_replay.jsonl`, `benign_*_manifest.json` | the legitimate login pages: the URL list, the capture log, the replay log and the brand-disjoint reference / calibration split |
| `run1_833/` | the derived data of the 833-page / 172-cluster measurement released as v1.0.0, unchanged: its result files, id lists, cluster files, perturbation manifests, fidelity covariates, 200-page VLM subset, audit metadata and the 23 `audit*.py` / `verify_*.py` scripts written against it, with its own `README.md`. One file in it is new rather than moved, `audit_perc.json`, the first run's own `audit_perc.py` output of 22 September 2026, which v1.0.0 did not carry |
| `run1_833/audit*.py`, `run1_833/verify_*.py` | the audit scripts written while checking the numbers of the **first** measurement; they re-derive its tables from its raw files and are kept so a reader can repeat the checks. They hard-code that run's population (833 pages, 172 clusters) and they sit beside that run's data rather than at the repository root; they were deliberately not re-pointed at the 1,206-page measurement. What each of the 23 can and cannot do from the released files was measured, script by script, and is listed in `EXPERIMENT_SETTINGS.md` Section 13: 3 reproduce their result, a 4th needs the legitimate-side files that stay at the root, `audit_perc.py` runs but writes an empty result because it compares renders, and the other 18 need the feature caches or the renders |
| `audit_u_*.py`, `audit_redo_*.json` | the independent statistical audit of the **reported** 1,206-page measurement, which the paper's audit subsection cites: the 10 scripts that were run and the 12 JSON files they wrote. Each script takes `u` (the 1,206-page run) or `f` (the 833-page run, as a cross-check) and writes `audit_redo_*_<u\|f>.json`, so every figure quoted in that subsection can be read out of a file instead of being recomputed. They are released as they were run, which means they carry the absolute path of the machine they ran on and they read the feature caches; see `EXPERIMENT_SETTINGS.md` Section 15 for what that implies |
| `vlm_gizlilik_tarama.py` | the scan behind the redaction of `vlm/desc_*.jsonl`: it lists every e-mail- and URL-shaped string in those files and classifies each as a reserved-domain placeholder (RFC 2606 / RFC 6761) or as a string a human has to rule on. Both patterns are matched case-insensitively, because the model transcribes the page's own capitalisation. The acceptance condition the released files meet is that no string in the eight phishing files awaits a ruling (`EXPERIMENT_SETTINGS.md` Section 16) |
| `figures/` | `make_figures.py` and the inputs it needs regenerate the four figures of the paper into `figures/out/`. `figures/input/panel/` holds the six renders (clean and five perturbations) of the single phishing page that Figures 1 and 4 of the paper show; they are the only phishing renders in the repository, the page carries no victim data, and its kit's servers have been offline since 2024 |
| `requirements.txt`, `requirements-freeze.txt` | pinned dependencies (direct / complete) |

Deliberately **not** included:

* **Phishing renders, phishing HTML and the parquet shards** (except the one
  figure page named above). The corpus is the
  public Hugging Face dataset `nyuuzyou/phishing-snapshots`; every page used
  here is identified by its `safe` id, which is the SHA-256 prefix of the
  dataset's record `id` and can be recomputed by `render.py` from the shards. Redistributing the
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

The 1,206-page gallery mixes renders of two dates: the 834 pages that the
first measurement also used were re-rendered on 22 September 2026 and are
re-used as they are, and the 372 pages that the enlarged content gate added
were rendered on 3 October 2026. Playwright and its bundled Chromium were
installed on 22 September 2026 and never updated; determinism across the two
dates was measured, not assumed (60 pages re-rendered in October, 59
byte-identical, the one exception differing in 34 of 1,049,088 pixels at an
amplitude of at most 2 grey levels). `EXPERIMENT_SETTINGS.md` Section 2 gives
the detail.

## Pipeline

The scripts are kept in the flat working-directory layout they were written
for (they import each other with `sys.path.insert(0, <own dir>)` and read data
files relative to the repository root). In order:

| stage | script | what it does |
|---|---|---|
| 1 download | `dl_shards.py`, `dl_retry.py` | pulls parquet shards 1–19 of `nyuuzyou/phishing-snapshots` and keeps the records with `has_password_field ∧ http_status == 200 ∧ html_length > 3000`; shard 0 was downloaded whole and filtered by `render.py` under the same predicate |
| 2 render | `render.py` via `worker.sh` (`--limit 700`) | renders the first 700 candidates of each shard in file order under the fixed capture protocol (1366×768, DPR 1, light, en-US, UTC, `domcontentloaded` + 1200 ms settle, 15 s navigation timeout, external subresources aborted) and writes one manifest row per attempt with the content covariates |
| 3 content gate | `gate_content.py --shards all` → `ids_gated_all.txt` | keeps renders that succeeded, show a visible password field, have more than 60 characters of text and at least 10 quantised colours. The gate runs over all 20 shards and admits 2,359 of the 10,520 successful renders; `--check <list>` compares against an existing id list. In the first measurement the same predicate had been applied to only 13 of the 20 shards, because the other 7 had not finished rendering when the gate was run; that list is kept as `run1_833/ids_gated_all.txt` (1,702 ids), and the 2,359 are a strict superset of it |
| 4 re-render + fidelity gate | `replay_ids.py`, `fidelity.py` → `ids_u.txt` (1,206 ids, sorted) | re-renders the gated pages over `file://` (so both classes take the identical path) and keeps the 1,206 pages whose solid-colour block area is at least 2 % of the canvas. The same 1,206 ids result when the gate is applied to the first-pass renders instead. `ids_u.txt` ends with a newline, so no id is lost to a shell `while read` loop and every result table has denominator 1,206 |
| 5 kit clustering | `dedup.py --maxdist 16` → `clusters_u.json`, `clusters_flat.json` | clusters the surviving renders by 256-bit difference hash, Hamming ≤ 16, into 248 clusters; the cluster id is carried through every later step |
| 6 perturb | `perturb.py --cond {A0n,A1,A3,A4,S1,S3,S7}` → `pert/` | re-renders each page with one CSS/DOM injection and records per page whether the injection changed the pixels (`applied`) |
| 7 legitimate side | `benign_fetch.py`, `render_html.py`, `split_benign.py` | captures the legitimate login pages once each (declared research user agent, robots.txt honoured, no form submitted), freezes them to HTML, replays them under the same protocol, and splits them brand-disjointly into a reference and a calibration set |
| 8 features | `features.py` → `feat_*.npz` | pixel-tier features through `pixel/` (unmodified) plus CLIP ViT-B/32 (`open_clip`, `openai` weights) |
| 9 primary analysis | `retention2.py --clean feat_A0.npz --conds A0n=feat_A0n.npz … S7=feat_S7.npz --clusters clusters_flat.json --applied pert --eps 1e-4 --out results_retention_v2.json` | rank-1 retention of the own kit cluster per configuration and condition (tie-tolerant hit rule, eps 1e-4), cluster-weighted, applied-only and intent-to-treat, with a kit-level bootstrap (B = 4000, seed 20260922) |
| 10 exclusion analysis | `retention_excl.py` → `results_retention_excl.json` | the same rule on five galleries: unchanged (`full`), the query's own clean render removed (`self_excl`), every byte-identical clean render removed (`dup_excl`), and two sensitivity galleries (`rmse1_excl`, `dhash0_excl`); each variant is also scored against the full gallery on the surviving population, with paired exclusion-minus-full and HOG-minus-CLIP contrasts |
| 10b paired contrasts | `paired_contrasts.py` → `results_paired_contrasts.json` | the two paired differences quoted in the text on the full gallery (HOG − CLIP under A3, LBP − fusion under A4): per-page hit differences averaged per cluster and over clusters, applied-only, same hit rule and bootstrap as stage 9 |
| 11 perceptibility | `perceptibility.py` → `results_perceptibility.json` | SSIM, mean ΔE2000 and the fraction of pixels above one JND between each perturbed render and its clean counterpart, on 200 pages drawn with a fixed seed from `ids_u.txt` |
| 12 semantic rung | `describe.py` via `vlm657.sh` → `vlm/desc_{A0,A0n,A1,A3,A4,S1,S3,S7}.jsonl`; the legitimate side via `run_benign_vlm.sh` → `vlm/desc_BENIGN.jsonl`, carried over unchanged from the first measurement; `vlm_retention.py` → `results_vlm_v2.json` | one structured description per render from Qwen3-VL on the first 200 ids of `ids_u.txt`, then brand / page-type / credential-intent retention against the clean description. `vlm_batch.sh`, `rerun_a1s3.sh` and `run_s3.sh` are the first measurement's drivers for this stage and are kept for provenance |
| 12b misattribution | `misattribution.py` → `results_misattribution.json` | the `y.atıf` column of Table 6 (a perturbed answer naming a brand other than the clean one, or any brand on a page unnamed in the clean run; denominator 195) with its named/unnamed split and the conditional count on the brand-eligible pages |
| 13 credential-intent detector | `f1_recheck.py` → `results_f1_recheck.json` | the confusion matrix of `asks_for_credentials` (positive class phishing) on clean phishing renders against the replayed legitimate pages, under four definitions of the negative set (model-labelled login pages; all parsed pages; visible password field in the replayed render; the same flag from the live capture) plus two parse-failure sensitivity checks, the always-positive baseline and cluster-weighted recall |
| 14 detector operating point | `score.py` → `results_pixel_clip.json` | every configuration read at a threshold calibrated on the legitimate calibration split at 5 % FPR (secondary analysis) |

`pipeline657.sh`, `pipeline657b.sh`, `pipeline657c.sh`, `pipeline657d.sh`,
`vlm657.sh`, `anlamsal_u.sh` and `duyarlik.sh` are the chain that was run for
the 1,206-page measurement, in that order, and they are released as the record
of it rather than as a one-shot driver: `pipeline657b.sh` repeats stages 4–6 of
`pipeline657.sh` (stages 0–3 had already finished), steps A and B of
`pipeline657c.sh` were redone by `pipeline657d.sh` from the link directory the
two scripts need, and the released result files were written by the last two
scripts of the chain — `anlamsal_u.sh` and `duyarlik.sh` — which re-ran the
seven analysis scripts at six-decimal rounding. Where a stage was run more than
once, `EXPERIMENT_SETTINGS.md` quotes the invocation that produced the released
file; the one difference that matters is stage 11, which `pipeline657c.sh` ran
over six conditions and 250 pages and `duyarlik.sh` then re-ran over seven
conditions and 200 pages, as `results_perceptibility.json` records.
`pipeline3.sh` chained stages 4–8 and 14 for the first measurement, and
`pipeline.sh` and `pipeline2.sh` are the two earlier chains, all three kept for
provenance. The
`657` scripts carry Turkish comments and are released as they were run rather
than translated, because a rewritten driver is no longer a record of what ran;
`EXPERIMENT_SETTINGS.md` states every command of theirs in English. The corpus
funnel (136,414 archive records → 100,000 downloaded → 25,646 after the record
filter → 12,119 after the per-shard cap → 10,520 rendered → 10,520 entering
the content gate → 2,359 → 1,206 → 248 kit clusters) is tabulated with every
predicate in `EXPERIMENT_SETTINGS.md`.

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

The three analysis commands above were checked against this release: run in a
bare checkout of these files they rewrite `results_vlm_v2.json`,
`results_f1_recheck.json` and `results_misattribution.json` byte for byte.
`vlm_retention.py` imports `cluster_bootstrap` and `wilson` from `score.py`, so
`score.py` has to stay at the repository root.

Stages 9–11 need the feature caches and the renders; once stage 8 has been
re-run they reproduce `results_retention_v2.json`,
`results_retention_excl.json`, `results_paired_contrasts.json` and
`results_perceptibility.json` with the
commands listed in `EXPERIMENT_SETTINGS.md`. All four of those commands were
re-run from the feature caches after this release was assembled, and each
rewrote its result file byte for byte, so the seven analysis scripts of this
release reproduce all seven of its result files exactly.

One ordering caution if you re-run the pipeline: `describe.py` writes its
resized model input beside the image it reads, as `.resized_<id>.png`, and
`features.py` collects `Path(dir).glob("*.png")`, which in `pathlib` matches
such dotted names. Run stage 8 before stage 12 on any directory, or filter
the dotted names out, otherwise the gallery silently grows. Neither released
run is affected — the feature caches hold exactly the documented number of
rows — but the order matters.

## The audit of the reported measurement

The paper's audit subsection reports an independent re-check of four claims about the primary table: how much of the page-weighted total weight the two
largest kits carry, how concentrated the hash column's hits under inversion are, whether the ordering claims survive the paired test, and whether rebuilding the kit clusters from CLIP embeddings moves any retention value. The 10 `audit_u_*.py` scripts are that computation and the 12 `audit_redo_*.json` files are its output, so each number in the subsection can be read out of a file:

| file | what it holds |
|---|---|
| `audit_redo_u.json` | the four claims on the 1,206-page run: `i_two_largest` (two largest kits, 171 + 129 of 1,206 pages, share 0.248756), `i_single_kit_share` (under inversion 171 of the hash column's 236 hits fall in one kit, 0.724576), `i_interval_widths` and `i_narrower` (cluster weighting narrows 34 of 48 intervals), `ii` (the colour column's float32 self-correlation artefact), `iii` (the ordering claims under the paired test) and `iv` (the CLIP re-clustering, 56 cells, largest shift 0.060602 at S3/HOG over all conditions and 0.034762 at A3/CLIP among the eight inversion cells) |
| `audit_redo_f.json` | the same four claims on the 833-page run, as a cross-check |
| `audit_redo_claim4_old_*.json` | claim (iv) under the first-pass hit rule, i.e. the estimator `run1_833/audit4.py` used, so the published "no value moved by more than 0.01" can be checked against its own estimator; the worst shift over the enlarged run is 0.060602 at S3/HOG |
| `audit_redo_extra*.json` | the colour artefact stated on pixel-identical rows, the ranking under the CLIP clustering, and every pair whose order the two clusterings disagree about, put through the paired test |

`audit_u_build.py` writes the per-page hit store the other scripts read (`audit_hits_u.npz` / `audit_hits_f2.npz`); like the feature caches it is not released, so the chain cannot be re-run from the repository alone. Three of the scripts print rather than write JSON (`audit_u_hashchk.py`, `audit_u_probe_firstrun.py`, `audit_u_probe_identical.py`). `EXPERIMENT_SETTINGS.md` Section 15 lists every script, its input and its output.

## The first measurement

`run1_833/` holds the derived data of the 833-page / 172-cluster measurement
released as v1.0.0. Every file in it is the same file as in the Zenodo archive
10.5281/zenodo.23080956; only the path changes. It is kept
because v1.0.0 is already archived and citable, because the paper reports the
agreement between the two measurements, and because the audit scripts in it
were written against it.

It is a strict subset in the sense that matters: the 834 fidelity-gated ids of
that run are all among the 1,206 of this one, the 372 added pages come from
the 7 shards the first content gate had not reached, and the ordering of the
eight configurations under inversion (A3) is identical in the two runs. The
enlarged gallery is harder — a query competes with more distractors — so
retention falls in most cells: of the 56 cells of Table 4, compared at full
precision, 31 are lower than in v1.0.0, 18 higher and 7 unchanged, and the
cells that rise are either at the ceiling already (the null condition, the hash
under A1/A4) or HOG, the one configuration the larger gallery helps. Compared
as the paper prints them, to three decimals, the counts are 31 / 17 / 8: the
8×8 column under inversion reads 0.012 in both runs while the stored values are
0.012295 and 0.011800. That is a property of the gallery, not
of the added pages.

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
release. Three DOIs are involved and they are not interchangeable:

* https://doi.org/10.5281/zenodo.23080955 is the **concept** DOI. It always
  resolves to the newest archived version and is the one the bibliography
  entry in the paper carries.
* https://doi.org/10.5281/zenodo.23145836 is the **version** DOI of the
  v2.0.0 archive, which carries the 1,206-page measurement the paper reports.
  Cite this one when you need the exact bytes the numbers came from; the
  concept DOI above will move on to later versions.
* https://doi.org/10.5281/zenodo.23080956 is the version DOI of the v1.0.0
  archive and stays bound to the 833-page measurement for good. It keeps
  resolving; it is not superseded, only smaller.
