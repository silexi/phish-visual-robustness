# Experiment settings

Every parameter behind the numbers in the paper, the exact commands that
produced the released result files, and the map from each table and figure of
the paper to the file it is read from. Section numbers refer to the paper.

## 1. Corpus funnel (paper Table 1, Section 3)

| stage | pages | how |
|---|---|---|
| archive | 136,414 | `nyuuzyou/phishing-snapshots`, 28 parquet shards (27 × 5,000 + 1 × 1,414), collected 24 July – 15 August 2024, CC0 |
| downloaded | 100,000 | the first 20 shards (0–19) |
| record filter | 25,646 | `has_password_field ∧ http_status == 200 ∧ html_length > 3000`, applied in `dl_shards.py` / `dl_retry.py` for shards 1–19 and inside `render.py` for shard 0 |
| per-shard cap | 12,119 | the *first* 700 candidates of each shard in file order (`worker.sh --limit 700`); 13 of the 20 shards had more than 700 candidates |
| rendered | 10,520 | 1,599 failures: 1,078 navigation timeouts (15 s), 376 loads interrupted by another navigation, 134 `net::ERR_TIMED_OUT`, 7 screenshot timeouts, 4 network changes |
| entered the content gate | 6,966 | successful renders of the 13 shards (0–7, 9–13) whose workers had finished when the gate was run; the other 7 shards (8, 14–19; 3,554 successful renders) were never gated — a timing accident, disclosed in the paper; the same predicate would have admitted 657 more pages |
| content gate | 1,702 | `ok ∧ pw_visible ∧ text_len > 60 ∧ n_colors ≥ 10` (`gate_content.py`, default 13 shards) |
| fidelity gate | 834 | after re-rendering over `file://` (`replay_ids.py`), `block_area ≥ 0.02` (`fidelity.py`); the same 834 ids result when the gate is applied to the first-pass renders |
| kit clusters | 172 | `dedup.py --maxdist 16`: 256-bit difference hash (17 × 16 Lanczos thumbnail, row-wise gradient), single-linkage at Hamming ≤ 16; 70 singleton clusters, median size 2, the two largest hold 129 and 107 pages (28 % of the corpus) |
| in the result tables | 833 | `ids_f.txt` had no trailing newline, so the shell `while read` loop that built `clean_sub/` skipped its last id (`ff85503356d552ee`); that page has a clean render and a perturbed render in every condition but no feature vector. All denominators in `results_retention_v2.json`, `results_retention_excl.json` and `results_pixel_clip.json` are therefore 833 (`n_query_rows` = 834, `n_pages` = 833) |

Declared languages of the 834 pages (recounted from the render manifests
restricted to `ids_f.txt`): 618 en, 21 pt, 13 fr, 5 ko, 5 de, 4 es, 2 zh,
1 it, 1 vi, 164 unspecified (sum 834). Median archived HTML size 143,841
bytes.

## 2. Render protocol (Section 4.1)

Identical in `render.py`, `replay_ids.py`, `render_html.py` and `perturb.py`
(`PROTOCOL` dict / module constants):

| setting | value |
|---|---|
| browser | Playwright 1.63.0, bundled Chromium, `--disable-lcd-text --hide-scrollbars` |
| viewport | 1366 × 768, `device_scale_factor` 1 |
| context | `color_scheme=light`, `locale=en-US`, `timezone_id=UTC` |
| navigation | `wait_until=domcontentloaded`, then a fixed 1,200 ms settle; navigation timeout 15,000 ms |
| network | every request except the page's own file is aborted (`ctx.route("**/*", abort)`); no external subresource is ever fetched |
| output | viewport PNG (no full-page capture) |
| serving mode | first pass (`render.py`): loopback HTTP; every later stage (`replay_ids.py`, `render_html.py`, `perturb.py`): `file://`. The switch is documented in `pipeline3.sh`; both classes and every condition take the identical `file://` path in the reported numbers |

Content covariates written per render: `pw_visible` (a password input with
non-zero bounding box), `text_len` (characters of `innerText`), `n_colors`
(colour bins after quantisation to 8 levels per channel that hold at least
0.05 % of the sampled pixels), plus the
fidelity covariates `ink_area` and `block_area` (`fidelity.py`, minimum run
24 px, near-white = every channel ≥ 238).

## 3. Perturbation conditions (Section 4.2, paper Table 2)

Injected at render time as one `<style>` element (`#__p` is an appended
overlay `div`), `perturb.py --cond <c>`; `applied` is `True` when the
rendered pixels differ from the clean render (or the geometry changed).

| cond | injection | applied / 833 |
|---|---|---|
| A0n | `#__p{}` (inert style; the null condition) | 30 (re-render noise only) |
| A1 | fixed full-screen overlay, `repeating-linear-gradient(0deg, rgba(0,0,0,.04) 0 1px, transparent 1px 3px)` | 833 |
| A3 | `html{filter:invert(1)}` | 698 |
| A4 | fixed 370 × 180 px top-left overlay, `repeating-linear-gradient(45deg,#000 0 2px,#fff 2px 4px)` | 833 |
| S1 | `img, svg, [class*=logo], [id*=logo], [class*=brand], [id*=brand] {visibility:hidden!important}` | 714 |
| S3 | `*{letter-spacing:.3em!important; word-spacing:.35em!important}` | 826 |
| S7 | fixed white footer bar, text `© 2024 Aventro Corporation — Aventro Secure Sign-In`, 16 px Arial | 833 |

The overlay `z-index` is 2147483647. Inversion left 135 pages unchanged; 129
of them are redeployments of one kit, so the non-applied cells fall in 3 of
the 172 clusters. The paper's Table 4 reports applied-only cells for every condition except
A0n, which is reported over all 172 clusters (the null condition changes no
pixels by design).

## 4. Representation configurations (Section 4.3, paper Table 3)

Feature extraction: `features.py` (pixel tier through `pixel/`, unmodified
except the fixed-length LBP histogram; CLIP through `open_clip`). Similarity
matrices: `retention2.py` / `retention_excl.py` (`rung_matrices`). Result
keys in the JSON files are given in brackets.

| key | configuration | input | feature | similarity |
|---|---|---|---|---|
| `P_color` | colour | full page, HSV (OpenCV 8-bit scale: H 0–180, S/V 0–256) | 8 × 8 × 8 histogram, 512 dims, min-max normalised | Pearson correlation (`HISTCMP_CORREL`), accumulated in float64 |
| `P5_triv8` | 8×8 | RGB, area-averaged 8 × 8 | 192-dim vector | `1 − d / d_max`, `d` Euclidean, `d_max` the largest distance among all query–gallery pairs of the condition |
| `P_ssim_global` | SSIM* | grey, Lanczos 256 × 256 | global μ, σ², σ_qr over all 65,536 pixels | global-statistics SSIM, C1 = (0.01·255)², C2 = (0.03·255)² (paper Eq. 1) |
| `P3_hash` | hash | grey, histogram equalisation, Gaussian blur r = 1, Lanczos | DCT hashes 32 × 32 and 64 × 64 (image resized to 4n × 4n, orthonormal 2-D DCT, top-left n × n block, the 2 × 2 DC corner weighted ×1.5, thresholded at the median of the unweighted block) + Haar LL hashes 32 × 32 and 64 × 64 (image 2n × 2n, one-level Haar, LL thresholded at its median); 10,240 bits | 1 − weighted Hamming fraction, block weights 0.6 (32 × 32) / 0.4 (64 × 64) |
| `P_lbp` | LBP | grey, 128 × 128 | uniform LBP, P = 8, R = 1, fixed 10-bin histogram, L1-normalised | histogram intersection Σ min(h, h′) |
| `P4_hog` | HOG | grey, 128 × 128 | 9 orientations, 8 × 8 px cells, 2 × 2 blocks, L2-Hys, 15 × 15 blocks → 8,100 dims | (cos θ + 1) / 2 |
| `P0_fusion` | fusion | full page + ROI crop | five terms | `0.5·hash + 0.15·colour + 0.10·SSIM* + 0.15·LBP + 0.10·HOG`, each non-SSIM term `0.7·ROI + 0.3·full` (paper Eq. 2); weights in `pixel/config.py` |
| `E1_clip` | CLIP | RGB, short side bicubic to 224, centre crop 224 × 224, CLIP mean/std | `open_clip` `ViT-B-32-quickgelu`, `pretrained="openai"`, 512-dim, L2-normalised | cosine (= dot product) |

ROI (fusion only): Canny edges of the blurred grey image, dilated; contours
with area > 1/1000 of the image filled; a crop of half the image width and
height (at least 30 %) around the densest point of the resulting map; fallback
crop x ∈ [25 %, 75 %], y ∈ [30 %, 80 %] when no contour is found.

Library versions: scikit-image 0.26.0, OpenCV 5.0.0, open_clip_torch 3.3.0,
PyTorch 2.14.0 (CPU), numpy 2.5.3 (`requirements.txt`).

## 5. Primary metric (Section 4.5, paper Table 4)

* Gallery: the 833 clean renders (the query's own clean render included; see
  Section 6 below for the exclusion galleries).
* Hit rule: the query's own kit cluster attains the row maximum of the
  similarity row within `eps = 1e-4` (ties at the maximum are the normal case
  because the gallery contains byte-identical renders).
* Estimand: mean over kit clusters of the per-cluster rank-1 retention
  (`cluster_mean`); page means (`page_mean`) are reported alongside but never
  used in the paper's headline numbers.
* Cells: `applied_only` (cells whose perturbation changed the pixels) for
  A1–S7; `itt_all` (all 172 clusters) for A0n.
* Uncertainty: cluster bootstrap, B = 4,000, seed 20260922, percentile 2.5 /
  97.5 (`boot95`); `wilson95_full` is the Wilson interval for the fraction of
  fully retained clusters and is not used in the tables.
* Command (as run, `pipeline3.sh` stage 6 produced the `.npz` inputs):

```bash
./venv/bin/python retention2.py --clean feat_A0.npz \
    --conds A0n=feat_A0n.npz A1=feat_A1.npz A3=feat_A3.npz A4=feat_A4.npz \
            S1=feat_S1.npz S3=feat_S3.npz S7=feat_S7.npz \
    --clusters clusters_flat.json --applied pert --eps 1e-4 \
    --out results_retention_v2.json
```

## 6. Exclusion galleries (Section 5.2, paper Table 5)

`retention_excl.py` (defaults reproduce the released file):

```bash
./venv/bin/python retention_excl.py --clean feat_A0.npz \
    --conds A0n A1 A3 A4 S1 S3 S7 --clusters clusters_flat.json \
    --applied pert --eps 1e-4 --out results_retention_excl.json
```

Gallery facts written at the top level of the file: 833 pages, 240 distinct
256 × 256 grey thumbnails, 712 pages with a byte-identical twin, 119
duplicate groups, none straddling two clusters, 269 distinct-byte pairs
within RMSE ≤ 1 grey level, 83 distinct-byte pairs with identical dHash.

| variant | candidates removed from the query's row |
|---|---|
| `full` | none (reproduces `results_retention_v2.json`) |
| `self_excl` | the query's own clean render |
| `dup_excl` | the own clean render and every clean render whose 256 × 256 grey thumbnail is byte-identical to it |
| `rmse1_excl` | every clean render within RMSE ≤ 1 grey level of the own clean render (sensitivity) |
| `dhash0_excl` | every clean render with the same 256-bit dHash as the own clean render (sensitivity) |

A query whose cluster has no candidate left is dropped; the surviving
population is `n_pages` / `n_clusters` per condition (A3 `dup_excl`: 376
pages, 29 clusters). Each variant carries the same population scored against
the full gallery (`rungs_full_gallery_same_population`) and, for `dup_excl`,
also under `self_excl` (`rungs_self_excl_same_population`), so galleries are
compared on one set of clusters. Paired contrasts (`paired_minus_full`,
`paired_hog_minus_clip*`) are the mean over clusters of the per-cluster
difference with the same cluster bootstrap.

The two paired differences quoted in the paper's text on the primary (full)
gallery, HOG − CLIP under A3 and LBP − fusion under A4, come from
`paired_contrasts.py --clean feat_A0.npz --conds A0n A1 A3 A4 S1 S3 S7 --clusters clusters_flat.json --applied pert --eps 1e-4 --out results_paired_contrasts.json`:
per page the rank-1 hit of one configuration minus the hit of the other,
averaged per cluster and then over clusters (`cluster_mean_diff`, applied-only,
same hit rule and bootstrap as Table 4), with the number of clusters whose
mean difference is strictly positive / negative. Each entry also carries a
`*_page_weighted` variant (page mean, cluster-resampled) for traceability to
an earlier audit; the paper quotes the cluster-weighted values only (A3
HOG − CLIP 0.282 [0.217; 0.350], 169 clusters; A4 LBP − fusion −0.734
[−0.798; −0.669], 172 clusters).

## 7. Perceptibility (Section 5.4, paper Table 7, Figure 3)

`perceptibility.py --clean clean_f/shots --pert pert --conds A0n A1 A3 A4 S1 S3 S7 --ids ids_f.txt --sample 200 --out results_perceptibility.json`

The 200 pages are a seeded random sample (`numpy.random.default_rng(20260922)`,
`choice` without replacement, inside the script) of the 834-id list
`ids_f.txt` (the script reads all 834 ids; the id that the feature stage
skipped is not among the 200 drawn, so the sample lies inside the 833-page
result population). This sample is **not** the VLM subset (`ids_vlm200.txt`, the
first 200 lines of `ids_fid_full.txt`); the two overlap only partially.
Re-running the command above reproduces `results_perceptibility.json`
byte for byte (verified 2026-09-30); running it on `ids_vlm200.txt` does
not. Per cell: SSIM (Wang 2004 parameters, Gaussian weights σ = 1.5,
`data_range=255`), mean CIEDE2000, its 99th percentile, and `f_JND` =
fraction of pixels with ΔE2000 > 1.0. The table reports medians over the
200 pages.

## 8. Vision-language model protocol (Section 4.4, paper Table 6)

| setting | value |
|---|---|
| checkpoint | `mlx-community/Qwen3-VL-8B-Instruct-8bit` (affine 8-bit quantisation, group size 64) |
| runtime | `mlx-vlm` 0.7.2, `mlx` 0.32.2, Apple M4 Pro, 64 GB |
| decoding | greedy (temperature 0), `max_tokens` 900 |
| image | longest side resized to 1,024 px (Lanczos) before the processor; one image per chat turn, no history |
| prompt | `vlm/PROMPT.txt`, verbatim (also embedded in `describe.py`) |
| output | one JSON object per render (`brand`, `logo_present`, `page_type`, `asks_for_credentials`, …); the raw text is kept in `raw`, the parsed object in `parsed` (`null` when no JSON object could be extracted) |
| subset | `head -200 ids_fid_full.txt` (file order, no random selection); 66 clusters, 196 pages with a parsable clean answer (65 clusters), largest cluster 33 pages |
| A1 / S3 | first run used the first 200 lines of `ids_f.txt` (only 51 pages in common); re-run on the 200 clean ids (`rerun_a1s3.sh`, `run_s3.sh`); the re-run list lacked a trailing newline, so those two conditions have 199 rows |
| legitimate side | `run_benign_vlm.sh`: first 120 of the 156 replayed captures in filename order (120 brand labels, 113 registrable domains); 119 parsed |
| throughput | median 11.05 s per image (28.5 tokens/s), median prompt 823 tokens, median answer 246 tokens, peak memory 11.0 GB |

Retention (`vlm_retention.py`), per page against the page's own clean
answer:

* brand: `re.sub(r"[^a-z0-9]+", "", brand.lower())` on both sides, equal
  after normalisation = retained; denominator = pages whose clean answer names
  a brand (148; 147 for A1/S3);
* abstention: clean brand non-null and perturbed brand `null`;
* page type and `asks_for_credentials`: exact equality, denominator = pages
  with a parsable clean answer (196; 195 for A1/S3);
* a perturbed answer that cannot be parsed counts as a miss on every channel;
* misattribution (Table 6, `y.atıf` column): the perturbed answer parses and
  names a brand that differs from the clean brand after normalisation,
  including brands invented for pages that were unnamed in the clean run;
  denominator = pages with a parsable clean answer present in the condition
  file (196; 195 for A1/S3). `misattribution.py` reproduces the column from
  `vlm/desc_*.jsonl` alone (`results_misattribution.json`: A0n 1/196, A1
  7/195, S3 8/195, A4 13/196, A3 13/196, S7 61/196, S1 11/196) together with
  the named/unnamed split quoted in Section 6.3 (S7: 20 + 41, A3: 1 + 12,
  S1: 11 + 0) and the conditional count on the 148 (147) brand-eligible pages;
* intervals: cluster bootstrap of the page proportion, B = 2,000, seed
  20260922; the table is page-weighted (stated in the paper).

## 9. Credential-intent detector (Section 6.2, paper Table 8)

`f1_recheck.py` (no arguments) → `results_f1_recheck.json`. Positive class
phishing; predicted positive `parsed.asks_for_credentials is True`; clean
phishing renders of the 200-page subset (196 parsed → TP 129, FN 67) against
the 120 described legitimate captures (119 parsed). Negative sets: (a) the
pages the model itself labels `login` / `password_entry` / `mfa_challenge`
(46; circular), (b) all parsed legitimate pages (119), (c) pages whose
replayed render shows a visible password field (`benign_replay/
manifest_replay.jsonl`, 31), (c′) the same flag from the live capture
(`benign/fetch_manifest.jsonl`, 28). Sensitivity: parse failures counted as
negative predictions; the field salvaged by regular expression from the
truncated raw text. Also reported: the F1 of the always-positive baseline,
TPR − FPR, Wilson 95 % intervals, and recall cluster-weighted (65 clusters,
cluster bootstrap B = 4,000, seed 20260922).

## 10. Detector operating point (secondary analysis)

`score.py` (`pipeline3.sh` stage 7): every configuration read at the
threshold that gives 5 % FPR on the brand-disjoint legitimate calibration
split (`split_benign.py`: reference / calibration), `results_pixel_clip.json`.
Not used in the paper's tables; kept because Section 4.5 argues why a
threshold-based reading is the wrong primary metric.

## 11. Table and figure → file map

| paper | file(s) | keys / script |
|---|---|---|
| Table 1 (funnel) | `gate_content.py` docstring, `ids_gated_all.txt`, `ids_f.txt`, `clusters_f.json` | counts above |
| Table 2 (conditions) | `perturb.py` `CONDITIONS`, `pert/manifest_*.jsonl` | `applied` counts |
| Table 3 (configurations) | `pixel/config.py`, `features.py`, `retention2.py` | Section 4 above |
| Table 4 (retention) | `results_retention_v2.json` | `conditions[c].rungs[r].applied_only.cluster_mean` / `.boot95`; A0n row: `.itt_all` |
| paired differences quoted in Sections 5–6 (A3 HOG − CLIP, A4 LBP − fusion) | `results_paired_contrasts.json` | `conditions[c].P4_hog_minus_E1_clip.cluster_mean_diff` / `.boot95`, `conditions[c].P_lbp_minus_P0_fusion…`; `paired_contrasts.py` |
| Table 5 (exclusion galleries) | `results_retention_excl.json` | `conditions[c].variants[v].rungs[r].cluster_mean`, `…rungs_full_gallery_same_population`, `…paired_minus_full[r].cluster_mean_diff` |
| Table 6 (semantic retention) | `results_vlm_v2.json`; misattribution: `results_misattribution.json` | `vlm_retention.py`; `misattribution.py` (`conditions[c].rate`, `.misattributed_on_named`, `.invented_on_unnamed`) |
| Table 7 (perceptibility) | `results_perceptibility.json` | medians |
| Table 8 (confusion matrix) | `results_f1_recheck.json` | `variants[]`, `cluster_weighted_recall` |
| Figure 1 | `figures/input/panel/*.png` | `figures/make_figures.py` |
| Figure 2 | `figures/input/results_retention_v2.json` | idem |
| Figure 3 | `figures/input/results_perceptibility.json`, `results_retention_v2.json`, `results_vlm_v2.json` | idem |
| Figure 4 | `figures/input/panel/clean.png`, `S7.png` | idem |

## 12. Known defects kept for the record

* The last id of `ids_f.txt` was skipped by the shell read loop (834 → 833);
  the file is released as it was.
* `P_ssim_global` is accumulated in float32 and cast afterwards; self-SSIM of
  identical renders ranges 0.99954–1.00094, wider than `eps`. Sampled impact:
  no hit flips at `eps ≤ 2e-3` on 120 rows, one flip at `5e-3`.
* The content gate covered 13 of 20 shards (Section 1 above).
* `results_pixel_clip.json` was regenerated with the float64 histogram fix;
  its `score_p95` values shifted cosmetically and it is not cited in the
  paper.
