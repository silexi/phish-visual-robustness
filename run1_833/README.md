# The first measurement: 833 pages, 172 kit clusters

This directory holds the derived data of the measurement released as **v1.0.0**
(Zenodo 10.5281/zenodo.23080956), together with the audit scripts written
against it. Every file in it carries the same content as its counterpart in
that archive; only the path has changed. One file is new rather than moved:
`audit_perc.json`, which is `audit_perc.py`'s own output of 22 September 2026
on this run's renders. v1.0.0 carried no copy of it. Nothing here is
superseded in the sense of being wrong — it is a smaller measurement of the
same protocol, and the agreement between it and the 1,206-page measurement the
paper now reports is itself a result: the ordering of the eight representations
under inversion (A3) is identical in the two runs, HOG first, CLIP second, the
five classical metrics far below.

Why it is kept: v1.0.0 is already archived and citable; the paper reports the
comparison between the two measurements; and the 23 `audit*.py` /
`verify_*.py` scripts were written while checking this run and hard-code its
population (`verify_analyst.py` asserts 833 clean ids; several print 172
clusters as the reference). Those scripts were deliberately not re-pointed at
the larger measurement, because an audit script rewritten after the fact is not
a record of anything.

They live in this directory, not at the repository root, and that is not a
filing preference. 16 of them build every path from
`Path(__file__).resolve().parent`, so a copy left at the root would find
`clusters_flat.json`, `pert/manifest_*.jsonl` and `vlm/desc_*.jsonl` there
under the same names — the 1,206-page files — and would audit the wrong run
without saying so. The remaining 7 (`verify_analyst_cmp.py`,
`verify_reviewer_orient.py`, `verify_reviewer_shapes.py`,
`verify_reviewer_ssim.py`, `verify_stat_f1.py`, `verify_stat_f1b.py`,
`verify_stat_f1c.py`) take their paths relative to the **working directory**
instead, so they audit whichever directory you are standing in: run them from
inside this one, never as `python run1_833/<script>.py` from the root.

What each of the 23 can do from the released files was measured by running all
23 from this directory (on an interpreter whose default encoding is UTF-8;
several of them `open()` the description files without an explicit encoding,
so `PYTHONUTF8=1` is needed on a platform that defaults to a single-byte code
page):

* **three reproduce their result with no argument**: `audit9.py`, `audit12.py`
  and `verify_stat_f1c.py`.
* **`verify_stat_f1b.py` runs once the legitimate side is beside it**: it reads
  `vlm/desc_BENIGN.jsonl`, `benign/fetch_manifest.jsonl` and
  `benign_replay/manifest_replay.jsonl` from the working directory, and those
  three stay at the repository root because both measurements share them. Copy
  them in at the same relative paths and it runs.
* **`audit_perc.py` exits 0 but produces nothing, and overwrites
  `audit_perc.json` with the empty result.** It compares `clean_sub/<id>.png`
  with `pert/<cond>/<id>.png` and skips every pair it cannot find; no render is
  released. The `audit_perc.json` here is the 200-rows-per-condition file it
  wrote on 22 September 2026, when the renders were present. Do not run it in a
  working copy you want to keep.
* **the other 18 cannot run** from the released files: they need the feature
  caches (`feat_*.npz`, `audit_hits.npz`), the benign replay renders
  (`benign_replay/shots`, which `verify_stat_f1.py` and
  `verify_reviewer_shapes.py` reach for) or another script's output
  (`verify_analyst_cmp.py` needs `verify_analyst_out.json`). Six of them also
  put `pixel/` on `sys.path` (`audit_hits.py`, `audit6.py`,
  `verify_analyst.py`, `verify_analyst_leak.py`, `verify_reviewer_orient.py`,
  `verify_reviewer_ssim.py`), so running one needs the caches *and* a layout
  where `pixel/` sits beside the script.

## What this run measured

| stage | pages |
|---|---|
| entered the content gate | 6,966 (the 13 of 20 shards whose render workers had finished: 0–7, 9–13) |
| content gate | 1,702 |
| fidelity gate | 834 |
| kit clusters | 172 (70 singletons, median size 2, two largest 129 and 107 pages) |
| in the result tables | 833 |

The 834 → 833 step is this run's documented off-by-one: `ids_f.txt` has no
trailing newline, so the shell `while read` loop that built the clean gallery
skipped its last id (`ff85503356d552ee`); that page has a clean render and a
perturbed render in every condition but no feature vector. All denominators in
this run's result files are therefore 833 (`n_query_rows` 834, `n_pages` 833).
Both of this run's shortcomings — the 13-shard content gate and the skipped id
— are resolved in the 1,206-page measurement, where the gate covers all 20
shards and every one of the 1,206 ids carries a feature vector.

`applied` counts of this run, out of 833: A0n 30, A1 833, A3 698, A4 833,
S1 714, S3 826, S7 833. The two paired differences quoted in the paper's text
were, in this run, A3 HOG − CLIP 0.282 [0.217; 0.350] over 169 clusters and
A4 LBP − fusion −0.734 [−0.798; −0.669] over 172 clusters.

## Files

| file | content |
|---|---|
| `ids_gated_all.txt` | the 1,702 content-gated ids of the 13 shards, in the order the hand run of 2026-09-22 produced them (not sorted; `gate_content.py` writes sorted output, so re-running it on those 13 shards reproduces this id set but not this line order — compare with `--check`) |
| `ids_f.txt` | the 834 fidelity-gated ids, sorted, **without** a trailing newline (released as it was) |
| `ids_fid_full.txt` | the same 834 ids in the order the fidelity stage produced them; the first 200 lines of the two files share only 51 ids |
| `ids_vlm200.txt` | this run's semantic-rung subset, `head -200 ids_fid_full.txt` |
| `clusters_f.json`, `clusters_flat.json` | the kit clustering of this run (834 pages, 172 clusters); `clusters_flat.json` is the flat page → cluster map the analysis scripts read |
| `clusters_full.json` | an earlier clustering of the same 834 pages (173 clusters), kept for provenance |
| `fidelity_f.jsonl` | `ink_area` and `block_area` per render for the 1,702 re-rendered pages |
| `pert/manifest_*.jsonl` | 834 rows per condition, with the per-page `applied` flag |
| `vlm/desc_*.jsonl` | this run's Qwen3-VL outputs for the clean baseline and the seven conditions; A1 and S3 have 199 rows because the re-run id list lacked a trailing newline. The legitimate side (`desc_BENIGN.jsonl`) is shared with the 1,206-page run and stays at the repository root, as does `PROMPT.txt` |
| `results_retention_v2.json` | the primary retention table of this run |
| `results_retention.json` | the same quantity from the earlier `retention.py`, kept for provenance |
| `results_retention_excl.json` | the five exclusion galleries |
| `results_paired_contrasts.json` | the two paired differences |
| `results_perceptibility.json` | SSIM / ΔE2000 / `f_JND`, 200 pages drawn from `ids_f.txt` |
| `results_vlm_v2.json` | semantic retention |
| `results_vlm_retention.json` | the same from the earlier script generation, kept for provenance |
| `results_misattribution.json` | the misattribution column |
| `results_f1_recheck.json` | the credential-intent confusion matrix |
| `results_pixel_clip.json` | the secondary detector operating point |
| `audit_meta.json` | the per-condition `n` / `n_applied` table `audit_hits.py` wrote for this run |
| `audit_perc.json` | `audit_perc.py`'s per-page SSIM / ΔE2000 / `f_JND` rows for A3, S1, S7 and A4 on the 200-page sample (200 rows per condition), written 22 September 2026. The only file here that v1.0.0 did not carry; see the warning above before re-running `audit_perc.py` |
| `audit*.py`, `verify_*.py` | the 23 audit scripts of this run, unchanged (see above) |

All result files here were written by the analysis scripts at their 4-decimal
rounding; the repository root now carries the same scripts at 6 decimals, with
the same computation and the same seeds, so re-running them against this run's
inputs gives these numbers with two more digits, not different numbers. That
was checked rather than assumed: the 6-decimal `misattribution.py`,
`f1_recheck.py` and `vlm_retention.py` re-run on this directory's inputs, and
the 6-decimal `retention2.py` re-run on this run's feature caches, agree with
`results_misattribution.json`, `results_f1_recheck.json`,
`results_vlm_v2.json` and `results_retention_v2.json` in every field once
rounded back to four decimals — zero differences.
