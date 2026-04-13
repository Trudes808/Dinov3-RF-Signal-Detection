# Holoscan Signal Detection Plan

## Objective

Reduce false detections in holoscan v2 that are driven by raw power level or front-end response shape rather than by signal structure that is locally distinct from the surrounding noise.

Target behavior:

- RF front-end correction should flatten the wideband floor enough that each spectrogram subsection can make a mostly local detection decision.
- High-power but structure-poor regions should not survive as signal detections just because they sit on an elevated floor.
- Real signals should remain detectable after correction, chunk scoring, and wideband merge.

## Working Hypothesis

- The current pipeline still allows raw or corrected power level to leak into the detection score, especially in subsections where the local floor is elevated.
- Chunk-local normalization and chunk-local quantile thresholds can stretch weak subsection evidence into a strong-looking score map.
- Global front-end reliability weighting may be reintroducing cross-subsection asymmetry even after the floor is flattened.
- Companion and support logic may be preserving detections in high-power regions that do not have enough structural evidence.

## Immediate Questions To Answer First

- What is causing the apparent smearing across the time domain, where the raw spectrogram looks locally sharp but the overlayed score and debug maps spread energy broadly along time?
- Why does the merged front-end reliability strongly favor regions of higher global power, and is that bias entering through the floor estimate, boost, reliability mapping, or merge weighting?

## Current Active Task

- Root-cause why merged frontend reliability heavily favors regions of higher global power.
- Decide whether frontend reliability should exist at all after RF frontend correction has already flattened the wideband response.
- Keep subsection power weighting local to each subsection instead of reintroducing another row-global reliability weight after correction.
- Preserve the recent time-smear and pixel-space-fusion work as background context, but defer further smear tuning until the frontend-reliability path is understood.
- Keep striping work closed unless the time-smear fix exposes a new striping regression.
- Defer other raw-power and striping follow-up until the frontend-reliability path is understood.

## Current Baseline Setup

Notebook and implementation:

- `signal_detection_holoscanv2.ipynb`
- `signal_detection_holoscanv2_helpers.py`

Current notebook operating point:

- subsection bandwidth: 25 MHz
- subsection overlap: 6.25 MHz
- ignored sideband per edge: 7 MHz
- DINO grouping: `k=8`, spatial weight `0.35`, score quantile `0.60`
- power fusion gain: `0.25`
- final merged threshold quantile: `0.88`
- merge support quantile: `0.70`
- companion floor: `0.30`
- frontend floor estimate: row quantile `25`, reference quantile `75`, smooth sigma `12`, max boost `12 dB`

## Current Experiments In The Notebook

### 1. End-to-end wideband baseline run

Purpose:

- Load a PGM, SigMF, or tensor snapshot.
- Apply global front-end correction.
- Adapt subsection size to the current input.
- Run chunked DINO detection and merge into one wideband result.

Outputs already inspected:

- subsection count
- merged threshold
- total runtime
- effective ignore-sideband settings

### 2. Detection path decomposition

Purpose:

- Inspect what drives detection in each subsection.
- Review the contribution of DINO score, power prior, companion gate, strict score, and support score.

Current diagnostic views:

- subsection coverage over the corrected spectrogram
- strict per-subsection score
- weighted subsection contribution back in full-band coordinates
- merged base score, merged support, and final mask overlay

### 3. Front-end correction overview

Purpose:

- Inspect the global floor estimate, response, boost, and reliability profile.
- Check whether the correction is flattening the floor uniformly enough across the band.

Current diagnostic views:

- front-end overview plot
- response, boost, and reliability profile with subsection probes

### 4. A/B subsection comparison for suspicious chunks

Purpose:

- Compare a strong-looking subsection against a weaker or cleaner subsection.
- Separate differences in DINO input, DINO gated score, power patch, companion patch, strict patch, and row reliability.

Current notebook probe:

- explicit comparison of subsection `2` vs subsection `19`

### 5. Raw power vs score correlation study

Purpose:

- Test whether subsection detections are correlated with raw chunk power instead of local signal evidence.
- Compare raw power statistics against score statistics and driver means.

Current measurements:

- raw chunk mean power
- raw chunk 75th percentile power
- corrected chunk mean power
- front-end reliability mean
- DINO seed mean
- DINO score mean
- power patch mean
- raw score mean and p95

<!--
### 6. Sparse evidence weighting experiment

Purpose:

- Check whether sparse-evidence weighting reduces broad, weak, inflated subsection contributions.
- Inspect how sparse weight changes the contribution that survives into the merge.

Current measurements:

- sparse weight
- support fraction
- upper-tail gap
- raw mean vs merge contribution mean
-->

<!--
### 7. Texture-augmented experiment

Purpose:

- Test whether texture recurrence helps recover structure that baseline v2 underweights.
- Compare baseline and texture-augmented merged masks and subsection scores.

Current measurements:

- subsection texture gain
- texture reliability
- structure clean score
- structure speckle score
- merged mask and merged score deltas
-->

### 8. Time-tiled DINO experiment inside a signal-bearing subsection

Purpose:

- Test whether breaking a subsection into time windows changes the DINO score behavior.
- Compare full-subsection DINO inference against tiled-in-time inference.

Current measurements:

- full subsection patch-grid shape
- per-tile patch-grid shape
- resized pixel score map
- tiled minus full DINO score delta

## Current Findings To Carry Forward

- A known v2 risk is repeated chunk-local normalization combined with chunk-local quantile thresholds. This can stretch weak subsection structure into a strong-looking score map.
- Strong or asymmetric behavior across subsections does not appear to come directly from other chunks changing a local strict score. A bigger driver is the global front-end reliability profile applied after correction.
- One previously observed issue is that subsection `2` had front-end reliability near `1.0` while subsection `19` was around `0.46` despite similar DINO gray inputs.
- The notebook already includes explicit correlation checks between raw power statistics and subsection score statistics, which is the right direction for validating the raw-power hypothesis.
- The open concern is that the corrected floor may still not be making subsections independent enough for fair local decision-making.
- Striping has been investigated enough to defer for now; the current priority is the separate time-smear issue.
- Current smear hypothesis: the major visual broadening appears when low-resolution patch maps are resized back to chunk pixels with bilinear interpolation, especially along the time axis, before the merged overlays are plotted.

## Things To Try Next

- For the same regions, capture stage-by-stage time profiles before and after each major operation: front-end correction, DINO resizing, score fusion, companion gating, subsection weighting, and wideband merge.
- Switch the patch-to-pixel visualization path away from bilinear interpolation along time and compare nearest or time-preserving resize against the current overlays.
- Verify whether the smear is already present in per-chunk strict and companion maps before wideband merge, or only after merge.
- Leave raw-power, reliability-bias, and other ablations deferred until the sharper time-domain overlays are in place.

<!-- Deferred until the striping, time-smear, and front-end reliability bias questions are answered:
- Replace chunk-local quantile calibration with a fixed or globally calibrated mapping for strict and support scores.
- Build a noise-relative power prior that uses residual above local floor instead of absolute corrected power.
- Compare global front-end correction alone against a two-stage scheme: coarse global flattening followed by local subsection floor normalization.
- Tighten companion/support gating in high-power but low-coherence regions.
-->

## Things Still Needing A Clear Answer

- Is the time smear being introduced before DINO, inside DINO score upsampling, during fusion, or during wideband merge?
- Why does merged front-end reliability track high global power so strongly even in regions that should be judged locally?
- Is the dominant false-positive driver raw power, corrected power, front-end reliability, or chunk-local normalization?
- Should subsection thresholding be local, global, or hybrid?
- Should the power prior be additive evidence, multiplicative evidence, or only a support gate?
- Is the front-end correction itself flattening the floor correctly in the problematic frequency regions?
- Are subsection size and overlap helping local independence, or are they still too wide for this use case?
- What metric should decide whether a change is actually better: mask pixels, subsection false-positive rate, rank ordering of known signals, or a labeled benchmark?

## Immediate Backlog

- [x] Mark striping investigation complete enough to defer while focusing on time smear.
- [ ] Record one time-smear example where the raw input is sharp but the overlayed maps spread evidence along time.
- [ ] Record the current smear root-cause evidence from the stage-by-stage temporal diagnostic.
- [ ] Update cells 13 and 14 so the plotted overlays preserve sharper time localization.
- [ ] Re-check whether the remaining smear is coming from chunk fusion or from wideband merge after the resize change.
- [ ] Keep high-global-power, correlation, and front-end reliability follow-up deferred until the smear plots are cleaned up.

## Experiment Log

### 2026-04-13

- Created this plan file to track holoscan v2 false-positive analysis.
- Documented the current notebook experiment set and the main working hypothesis: subsection detections are still being biased by power level and/or cross-band reliability weighting instead of purely local noise-relative structure.
- Marked striping work complete enough to defer.
- Promoted time-domain smear in notebook cells 13 and 14 to the active task.
- Captured the current working hypothesis that bilinear patch-to-pixel resizing is a major source of the visible time smear in the overlays.
- Added a live notebook shape-audit cell to print the merged strict-score shape, the wideband spectrogram shape, and representative subsection patch-to-pixel mapping ratios for the current run.
- Confirmed with a live notebook diagnostic that the current `32 x 19` strict-score map is coming from DINO patch tokens on a resized subsection input of `512 x 304` with patch size `16`, not from the final wideband merge canvas. The current subsection example is `1025 x 625` in freq x time, resized before DINO to `512 x 304`, then scored on a `32 x 19` token grid and expanded back to subsection pixels.
- Confirmed with a live notebook diagnostic that full-resolution DINO on the current CPU path would raise one representative subsection from `608` tokens (`32 x 19`) to `2496` tokens (`64 x 39`) after patch cropping, which is a `4.1x` token increase and about a `16.9x` self-attention cost increase per subsection. With the current `27` subsection run, full-resolution CPU DINO is likely too expensive for routine notebook iteration.
- Confirmed that power and companion/coherence evidence already exist in pixel space before patch reduction. `power_prior_patch_map` builds local support from the full chunk pixels and then averages to the patch grid, while the coherence path builds `coherence_gate_px`, `residual_px`, and `companion_px` in pixel space before averaging to patches. This means the main design choice is whether to keep DINO coarse but fuse the other evidence later at pixel resolution instead of forcing everything through the DINO patch grid.
- Started the implementation change to make coherence resolution independent from DINO token resolution and to perform strict-score fusion in pixel space after DINO upsampling.
- Verified the helper behavior in the workspace Python environment with a synthetic `1025 x 625` chunk and a mocked `32 x 19` DINO grid: coherence now operates at `1025 x 625`, and `strict_px`, `score_px`, `power_px`, and `companion_px` all stay at `1025 x 625` while the DINO patch summary remains `32 x 19`.
- Temporarily changed the default config to bypass coherence gating entirely with `use_coherence_gate=False` so the notebook can run quickly again. The bypass path preserves the DINO score and zeros the coherence-derived companion inputs, which effectively removes coherence and companion gating without deleting the code needed to restore them later.
- Shifted the active task from time-smear tuning to investigating whether merged frontend reliability is reintroducing a row-global power bias after frontend correction, and whether that weighting should simply be removed.