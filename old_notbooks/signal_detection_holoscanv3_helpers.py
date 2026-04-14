from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy import ndimage as ndi
from sklearn.neighbors import NearestNeighbors

Image.MAX_IMAGE_PIXELS = None

from signal_detection_holoscanv2_helpers import (
    WidebandChunkConfig,
    _chunk_blend_weights,
    _display_db_window,
    _normalize_map01_local,
    _resize_patch_map_to_pixels,
    _resize_patch_mask_to_pixels,
    _robust_high_quantile_threshold,
    _smooth_binary_label_map,
    apply_coherence_gate_to_dino_result,
    apply_global_frontend_correction,
    build_frequency_chunks,
    burst_companion_gate,
    compute_ignore_sideband_rows,
    dino_grouping_from_spectrogram,
    generate_spectrogram,
    infer_input_kind,
    load_dino_model,
    load_input_record as load_input_record_base,
    load_sigmf_samples,
    merge_chunk_results,
    plot_frontend_overview,
    plot_merged_debug,
    plot_merged_detection,
    power_prior_patch_map,
    read_sigmf_meta,
)


@dataclass
class HoloscanChunkConfig(WidebandChunkConfig):
    texture_k: int = 6
    texture_q: float = 0.90
    power_q: float = 0.90
    pipeline_gap_floor: float = 0.10
    pipeline_final_threshold: float = 0.20
    pipeline_final_threshold_no_speckle: float = 0.10
    pipeline_component_min_size: int = 5
    pipeline_component_min_size_no_speckle: int = 2
    pipeline_power_rescue_floor: float = 0.10
    pipeline_power_rescue_gain: float = 2.0
    pipeline_strong_speckle_min_component: int = 10
    texture_speckle_clean_threshold: float = 0.85
    texture_speckle_strong_threshold: float = 0.20


def infer_expected_spectrogram_shape(input_path: str | Path) -> tuple[int, int] | None:
    match = re.search(r"_(\d+)x(\d+)\.pgm$", str(input_path))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def infer_expected_time_bins(input_path: str | Path) -> int | None:
    shape = infer_expected_spectrogram_shape(input_path)
    if shape is None:
        return None
    return int(shape[0])


def infer_expected_frequency_bins(input_path: str | Path) -> int | None:
    shape = infer_expected_spectrogram_shape(input_path)
    if shape is None:
        return None
    return int(shape[1])


def _sigmf_duration_for_time_bins(time_bins: int, fft_size: int, noverlap: int, sample_rate_hz: float) -> float:
    time_bins = max(int(time_bins), 1)
    step_samples = max(int(fft_size) - int(noverlap), 1)
    total_samples = int(fft_size) + max(time_bins - 1, 0) * step_samples
    return float(total_samples / float(sample_rate_hz))


def load_input_record(
    input_path: str | Path,
    input_kind: str = "auto",
    fft_size: int = 1024,
    noverlap: int = 512,
    sigmf_capture_index: int = 0,
    sigmf_channel: int = 0,
    sigmf_window_start_s: float = 0.0,
    sigmf_window_duration_s: float | None = 1.0,
    usrp_summary_path: str | Path | None = None,
    sigmf_target_time_bins: int | None = None,
    sigmf_target_freq_bins: int | None = None,
    sigmf_slice_offset_windows: int = 0,
) -> dict[str, Any]:
    resolved_kind = infer_input_kind(input_path, input_kind)
    if resolved_kind != "sigmf" or sigmf_target_time_bins is None:
        return load_input_record_base(
            input_path=input_path,
            input_kind=input_kind,
            fft_size=fft_size,
            noverlap=noverlap,
            sigmf_capture_index=sigmf_capture_index,
            sigmf_channel=sigmf_channel,
            sigmf_window_start_s=sigmf_window_start_s,
            sigmf_window_duration_s=sigmf_window_duration_s,
            usrp_summary_path=usrp_summary_path,
        )

    _, global_info, captures, annotations = read_sigmf_meta(input_path)
    sample_rate_hz = float(global_info.get("core:sample_rate"))
    capture = captures[sigmf_capture_index] if captures else {}
    center_frequency_hz = capture.get("core:frequency", None)
    meta = {
        "sample_rate_hz": sample_rate_hz,
        "center_frequency_hz": None if center_frequency_hz is None else float(center_frequency_hz),
        "annotations": annotations,
    }

    effective_fft_size = int(sigmf_target_freq_bins) if sigmf_target_freq_bins is not None else int(fft_size)
    effective_fft_size = max(effective_fft_size, 16)
    overlap_ratio = float(np.clip(float(noverlap) / max(float(fft_size), 1.0), 0.0, 0.98))
    effective_noverlap = int(round(effective_fft_size * overlap_ratio))
    effective_noverlap = min(max(effective_noverlap, 0), effective_fft_size - 1)

    slice_duration_s = _sigmf_duration_for_time_bins(
        time_bins=sigmf_target_time_bins,
        fft_size=effective_fft_size,
        noverlap=effective_noverlap,
        sample_rate_hz=sample_rate_hz,
    )
    effective_start_s = float(sigmf_window_start_s) + max(int(sigmf_slice_offset_windows), 0) * slice_duration_s

    samples, meta = load_sigmf_samples(
        meta_path=input_path,
        start_s=effective_start_s,
        duration_s=slice_duration_s,
        capture_index=sigmf_capture_index,
        channel=sigmf_channel,
    )
    freq_axis_hz, time_axis_s, sxx_db = generate_spectrogram(
        samples,
        sample_rate_hz=meta["sample_rate_hz"],
        fft_size=effective_fft_size,
        noverlap=effective_noverlap,
        center_frequency_hz=meta["center_frequency_hz"],
    )
    if sxx_db.shape[1] > int(sigmf_target_time_bins):
        sxx_db = np.ascontiguousarray(sxx_db[:, : int(sigmf_target_time_bins)])
        time_axis_s = np.ascontiguousarray(time_axis_s[: int(sigmf_target_time_bins)])

    return {
        "input_kind": "sigmf",
        "input_path": str(input_path),
        "sxx_db": sxx_db,
        "freq_axis_hz": freq_axis_hz,
        "time_axis_s": time_axis_s,
        "center_frequency_hz": meta["center_frequency_hz"],
        "sample_rate_hz": meta["sample_rate_hz"],
        "annotations": meta["annotations"],
        "sigmf_slice_info": {
            "target_time_bins": int(sigmf_target_time_bins),
            "target_freq_bins": int(effective_fft_size),
            "effective_noverlap": int(effective_noverlap),
            "slice_duration_s": float(slice_duration_s),
            "slice_offset_windows": int(sigmf_slice_offset_windows),
            "effective_start_s": float(effective_start_s),
        },
    }


def spectrogram_to_rgb(sxx_db: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> Image.Image:
    from matplotlib import colormaps

    x_db = np.asarray(sxx_db, dtype=np.float32)
    if vmin is None or vmax is None:
        vmin, vmax = robust_fixed_db_window(x_db)
    span = max(float(vmax) - float(vmin), 1e-6)
    x01 = np.clip((x_db - float(vmin)) / span, 0.0, 1.0)
    rgb = (colormaps["viridis"](x01)[..., :3] * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def robust_fixed_db_window(
    sxx_frame_db: np.ndarray,
    low_q: float = 1.0,
    high_q: float = 99.0,
    min_span_db: float = 12.0,
    floor_db: float = -140.0,
    ceil_db: float = -20.0,
) -> tuple[float, float]:
    vals = np.asarray(sxx_frame_db, dtype=np.float32).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -110.0, -98.0

    lo = float(np.percentile(vals, low_q))
    hi = float(np.percentile(vals, high_q))
    if hi < lo:
        lo, hi = hi, lo
    lo = float(np.clip(lo, floor_db, ceil_db))
    hi = float(np.clip(hi, floor_db, ceil_db))
    span = min(max(hi - lo, min_span_db), ceil_db - floor_db)
    center = float(np.clip(0.5 * (lo + hi), floor_db + 0.5 * span, ceil_db - 0.5 * span))
    vmin = center - 0.5 * span
    vmax = center + 0.5 * span
    return float(vmin), float(max(vmax, vmin + 1.0))


def nonlocal_texture_recurrence_mask(img_rgb: Image.Image, patch_h: int, patch_w: int, patch_size: int, k: int = 6, q: float = 0.90):
    gray = np.asarray(img_rgb.convert("L"), dtype=np.float32) / 255.0
    rows = int(patch_h) * int(patch_size)
    cols = int(patch_w) * int(patch_size)
    gray = gray[:rows, :cols]
    patches = gray.reshape(patch_h, patch_size, patch_w, patch_size).transpose(0, 2, 1, 3).reshape(-1, patch_size * patch_size)
    patches = patches - patches.mean(axis=1, keepdims=True)
    patches = patches / np.maximum(patches.std(axis=1, keepdims=True), 1e-6)
    nn = NearestNeighbors(n_neighbors=min(int(k) + 1, patches.shape[0]), metric="cosine")
    nn.fit(patches)
    dist, _ = nn.kneighbors(patches)
    score = (-dist[:, 1:].mean(axis=1)).reshape(patch_h, patch_w).astype(np.float32)
    threshold = float(np.quantile(score, float(np.clip(q, 0.50, 0.99))))
    mask = _smooth_binary_label_map((score >= threshold).astype(np.uint8), iters=1, min_component_size=3)
    return mask.astype(np.uint8), score.astype(np.float32), threshold


def threshold_patch_score(score_patch: np.ndarray, q: float = 0.90):
    score_patch = np.asarray(score_patch, dtype=np.float32)
    threshold = float(np.quantile(score_patch, float(np.clip(q, 0.50, 0.99))))
    return (score_patch >= threshold).astype(np.uint8), threshold


def threshold_like_reference(score_map: np.ndarray, ref_mask: np.ndarray, smooth_iters: int = 1, min_component_size: int = 3):
    score_map = np.asarray(score_map, dtype=np.float32)
    target_fg = float(np.clip(np.mean(ref_mask), 0.02, 0.60))
    threshold = float(np.quantile(score_map, 1.0 - target_fg))
    mask = _smooth_binary_label_map((score_map >= threshold).astype(np.uint8), iters=smooth_iters, min_component_size=min_component_size)
    return mask.astype(np.uint8), threshold, target_fg


def _soft_gate_score_map(
    score_map: np.ndarray,
    gate_patch: np.ndarray,
    floor: float = 0.25,
    score_low_q: float = 5.0,
    score_high_q: float = 95.0,
    gate_low_q: float = 5.0,
    gate_high_q: float = 95.0,
):
    score_n = _normalize_map01_local(score_map, score_low_q, score_high_q)
    gate_n = _normalize_map01_local(gate_patch, gate_low_q, gate_high_q)
    gated = score_n * (float(floor) + (1.0 - float(floor)) * gate_n)
    return _normalize_map01_local(gated, score_low_q, score_high_q).astype(np.float32), score_n.astype(np.float32), gate_n.astype(np.float32)


def apply_coherence_gate_to_power_score(
    power_score_map: np.ndarray,
    reference_mask: np.ndarray,
    coherence_gate_patch: np.ndarray,
    gate_floor: float = 0.25,
    smooth_iters: int = 1,
    min_component_size: int = 3,
):
    raw_score = np.asarray(power_score_map, dtype=np.float32)
    raw_mask = np.asarray(reference_mask, dtype=np.uint8)
    gated_score, raw_score_norm, coherence_gate = _soft_gate_score_map(raw_score, coherence_gate_patch, floor=gate_floor)
    gated_mask, gated_thr, target_fg = threshold_like_reference(
        gated_score,
        raw_mask,
        smooth_iters=smooth_iters,
        min_component_size=min_component_size,
    )
    return {
        "score": gated_score.astype(np.float32),
        "mask": gated_mask.astype(np.uint8),
        "threshold": float(gated_thr),
        "target_foreground_fraction": float(target_fg),
        "raw_score_norm": raw_score_norm.astype(np.float32),
        "coherence_gate_patch": coherence_gate.astype(np.float32),
    }


def _pipeline_safe_quantile(arr: np.ndarray, q: float) -> float:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, q))


def _pipeline_component_speckle_score(mask: np.ndarray, min_size: int = 6) -> float:
    mask = np.asarray(mask, dtype=np.uint8)
    total = int(mask.sum())
    if total == 0:
        return 0.0
    labels, n_labels = ndi.label(mask)
    if n_labels == 0:
        return 0.0
    counts = np.bincount(labels.ravel())
    if counts.size == 0:
        return 0.0
    counts[0] = 0
    small_pixels = int(counts[(counts > 0) & (counts < int(min_size))].sum())
    return float(np.clip(1.0 - (small_pixels / max(total, 1)), 0.0, 1.0))


def _pipeline_texture_reliability(texture_map: np.ndarray, agreement_map: np.ndarray) -> float:
    eps = 1e-6
    texture_map = np.asarray(texture_map, dtype=np.float32)
    agreement_map = np.asarray(agreement_map, dtype=np.float32)
    q20 = _pipeline_safe_quantile(texture_map, 0.20)
    q60 = _pipeline_safe_quantile(texture_map, 0.60)
    q85 = _pipeline_safe_quantile(texture_map, 0.85)
    q95 = _pipeline_safe_quantile(texture_map, 0.95)
    peakiness = float(np.clip((q95 - q60) / max(q95 - q20, eps), 0.0, 1.0))
    texture_top_mask = texture_map >= q85
    speckle_score = _pipeline_component_speckle_score(texture_top_mask, min_size=6)
    coherence_support = float(np.mean(agreement_map[texture_top_mask])) if texture_top_mask.any() else 0.0
    agreement_top_mask = agreement_map >= _pipeline_safe_quantile(agreement_map, 0.85)
    union = texture_top_mask | agreement_top_mask
    overlap = float(np.logical_and(texture_top_mask, agreement_top_mask).sum()) / float(union.sum()) if union.any() else 0.0
    return float(np.clip(0.35 * peakiness + 0.25 * speckle_score + 0.20 * coherence_support + 0.20 * overlap, 0.0, 1.0))


def _pipeline_speckle_transition_thresholds(clean_threshold: float, strong_threshold: float) -> tuple[float, float]:
    clean_thr = float(np.clip(clean_threshold, 1e-3, 1.0))
    strong_thr = float(np.clip(strong_threshold, 0.0, clean_thr - 1e-3))
    if strong_thr >= clean_thr:
        strong_thr = max(0.0, clean_thr - 1e-3)
    return strong_thr, clean_thr


def _pipeline_structure_texture_passthrough(
    structure_map: np.ndarray,
    reliability: float,
    clean_threshold: float,
    strong_threshold: float,
) -> dict[str, float]:
    structure_map = np.asarray(structure_map, dtype=np.float32)
    q70 = _pipeline_safe_quantile(structure_map, 0.70)
    q90 = _pipeline_safe_quantile(structure_map, 0.90)
    structure_top_mask = structure_map >= q70
    structure_peak_mask = structure_map >= q90
    min_size = max(4, int(round(4 + 4 * float(reliability))))
    structure_speckle = _pipeline_component_speckle_score(structure_top_mask, min_size=min_size)
    structure_peak = float(np.mean(structure_map[structure_peak_mask])) if structure_peak_mask.any() else 0.0
    structure_clean_score = float(np.clip(0.70 * structure_speckle + 0.30 * structure_peak, 0.0, 1.0))
    speckle_strong_thr, speckle_clean_thr = _pipeline_speckle_transition_thresholds(clean_threshold, strong_threshold)
    if structure_speckle >= speckle_clean_thr:
        clean_ramp = float(np.clip((structure_speckle - speckle_clean_thr) / max(1.0 - speckle_clean_thr, 1e-6), 0.0, 1.0))
        texture_passthrough = 0.25 + 0.75 * clean_ramp
    elif structure_speckle >= speckle_strong_thr:
        speckle_ramp = float(np.clip((structure_speckle - speckle_strong_thr) / max(speckle_clean_thr - speckle_strong_thr, 1e-6), 0.0, 1.0))
        texture_passthrough = 0.05 + 0.20 * speckle_ramp
    else:
        texture_passthrough = 0.05
    return {
        "structure_speckle_score": float(structure_speckle),
        "structure_peak_score": float(structure_peak),
        "structure_clean_score": float(structure_clean_score),
        "texture_passthrough": float(np.clip(texture_passthrough, 0.05, 1.0)),
    }


def build_multilevel_gap_pipeline(
    score_map: np.ndarray,
    structure_map: np.ndarray,
    texture_map: np.ndarray,
    reliability: float,
    gap_floor: float,
    raw_power_map: np.ndarray,
    power_rescue_floor: float,
    power_rescue_gain: float,
    clean_threshold: float,
    strong_threshold: float,
) -> dict[str, Any]:
    score_map = np.asarray(score_map, dtype=np.float32)
    structure_map = np.asarray(structure_map, dtype=np.float32)
    texture_map = np.asarray(texture_map, dtype=np.float32)
    raw_power_map = np.asarray(raw_power_map, dtype=np.float32)

    strong_thr = float(0.44 + 0.12 * reliability)
    mid_thr = float(0.24 + 0.10 * reliability)
    low_thr = float(0.10 + 0.08 * reliability)
    structure_thr = float(0.45 + 0.12 * reliability)
    score_pass_gate = (score_map >= low_thr).astype(np.float32)

    strong_mid_overlap = float(np.clip(0.15 * max(strong_thr - mid_thr, 0.0), 0.012, 0.035))
    mid_low_overlap = float(np.clip(0.15 * max(mid_thr - low_thr, 0.0), 0.012, 0.030))
    mid_upper = float(min(1.0, strong_thr + strong_mid_overlap))
    low_upper = float(min(1.0, mid_thr + mid_low_overlap))

    raw_power_local_mean = ndi.uniform_filter(_normalize_map01_local(raw_power_map, 5.0, 95.0), size=3, mode="nearest").astype(np.float32)
    power_rescue_floor = float(np.clip(power_rescue_floor, 0.0, 0.99))
    power_rescue_gain = float(np.clip(power_rescue_gain, 0.0, 2.0))
    power_rescue_term = np.clip(
        (raw_power_local_mean - power_rescue_floor) / max(1.0 - power_rescue_floor, 1e-6),
        0.0,
        1.0,
    ).astype(np.float32)
    structure_rescue_span = float(max(structure_thr - low_thr, 0.05))
    power_weighted_structure = np.clip(
        structure_map + power_rescue_gain * structure_rescue_span * (score_pass_gate * power_rescue_term),
        0.0,
        1.0,
    ).astype(np.float32)

    strong_candidate_mask = (score_map >= strong_thr).astype(np.uint8)
    mid_candidate_mask = np.logical_and(score_map >= mid_thr, score_map < mid_upper).astype(np.uint8)
    low_candidate_mask = np.logical_and(score_map >= low_thr, score_map < low_upper).astype(np.uint8)

    strong_score = (score_map * strong_candidate_mask.astype(np.float32)).astype(np.float32)
    mid_score = (score_map * mid_candidate_mask.astype(np.float32)).astype(np.float32)
    low_candidate_score = (score_map * low_candidate_mask.astype(np.float32)).astype(np.float32)

    texture_policy = _pipeline_structure_texture_passthrough(
        structure_map=structure_map,
        reliability=reliability,
        clean_threshold=clean_threshold,
        strong_threshold=strong_threshold,
    )
    texture_weight = float(np.clip(texture_policy["texture_passthrough"], gap_floor, 1.0))
    speckle_strong_thr, speckle_clean_thr = _pipeline_speckle_transition_thresholds(clean_threshold, strong_threshold)
    speckle_score = float(texture_policy["structure_speckle_score"])
    band_weight = float(np.clip((speckle_score - speckle_strong_thr) / max(speckle_clean_thr - speckle_strong_thr, 1e-6), 0.0, 1.0))

    norm_strong = _normalize_map01_local(strong_score, 5.0, 95.0)
    norm_mid = _normalize_map01_local(mid_score, 5.0, 95.0)
    norm_low = _normalize_map01_local(low_candidate_score, 5.0, 95.0)
    combined_band_score = np.clip(norm_strong + band_weight * norm_mid + band_weight * norm_low, 0.0, 1.0).astype(np.float32)
    texture_passthrough_score = (texture_weight * texture_map).astype(np.float32)
    pre_gap_score = np.clip(combined_band_score + texture_passthrough_score, 0.0, 1.0).astype(np.float32)
    final_map = _normalize_map01_local(pre_gap_score, 5.0, 95.0).astype(np.float32)

    return {
        "strong_thr": strong_thr,
        "mid_thr": mid_thr,
        "low_thr": low_thr,
        "structure_thr": structure_thr,
        "raw_power_local_mean": raw_power_local_mean,
        "power_rescue_term": power_rescue_term,
        "power_weighted_structure": power_weighted_structure,
        "strong_score": strong_score,
        "mid_score": mid_score,
        "low_candidate_score": low_candidate_score,
        "combined_band_score": combined_band_score,
        "texture_passthrough_score": texture_passthrough_score,
        "pre_gap_score": pre_gap_score,
        "final_map": final_map,
        "texture_passthrough": texture_weight,
        "texture_penalty_strength": float(1.0 - texture_weight),
        "structure_speckle_score": float(texture_policy["structure_speckle_score"]),
        "structure_peak_score": float(texture_policy["structure_peak_score"]),
        "structure_clean_score": float(texture_policy["structure_clean_score"]),
        "band_weight": band_weight,
    }


def _remove_small_components(mask: np.ndarray, min_component_size: int) -> np.ndarray:
    out = np.asarray(mask, dtype=np.uint8).copy()
    if int(min_component_size) <= 1:
        return out
    comp, n_comp = ndimage.label(out)
    if n_comp <= 0:
        return out
    sizes = ndimage.sum(out, comp, index=np.arange(1, n_comp + 1))
    small_ids = np.where(np.asarray(sizes) < int(min_component_size))[0] + 1
    if len(small_ids) > 0:
        out[np.isin(comp, small_ids)] = 0
    return out.astype(np.uint8)


def run_chunk_detector_v3(sxx_db_chunk: np.ndarray, model, patch_size: int, device: str, cfg: HoloscanChunkConfig) -> dict[str, Any]:
    dino_group = dino_grouping_from_spectrogram(
        sxx_db_chunk,
        model=model,
        patch_size=patch_size,
        device=device,
        db_min=cfg.dino_db_min,
        db_max=cfg.dino_db_max,
        feature_knn=cfg.dino_group_k,
        spatial_weight=cfg.dino_group_spatial_weight,
        score_q=cfg.dino_group_score_q,
        min_component_size=cfg.min_component_size,
    )
    dino_gated = apply_coherence_gate_to_dino_result(
        dino_group,
        sxx_db_chunk,
        gate_floor=cfg.dino_coherence_gate_floor,
        min_component_size=max(3, cfg.min_component_size // 2),
    )
    patch_h, patch_w = tuple(dino_gated["shape"])
    chunk_rgb = spectrogram_to_rgb(sxx_db_chunk)
    texture_mask, texture_score, texture_thr = nonlocal_texture_recurrence_mask(
        chunk_rgb,
        patch_h=patch_h,
        patch_w=patch_w,
        patch_size=patch_size,
        k=cfg.texture_k,
        q=cfg.texture_q,
    )
    power_raw_score = power_prior_patch_map(sxx_db_chunk, patch_h, patch_w)
    power_raw_mask, power_raw_thr = threshold_patch_score(power_raw_score, q=cfg.power_q)
    power_gated = apply_coherence_gate_to_power_score(
        power_raw_score,
        power_raw_mask,
        dino_gated["coherence_gate_patch"],
        gate_floor=cfg.dino_coherence_gate_floor,
        smooth_iters=1,
        min_component_size=max(3, cfg.min_component_size // 2),
    )
    burst_gate = burst_companion_gate(
        sxx_db_chunk,
        coherence_gate_px=dino_gated["coherence_gate_px"],
        residual_px=dino_gated["coherence_residual_px"],
        patch_h=patch_h,
        patch_w=patch_w,
        residual_gain=cfg.companion_residual_gain,
        temporal_contrast_gain=cfg.companion_temporal_contrast_gain,
        coherence_gain=cfg.companion_coherence_gain,
    )

    dino_map = _normalize_map01_local(dino_gated["score"], 5.0, 95.0)
    power_map = _normalize_map01_local(power_gated["score"], 5.0, 95.0)
    texture_map = _normalize_map01_local(texture_score, 5.0, 95.0)
    structure_map = _normalize_map01_local(dino_gated["coherence_gate_patch"], 5.0, 95.0)
    agreement_map = np.sqrt(np.clip(dino_map * power_map, 0.0, None)).astype(np.float32)
    texture_reliability = _pipeline_texture_reliability(texture_map, agreement_map)

    pipeline = build_multilevel_gap_pipeline(
        agreement_map,
        structure_map,
        texture_map,
        reliability=texture_reliability,
        gap_floor=cfg.pipeline_gap_floor,
        raw_power_map=power_raw_score,
        power_rescue_floor=cfg.pipeline_power_rescue_floor,
        power_rescue_gain=cfg.pipeline_power_rescue_gain,
        clean_threshold=cfg.texture_speckle_clean_threshold,
        strong_threshold=cfg.texture_speckle_strong_threshold,
    )

    speckle_active = float(pipeline["band_weight"]) < 1.0
    final_threshold = cfg.pipeline_final_threshold if speckle_active else cfg.pipeline_final_threshold_no_speckle
    final_min_component = cfg.pipeline_strong_speckle_min_component if speckle_active else cfg.pipeline_component_min_size_no_speckle
    final_mask_patch = _remove_small_components((pipeline["final_map"] >= float(final_threshold)).astype(np.uint8), final_min_component)

    merge_companion_patch = _normalize_map01_local(
        float(pipeline["texture_passthrough"]) * texture_map
        + float(1.0 - pipeline["texture_passthrough"]) * burst_gate["companion_patch"],
        5.0,
        95.0,
    )
    support_patch = _normalize_map01_local(
        0.55 * pipeline["final_map"] + 0.25 * agreement_map + 0.20 * merge_companion_patch,
        5.0,
        95.0,
    )
    support_thr = float(np.quantile(support_patch, float(np.clip(cfg.merge_support_q, 0.50, 0.95))))
    support_mask_patch = _smooth_binary_label_map(
        np.logical_or(support_patch >= support_thr, final_mask_patch > 0).astype(np.uint8),
        iters=1,
        min_component_size=max(3, cfg.min_component_size // 2),
    )

    score_px = _resize_patch_map_to_pixels(pipeline["final_map"], sxx_db_chunk.shape[0], sxx_db_chunk.shape[1], Image.BILINEAR)
    mask_px = _resize_patch_mask_to_pixels(final_mask_patch, sxx_db_chunk.shape[0], sxx_db_chunk.shape[1])
    support_px = _resize_patch_mask_to_pixels(support_mask_patch, sxx_db_chunk.shape[0], sxx_db_chunk.shape[1])
    companion_px = _resize_patch_map_to_pixels(merge_companion_patch, sxx_db_chunk.shape[0], sxx_db_chunk.shape[1], Image.BILINEAR)

    return {
        "dino_group": dino_group,
        "dino_gated": dino_gated,
        "texture_mask_patch": texture_mask.astype(np.uint8),
        "texture_score": texture_score.astype(np.float32),
        "texture_threshold": float(texture_thr),
        "texture_reliability": float(texture_reliability),
        "power_raw_score": power_raw_score.astype(np.float32),
        "power_raw_mask": power_raw_mask.astype(np.uint8),
        "power_raw_threshold": float(power_raw_thr),
        "power_gated": power_gated,
        "burst_gate": burst_gate,
        "chunk_rgb": chunk_rgb,
        "agreement_map": agreement_map.astype(np.float32),
        "structure_map": structure_map.astype(np.float32),
        "pipeline": pipeline,
        "speckle_active": bool(speckle_active),
        "final_threshold": float(final_threshold),
        "final_min_component": int(final_min_component),
        "final_mask_patch": final_mask_patch.astype(np.uint8),
        "support_patch": support_patch.astype(np.float32),
        "support_threshold": float(support_thr),
        "support_mask_patch": support_mask_patch.astype(np.uint8),
        "score_px": score_px.astype(np.float32),
        "mask_px": mask_px.astype(bool),
        "support_px": support_px.astype(bool),
        "companion_px": companion_px.astype(np.float32),
    }


def run_chunked_pipeline_v3(
    input_record: dict[str, Any],
    model,
    patch_size: int,
    device: str,
    cfg: HoloscanChunkConfig,
) -> dict[str, Any]:
    ignore_info = compute_ignore_sideband_rows(
        input_record["freq_axis_hz"],
        ignore_sideband_percent=cfg.ignore_sideband_percent,
        min_keep_rows=max(int(patch_size), 16),
        ignore_sideband_hz=cfg.ignore_sideband_hz,
    )
    valid_row_mask = np.asarray(ignore_info["valid_row_mask"], dtype=bool)
    correction = apply_global_frontend_correction(
        input_record["sxx_db"],
        row_q=cfg.frontend_row_q,
        reference_q=cfg.frontend_reference_q,
        smooth_sigma=cfg.frontend_smooth_sigma,
        max_boost_db=cfg.frontend_max_boost_db,
        valid_row_mask=valid_row_mask,
    )
    corrected_sxx_db = np.asarray(correction["corrected_sxx_db"], dtype=np.float32)
    chunk_plan = build_frequency_chunks(
        input_record["freq_axis_hz"],
        chunk_bandwidth_hz=cfg.chunk_bandwidth_hz,
        chunk_overlap_hz=cfg.chunk_overlap_hz,
        min_rows=max(int(patch_size), 16),
        valid_row_mask=valid_row_mask,
    )
    chunk_results: list[dict[str, Any]] = []
    for chunk in chunk_plan:
        row_slice = slice(chunk["row_start"], chunk["row_stop"])
        detection = run_chunk_detector_v3(corrected_sxx_db[row_slice, :], model, patch_size, device, cfg)
        chunk_results.append({**chunk, **detection, "valid_row_mask": valid_row_mask})
    merged = merge_chunk_results(
        corrected_sxx_db.shape,
        chunk_results,
        final_score_q=cfg.final_score_q,
        min_component_size=cfg.min_component_size,
    )
    return {
        "input_record": input_record,
        "config": cfg,
        "frontend": correction,
        "corrected_sxx_db": corrected_sxx_db,
        "ignore_sideband": ignore_info,
        "chunk_plan": chunk_plan,
        "chunk_results": chunk_results,
        **merged,
    }


def summarize_chunk_policy(pipeline_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk in pipeline_result.get("chunk_results", []):
        policy = chunk["pipeline"]
        rows.append(
            {
                "chunk_index": int(chunk["chunk_index"]),
                "row_start": int(chunk["row_start"]),
                "row_stop": int(chunk["row_stop"]),
                "freq_start_mhz": float(chunk["freq_start_hz"]) / 1e6,
                "freq_stop_mhz": float(chunk["freq_stop_hz"]) / 1e6,
                "texture_reliability": float(chunk["texture_reliability"]),
                "structure_speckle_score": float(policy["structure_speckle_score"]),
                "texture_passthrough": float(policy["texture_passthrough"]),
                "band_weight": float(policy["band_weight"]),
                "speckle_active": bool(chunk["speckle_active"]),
                "final_threshold": float(chunk["final_threshold"]),
            }
        )
    return rows


def plot_chunk_policy_examples(pipeline_result: dict[str, Any], max_chunks: int = 4, figsize: tuple[int, int] = (28, 5)):
    chunk_results = pipeline_result.get("chunk_results", [])[: max(1, int(max_chunks))]
    fig, axes = plt.subplots(len(chunk_results), 5, figsize=(figsize[0], figsize[1] * len(chunk_results)), constrained_layout=True)
    if len(chunk_results) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_idx, chunk in enumerate(chunk_results):
        tile = pipeline_result["corrected_sxx_db"][chunk["row_start"]:chunk["row_stop"], :]
        axes[row_idx][0].imshow(tile, aspect="auto", origin="lower", cmap="viridis")
        axes[row_idx][0].set_title(f"Chunk {chunk['chunk_index']} corrected tile")
        axes[row_idx][1].imshow(chunk["chunk_rgb"])
        axes[row_idx][1].set_title("Texture source")
        axes[row_idx][2].imshow(chunk["texture_score"], cmap="cividis", vmin=0.0, vmax=1.0)
        axes[row_idx][2].set_title(f"Texture score | rel={chunk['texture_reliability']:.2f}")
        axes[row_idx][3].imshow(chunk["agreement_map"], cmap="magma", vmin=0.0, vmax=1.0)
        axes[row_idx][3].set_title("DINO-power agreement")
        axes[row_idx][4].imshow(chunk["pipeline"]["final_map"], cmap="magma", vmin=0.0, vmax=1.0)
        axes[row_idx][4].imshow(np.where(chunk["final_mask_patch"] > 0, 1.0, np.nan), cmap="autumn", alpha=0.35, vmin=0.0, vmax=1.0)
        axes[row_idx][4].set_title(f"Final map | tex={chunk['pipeline']['texture_passthrough']:.2f}")
        for ax in axes[row_idx]:
            ax.set_xticks([])
            ax.set_yticks([])
    return fig, axes


__all__ = [
    "HoloscanChunkConfig",
    "infer_expected_frequency_bins",
    "infer_expected_spectrogram_shape",
    "infer_expected_time_bins",
    "load_dino_model",
    "load_input_record",
    "plot_chunk_policy_examples",
    "plot_frontend_overview",
    "plot_merged_debug",
    "plot_merged_detection",
    "run_chunked_pipeline_v3",
    "summarize_chunk_policy",
]