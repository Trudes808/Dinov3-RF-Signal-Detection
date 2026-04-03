from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy import ndimage, signal
from scipy import ndimage as ndi
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class WidebandChunkConfig:
    chunk_bandwidth_hz: float = 50e6
    chunk_overlap_hz: float = 10e6
    ignore_sideband_hz: float = 0.0
    dino_db_min: float = -110.0
    dino_db_max: float = -40.0
    dino_group_k: int = 8
    dino_group_spatial_weight: float = 0.35
    dino_group_score_q: float = 0.60
    dino_coherence_gate_floor: float = 0.25
    power_fusion_gain: float = 0.25
    final_score_q: float = 0.85
    merge_support_q: float = 0.70
    merge_companion_floor: float = 0.30
    companion_residual_gain: float = 0.45
    companion_temporal_contrast_gain: float = 0.35
    companion_coherence_gain: float = 0.20
    min_component_size: int = 6
    frontend_row_q: float = 25.0
    frontend_reference_q: float = 75.0
    frontend_smooth_sigma: float = 12.0
    frontend_max_boost_db: float = 12.0
    parallel_backend: str = "serial"
    max_workers: int = 1


def infer_input_kind(input_path: str | Path, explicit_kind: str = "auto") -> str:
    if explicit_kind in {"pgm", "sigmf"}:
        return explicit_kind
    suffix = Path(input_path).suffix.lower()
    if suffix == ".pgm":
        return "pgm"
    if suffix == ".sigmf-meta":
        return "sigmf"
    raise ValueError(f"Unsupported input type for {input_path}")


def infer_channel_from_filename(path: str | Path) -> int | None:
    match = re.search(r"_ch(\d+)_|ch(\d+)", str(path))
    if not match:
        return None
    for group in match.groups():
        if group is not None:
            return int(group)
    return None


def load_usrp_spectrogram_summary(summary_path: str | Path | None) -> dict[str, Any] | None:
    if summary_path is None:
        return None
    summary_file = Path(summary_path)
    if not summary_file.exists():
        return None
    with summary_file.open("r") as f:
        return json.load(f)


def read_pgm(path: str | Path) -> np.ndarray:
    arr = np.asarray(Image.open(path), dtype=np.float32)
    return np.ascontiguousarray(arr.T)


def read_sigmf_meta(meta_path: str | Path):
    meta_path = Path(meta_path)
    with meta_path.open("r") as f:
        meta = json.load(f)
    global_info = meta.get("global", {})
    captures = meta.get("captures", [])
    annotations = meta.get("annotations", [])
    return meta, global_info, captures, annotations


def _sigmf_dtype_info(datatype: str):
    if not datatype:
        raise ValueError("SigMF datatype is missing")
    if datatype.endswith("_le"):
        endian = "<"
        base = datatype[:-3]
    elif datatype.endswith("_be"):
        endian = ">"
        base = datatype[:-3]
    else:
        endian = "<"
        base = datatype

    is_complex = base.startswith("c")
    scalar_spec = base[1:] if is_complex else base
    scalar_kind = scalar_spec[0]
    bits = int(scalar_spec[1:])
    bytes_per = bits // 8
    kind_map = {"i": "i", "u": "u", "f": "f"}
    if scalar_kind not in kind_map:
        raise ValueError(f"Unsupported SigMF datatype: {datatype}")
    dtype = np.dtype(f"{endian}{kind_map[scalar_kind]}{bytes_per}")
    return dtype, is_complex


def _load_sigmf_iq(data_path: str | Path, dtype, is_complex: bool, start_sample: int, count: int | None, num_channels: int = 1, channel: int = 0):
    data_path = Path(data_path)
    bytes_per_scalar = dtype.itemsize
    scalars_per_sample = (2 if is_complex else 1) * num_channels
    file_size = data_path.stat().st_size
    total_samples = file_size // (bytes_per_scalar * scalars_per_sample)
    if start_sample < 0 or start_sample >= total_samples:
        raise ValueError("start_sample is outside file bounds")
    if count is None:
        count = total_samples - start_sample
    count = min(count, total_samples - start_sample)
    scalar_start = start_sample * scalars_per_sample
    scalar_count = count * scalars_per_sample
    data = np.memmap(
        data_path,
        dtype=dtype,
        mode="r",
        offset=scalar_start * bytes_per_scalar,
        shape=(scalar_count,),
    )
    if is_complex:
        data = data.reshape(-1, num_channels, 2)
        i = data[:, channel, 0].astype(np.float32)
        q = data[:, channel, 1].astype(np.float32)
        return np.asarray(i + 1j * q)
    data = data.reshape(-1, num_channels)
    return np.asarray(data[:, channel].astype(np.float32))


def load_sigmf_samples(meta_path: str | Path, start_s: float = 0.0, duration_s: float | None = 1.0, capture_index: int = 0, channel: int = 0):
    _, global_info, captures, annotations = read_sigmf_meta(meta_path)
    sample_rate = float(global_info.get("core:sample_rate"))
    datatype = global_info.get("core:datatype")
    num_channels = int(global_info.get("core:num_channels", 1))
    capture = captures[capture_index] if captures else {}
    capture_start = int(capture.get("core:sample_start", 0))
    center_frequency = capture.get("core:frequency", None)
    dtype, is_complex = _sigmf_dtype_info(datatype)
    start_sample = capture_start + int(start_s * sample_rate)
    count = int(duration_s * sample_rate) if duration_s is not None else None
    data_path = str(meta_path).replace(".sigmf-meta", ".sigmf-data")
    samples = _load_sigmf_iq(
        data_path=data_path,
        dtype=dtype,
        is_complex=is_complex,
        start_sample=start_sample,
        count=count,
        num_channels=num_channels,
        channel=channel,
    )
    return samples, {
        "sample_rate_hz": sample_rate,
        "center_frequency_hz": None if center_frequency is None else float(center_frequency),
        "annotations": annotations,
    }


def generate_spectrogram(iq_data: np.ndarray, sample_rate_hz: float, fft_size: int = 1024, noverlap: int = 512, center_frequency_hz: float | None = None):
    freq_axis_hz, time_axis_s, sxx = signal.spectrogram(
        iq_data,
        fs=sample_rate_hz,
        nperseg=fft_size,
        noverlap=noverlap,
        return_onesided=False,
    )
    sxx = np.fft.fftshift(sxx, axes=0)
    freq_axis_hz = np.fft.fftshift(freq_axis_hz)
    if center_frequency_hz is not None:
        freq_axis_hz = freq_axis_hz + center_frequency_hz
    sxx_db = 10.0 * np.log10(sxx + 1e-10)
    return freq_axis_hz.astype(np.float32), time_axis_s.astype(np.float32), sxx_db.astype(np.float32)


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
) -> dict[str, Any]:
    input_path = Path(input_path)
    resolved_kind = infer_input_kind(input_path, input_kind)
    if resolved_kind == "pgm":
        summary = load_usrp_spectrogram_summary(usrp_summary_path)
        channel = infer_channel_from_filename(input_path)
        sxx_db = read_pgm(input_path)
        time_axis_s = np.arange(sxx_db.shape[1], dtype=np.float32)
        center_frequency_hz = None
        sample_rate_hz = None
        if summary is not None and channel is not None:
            cfg = summary.get("channel_configs", {}).get(str(channel), {})
            sample_rate_hz = float(cfg.get("sample_rate_hz", 0.0)) if cfg.get("sample_rate_hz") is not None else None
            center_frequency_hz = float(cfg.get("center_freq_hz", 0.0)) if cfg.get("center_freq_hz") is not None else None
        if sample_rate_hz is not None and sample_rate_hz > 0:
            half_span = 0.5 * sample_rate_hz
            freq_axis_hz = np.linspace(-half_span, half_span, sxx_db.shape[0], endpoint=False, dtype=np.float32)
            if center_frequency_hz is not None:
                freq_axis_hz = freq_axis_hz + center_frequency_hz
        else:
            freq_axis_hz = np.arange(sxx_db.shape[0], dtype=np.float32)
        return {
            "input_kind": "pgm",
            "input_path": str(input_path),
            "sxx_db": sxx_db.astype(np.float32),
            "freq_axis_hz": freq_axis_hz.astype(np.float32),
            "time_axis_s": time_axis_s,
            "center_frequency_hz": center_frequency_hz,
            "sample_rate_hz": sample_rate_hz,
            "annotations": [],
        }

    samples, meta = load_sigmf_samples(
        meta_path=input_path,
        start_s=sigmf_window_start_s,
        duration_s=sigmf_window_duration_s,
        capture_index=sigmf_capture_index,
        channel=sigmf_channel,
    )
    freq_axis_hz, time_axis_s, sxx_db = generate_spectrogram(
        samples,
        sample_rate_hz=meta["sample_rate_hz"],
        fft_size=fft_size,
        noverlap=noverlap,
        center_frequency_hz=meta["center_frequency_hz"],
    )
    return {
        "input_kind": "sigmf",
        "input_path": str(input_path),
        "sxx_db": sxx_db,
        "freq_axis_hz": freq_axis_hz,
        "time_axis_s": time_axis_s,
        "center_frequency_hz": meta["center_frequency_hz"],
        "sample_rate_hz": meta["sample_rate_hz"],
        "annotations": meta["annotations"],
    }


def apply_global_frontend_correction(
    sxx_db: np.ndarray,
    row_q: float = 25.0,
    reference_q: float = 75.0,
    smooth_sigma: float = 12.0,
    max_boost_db: float = 12.0,
    valid_row_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    sxx_db = np.asarray(sxx_db, dtype=np.float32)
    if valid_row_mask is None:
        valid_row_mask = np.ones(sxx_db.shape[0], dtype=bool)
    else:
        valid_row_mask = np.asarray(valid_row_mask, dtype=bool).reshape(-1)
        if valid_row_mask.shape[0] != sxx_db.shape[0]:
            raise ValueError("valid_row_mask length must match the number of spectrogram rows")
    if not np.any(valid_row_mask):
        raise ValueError("valid_row_mask excludes all rows")
    row_floor_db = np.percentile(sxx_db, row_q, axis=1).astype(np.float32)
    response_db = ndimage.gaussian_filter1d(row_floor_db, sigma=max(float(smooth_sigma), 1.0), mode="nearest").astype(np.float32)
    reference_db = float(np.percentile(response_db[valid_row_mask], reference_q))
    boost_db = np.clip(reference_db - response_db, 0.0, float(max_boost_db)).astype(np.float32)
    corrected_sxx_db = (sxx_db + boost_db[:, None]).astype(np.float32)
    return {
        "row_floor_db": row_floor_db,
        "response_db": response_db,
        "reference_db": reference_db,
        "boost_db": boost_db,
        "corrected_sxx_db": corrected_sxx_db,
        "valid_row_mask": valid_row_mask.astype(bool),
    }


def compute_ignore_sideband_rows(freq_axis_hz: np.ndarray, ignore_sideband_hz: float, min_keep_rows: int = 16) -> dict[str, float | int | np.ndarray]:
    freq_axis_hz = np.asarray(freq_axis_hz, dtype=np.float32).reshape(-1)
    num_rows = int(freq_axis_hz.size)
    info: dict[str, float | int | np.ndarray] = {
        "requested_hz": float(max(0.0, ignore_sideband_hz)),
        "requested_bins": 0,
        "applied_hz": 0.0,
        "applied_bins": 0,
        "bin_hz": 0.0,
        "valid_row_mask": np.ones(num_rows, dtype=bool),
    }
    if num_rows < 2 or ignore_sideband_hz <= 0.0:
        return info

    bin_hz = float(np.median(np.abs(np.diff(freq_axis_hz))))
    if not np.isfinite(bin_hz) or bin_hz <= 0.0:
        return info

    requested_bins = int(np.ceil(float(ignore_sideband_hz) / bin_hz))
    max_bins = max(0, (num_rows - int(max(1, min_keep_rows))) // 2)
    applied_bins = int(np.clip(requested_bins, 0, max_bins))
    valid_row_mask = np.ones(num_rows, dtype=bool)
    if applied_bins > 0:
        valid_row_mask[:applied_bins] = False
        valid_row_mask[-applied_bins:] = False

    info.update(
        {
            "requested_bins": int(requested_bins),
            "applied_hz": float(applied_bins * bin_hz),
            "applied_bins": int(applied_bins),
            "bin_hz": bin_hz,
            "valid_row_mask": valid_row_mask,
        }
    )
    return info


def build_frequency_chunks(freq_axis_hz: np.ndarray, chunk_bandwidth_hz: float, chunk_overlap_hz: float, min_rows: int = 16, valid_row_mask: np.ndarray | None = None) -> list[dict[str, Any]]:
    freq_axis_hz = np.asarray(freq_axis_hz, dtype=np.float32).reshape(-1)
    if freq_axis_hz.size == 0:
        return []
    if valid_row_mask is None:
        valid_row_mask = np.ones(freq_axis_hz.shape[0], dtype=bool)
    else:
        valid_row_mask = np.asarray(valid_row_mask, dtype=bool).reshape(-1)
        if valid_row_mask.shape[0] != freq_axis_hz.shape[0]:
            raise ValueError("valid_row_mask length must match freq_axis_hz length")
    valid_idx = np.flatnonzero(valid_row_mask)
    if valid_idx.size == 0:
        return []
    valid_freq_axis_hz = freq_axis_hz[valid_idx]
    freq_min = float(np.min(valid_freq_axis_hz))
    freq_max = float(np.max(valid_freq_axis_hz))
    if chunk_bandwidth_hz <= 0:
        raise ValueError("chunk_bandwidth_hz must be positive")
    step_hz = chunk_bandwidth_hz - chunk_overlap_hz
    if step_hz <= 0:
        raise ValueError("chunk_bandwidth_hz must be larger than chunk_overlap_hz")

    chunks: list[dict[str, Any]] = []
    chunk_start_hz = freq_min
    chunk_index = 0
    while chunk_start_hz < freq_max + 1e-6:
        chunk_stop_hz = min(chunk_start_hz + chunk_bandwidth_hz, freq_max)
        in_chunk = valid_idx[(valid_freq_axis_hz >= chunk_start_hz) & (valid_freq_axis_hz <= chunk_stop_hz)]
        if in_chunk.size >= int(min_rows):
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "row_start": int(in_chunk[0]),
                    "row_stop": int(in_chunk[-1]) + 1,
                    "freq_start_hz": float(freq_axis_hz[in_chunk[0]]),
                    "freq_stop_hz": float(freq_axis_hz[in_chunk[-1]]),
                }
            )
            chunk_index += 1
        if chunk_stop_hz >= freq_max:
            break
        chunk_start_hz += step_hz
    return chunks


def load_dino_model(
    dino_repo_dir: str | Path,
    model_name: str,
    weights_path: str | Path,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    repo_dir = Path(dino_repo_dir)
    weights_file = Path(weights_path)
    if not weights_file.exists():
        raise FileNotFoundError(f"DINO weights not found: {weights_file}")
    if repo_dir.exists() and (repo_dir / "hubconf.py").exists():
        model = torch.hub.load(
            repo_or_dir=str(repo_dir),
            model=model_name,
            source="local",
            weights=str(weights_file),
        )
    else:
        model = torch.hub.load(
            repo_or_dir="facebookresearch/dinov3",
            model=model_name,
            source="github",
            weights=str(weights_file),
        )
    model = model.to(device).eval()
    patch_size = int(getattr(model, "patch_size", 16))
    return model, patch_size, device


def _prep_dino_image(img_rgb: Image.Image, patch_size: int) -> Image.Image:
    width = (img_rgb.size[0] // patch_size) * patch_size
    height = (img_rgb.size[1] // patch_size) * patch_size
    return img_rgb.crop((0, 0, width, height))


def _extract_dino_features_from_rgb(img_rgb: Image.Image, model, patch_size: int, device: str):
    img_rgb = _prep_dino_image(img_rgb, patch_size)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    x = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.get_intermediate_layers(x, n=1, reshape=True, norm=True)[0]
    feat = feat.squeeze().view(feat.shape[1], -1).permute(1, 0).cpu().numpy()
    grid_h = img_rgb.size[1] // patch_size
    grid_w = img_rgb.size[0] // patch_size
    return feat.astype(np.float32), int(grid_h), int(grid_w), img_rgb


def _normalize_map01_local(x: np.ndarray, low_q: float = 5.0, high_q: float = 95.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo = float(np.percentile(vals, low_q))
    hi = float(np.percentile(vals, high_q))
    if hi <= lo:
        hi = lo + 1e-6
    out = (x - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _normalize_vector01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x.astype(np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi <= lo + 1e-8:
        return np.ones_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _robust_high_quantile_threshold(values: np.ndarray, q: float, saturation: float = 0.9995) -> float:
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    q = float(np.clip(q, 0.50, 0.99))
    threshold = float(np.quantile(vals, q))
    if threshold < saturation:
        return threshold
    unsaturated = vals[vals < saturation]
    if unsaturated.size == 0:
        return float(saturation)
    return float(np.quantile(unsaturated, min(q, 0.90)))


def patch_mean_map(x_px: np.ndarray, patch_h: int, patch_w: int) -> np.ndarray:
    x_px = np.asarray(x_px, dtype=np.float32)
    block_h = max(1, x_px.shape[0] // patch_h)
    block_w = max(1, x_px.shape[1] // patch_w)
    h_use = patch_h * block_h
    w_use = patch_w * block_w
    x_crop = x_px[:h_use, :w_use]
    return x_crop.reshape(patch_h, block_h, patch_w, block_w).mean(axis=(1, 3)).astype(np.float32)


def _spectrogram_trend_map(sxx_db_local: np.ndarray) -> np.ndarray:
    x_db = np.asarray(sxx_db_local, dtype=np.float32)
    row_sigma = max(1.0, x_db.shape[0] / 32.0)
    col_sigma = max(1.0, x_db.shape[1] / 32.0)
    row_trend = ndimage.gaussian_filter1d(np.mean(x_db, axis=1), sigma=row_sigma, mode="nearest")[:, None]
    col_trend = ndimage.gaussian_filter1d(np.mean(x_db, axis=0), sigma=col_sigma, mode="nearest")[None, :]
    return (row_trend + col_trend - float(np.mean(x_db))).astype(np.float32)


def _signed_residual_to_unit(x: np.ndarray, q: float = 95.0) -> np.ndarray:
    vals = np.abs(np.asarray(x, dtype=np.float32))
    scale = float(np.percentile(vals[np.isfinite(vals)], q)) if np.any(np.isfinite(vals)) else 1.0
    scale = max(scale, 1e-6)
    return np.clip(0.5 + 0.5 * (x / scale), 0.0, 1.0).astype(np.float32)


def build_signal_agnostic_dino_input(sxx_db_local: np.ndarray, db_min: float = -110.0, db_max: float = -40.0):
    x_db = np.asarray(sxx_db_local, dtype=np.float32)
    span = max(float(db_max - db_min), 1e-6)
    fixed_gray = np.clip((x_db - float(db_min)) / span, 0.0, 1.0).astype(np.float32)
    trend = _spectrogram_trend_map(x_db)
    detrended = (x_db - trend).astype(np.float32)
    local_mean = ndimage.uniform_filter(detrended, size=(7, 7), mode="nearest")
    local_resid = detrended - local_mean
    local_scale = np.sqrt(ndimage.uniform_filter(local_resid ** 2, size=(9, 9), mode="nearest") + 1e-6).astype(np.float32)
    local_z = (local_resid / np.maximum(local_scale, 1e-4)).astype(np.float32)
    abs_detrended = _normalize_map01_local(detrended, low_q=2.0, high_q=98.0)
    local_resid_n = _signed_residual_to_unit(local_z, q=95.0)
    combined = (0.70 * local_resid_n + 0.30 * abs_detrended).astype(np.float32)
    if float(np.std(combined)) < 0.02:
        combined = _normalize_map01_local(detrended, low_q=1.0, high_q=99.0)
    gray_u8 = np.clip(np.round(255.0 * combined), 0, 255).astype(np.uint8)
    img_rgb = Image.fromarray(np.stack([gray_u8, gray_u8, gray_u8], axis=-1), mode="RGB")
    return img_rgb, {
        "variant": "signal_agnostic_gray",
        "input_gray01": combined.astype(np.float32),
        "fixed_gray01": fixed_gray.astype(np.float32),
        "trend_db": trend.astype(np.float32),
        "detrended_db": detrended.astype(np.float32),
    }


def dino_seed_patch_map(sxx_db_local: np.ndarray, patch_h: int, patch_w: int) -> np.ndarray:
    x_db = np.asarray(sxx_db_local, dtype=np.float32)
    p_lin = np.power(10.0, x_db / 10.0)
    p_floor = max(float(np.percentile(p_lin, 30.0)), 1e-20)
    rel_db = 10.0 * np.log10(np.maximum(p_lin, 1e-20) / p_floor)
    rel_db = np.clip(rel_db, -10.0, 25.0)
    persistence_px = ndimage.uniform_filter(rel_db, size=(1, 7), mode="nearest")
    local_contrast_px = rel_db - ndimage.uniform_filter(rel_db, size=(5, 5), mode="nearest")
    persistence_n = _normalize_map01_local(persistence_px)
    contrast_n = _normalize_map01_local(local_contrast_px)
    seed_px = (0.65 * persistence_n + 0.35 * contrast_n).astype(np.float32)
    return patch_mean_map(seed_px, patch_h, patch_w)


def _smooth_binary_label_map(label_map: np.ndarray, iters: int = 2, min_component_size: int = 6) -> np.ndarray:
    out = label_map.copy().astype(np.uint8)
    for _ in range(int(iters)):
        avg = ndimage.uniform_filter(out.astype(np.float32), size=3, mode="nearest")
        out = (avg >= 0.5).astype(np.uint8)
    comp, n_comp = ndimage.label(out)
    if n_comp > 0:
        sizes = ndimage.sum(out, comp, index=np.arange(1, n_comp + 1))
        small_ids = np.where(sizes < int(min_component_size))[0] + 1
        if len(small_ids) > 0:
            small_mask = np.isin(comp, small_ids)
            neigh = ndimage.uniform_filter(out.astype(np.float32), size=3, mode="nearest")
            out[small_mask] = (neigh[small_mask] >= 0.5).astype(np.uint8)
    return out


def _feature_affinity_matrix(x_embed: np.ndarray, k: int = 8) -> np.ndarray:
    x_embed = np.asarray(x_embed, dtype=np.float32)
    n = x_embed.shape[0]
    nn = NearestNeighbors(n_neighbors=min(int(k) + 1, n), metric="cosine")
    nn.fit(x_embed)
    dist, idx = nn.kneighbors(x_embed)
    valid = dist[:, 1:]
    sigma = float(np.median(valid[valid > 0])) if np.any(valid > 0) else 1.0
    sigma = max(sigma, 1e-3)
    aff = np.zeros((n, n), dtype=np.float32)
    for row in range(n):
        for col, distance in zip(idx[row, 1:], dist[row, 1:]):
            weight = float(np.exp(-(distance ** 2) / (2.0 * sigma ** 2)))
            aff[row, col] = max(aff[row, col], weight)
            aff[col, row] = max(aff[col, row], weight)
    np.fill_diagonal(aff, 1.0)
    return aff


def _mutual_knn_affinity(aff: np.ndarray, top_k: int = 8, keep_q: float = 0.40) -> np.ndarray:
    aff = np.asarray(aff, dtype=np.float32)
    n = aff.shape[0]
    top_k = int(np.clip(top_k, 1, max(1, n - 1)))
    knn_mask = np.zeros((n, n), dtype=bool)
    for idx in range(n):
        order = np.argsort(aff[idx])[::-1]
        kept = [j for j in order if j != idx][:top_k]
        knn_mask[idx, kept] = True
    mutual = np.logical_and(knn_mask, knn_mask.T)
    vals = aff[mutual]
    vals = vals[vals > 0]
    if vals.size == 0:
        local_aff = aff.copy()
        np.fill_diagonal(local_aff, 1.0)
        return local_aff
    keep_thr = float(np.quantile(vals, float(np.clip(keep_q, 0.0, 0.95))))
    local_aff = np.where(np.logical_and(mutual, aff >= keep_thr), aff, 0.0).astype(np.float32)
    np.fill_diagonal(local_aff, 1.0)
    return local_aff


def _inject_spatial_shortcuts(local_aff: np.ndarray, full_aff: np.ndarray, patch_h: int, patch_w: int, spatial_weight: float = 0.20) -> np.ndarray:
    out = np.asarray(local_aff, dtype=np.float32).copy()
    full_aff = np.asarray(full_aff, dtype=np.float32)
    for row in range(patch_h):
        for col in range(patch_w):
            idx0 = row * patch_w + col
            for rr in range(max(0, row - 1), min(patch_h, row + 2)):
                for cc in range(max(0, col - 1), min(patch_w, col + 2)):
                    if rr == row and cc == col:
                        continue
                    idx1 = rr * patch_w + cc
                    base_weight = float(full_aff[idx0, idx1])
                    if base_weight <= 0.0:
                        continue
                    out[idx0, idx1] = max(out[idx0, idx1], float(spatial_weight * base_weight))
                    out[idx1, idx0] = max(out[idx1, idx0], float(spatial_weight * base_weight))
    np.fill_diagonal(out, 1.0)
    return out


def _row_normalize(mat: np.ndarray) -> np.ndarray:
    row_sum = np.sum(mat, axis=1, keepdims=True)
    return mat / np.maximum(row_sum, 1e-6)


def _local_affinity_score_map(local_aff: np.ndarray, patch_h: int, patch_w: int) -> np.ndarray:
    trans = _row_normalize(local_aff)
    trans2 = trans @ trans
    trans3 = trans2 @ trans
    weighted_degree = np.sum(local_aff, axis=1) - 1.0
    two_hop_return = np.diag(trans2)
    three_hop_return = np.diag(trans3)
    spatial_strength = np.zeros(local_aff.shape[0], dtype=np.float32)
    for row in range(patch_h):
        for col in range(patch_w):
            idx0 = row * patch_w + col
            vals = []
            for rr in range(max(0, row - 1), min(patch_h, row + 2)):
                for cc in range(max(0, col - 1), min(patch_w, col + 2)):
                    if rr == row and cc == col:
                        continue
                    idx1 = rr * patch_w + cc
                    vals.append(float(local_aff[idx0, idx1]))
            spatial_strength[idx0] = float(np.mean(vals)) if vals else 0.0
    score = (
        0.35 * _normalize_vector01(weighted_degree)
        + 0.30 * _normalize_vector01(two_hop_return)
        + 0.20 * _normalize_vector01(three_hop_return)
        + 0.15 * _normalize_vector01(spatial_strength)
    ).astype(np.float32)
    return _normalize_map01_local(score.reshape(patch_h, patch_w))


def dino_grouping_from_spectrogram(
    sxx_db_local: np.ndarray,
    model,
    patch_size: int,
    device: str,
    db_min: float = -110.0,
    db_max: float = -40.0,
    feature_knn: int = 8,
    spatial_weight: float = 0.35,
    score_q: float = 0.60,
    min_component_size: int = 6,
):
    img_rgb, input_debug = build_signal_agnostic_dino_input(sxx_db_local, db_min=db_min, db_max=db_max)
    feat_local, grid_h, grid_w, img_used = _extract_dino_features_from_rgb(img_rgb, model, patch_size, device)
    seed_patch = dino_seed_patch_map(sxx_db_local, grid_h, grid_w)
    x = np.asarray(feat_local, dtype=np.float32)
    d = min(12, x.shape[1], x.shape[0] - 1)
    if d >= 1:
        x = PCA(n_components=d, random_state=42).fit_transform(x)
    x = x - x.mean(axis=0, keepdims=True)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
    full_aff = _feature_affinity_matrix(x, k=feature_knn)
    local_aff = _mutual_knn_affinity(full_aff, top_k=feature_knn, keep_q=0.40)
    local_aff = _inject_spatial_shortcuts(local_aff, full_aff, grid_h, grid_w, spatial_weight=spatial_weight)
    support_map = _local_affinity_score_map(local_aff, grid_h, grid_w)
    seed_norm = _normalize_map01_local(seed_patch)
    score_map = _normalize_map01_local(0.75 * support_map + 0.25 * seed_norm)
    thr = float(np.quantile(score_map, float(np.clip(score_q, 0.50, 0.95))))
    mask = _smooth_binary_label_map((score_map >= thr).astype(np.uint8), iters=1, min_component_size=min_component_size)
    return {
        "mask": mask.astype(np.uint8),
        "score": score_map.astype(np.float32),
        "threshold": float(thr),
        "input_img": img_used,
        "seed_patch": seed_patch.astype(np.float32),
        "shape": (grid_h, grid_w),
        "features": feat_local,
        "input_debug": input_debug,
    }


def residual_background_spectrogram(sxx_db_local: np.ndarray, bg_percentile: float = 35.0):
    x_db = np.asarray(sxx_db_local, dtype=np.float32)
    bg_freq = max(9, int(2 * max(1, x_db.shape[0] // 24) + 1))
    bg_time = max(9, int(2 * max(1, x_db.shape[1] // 24) + 1))
    background = ndimage.percentile_filter(
        x_db,
        percentile=float(bg_percentile),
        size=(bg_freq, bg_time),
        mode="nearest",
    ).astype(np.float32)
    residual_db = np.maximum(x_db - background, 0.0).astype(np.float32)
    residual_n = _normalize_map01_local(residual_db, 5.0, 99.0)
    return residual_db, residual_n, background


def _structure_tensor_components(x_n: np.ndarray, grad_sigma: float, integ_sigma: float):
    grad_f = ndimage.gaussian_filter(x_n, sigma=grad_sigma, order=[1, 0], mode="nearest")
    grad_t = ndimage.gaussian_filter(x_n, sigma=grad_sigma, order=[0, 1], mode="nearest")
    j_ff = ndimage.gaussian_filter(grad_f * grad_f, sigma=integ_sigma, mode="nearest")
    j_ft = ndimage.gaussian_filter(grad_f * grad_t, sigma=integ_sigma, mode="nearest")
    j_tt = ndimage.gaussian_filter(grad_t * grad_t, sigma=integ_sigma, mode="nearest")
    delta = np.sqrt(np.maximum((j_ff - j_tt) ** 2 + 4.0 * (j_ft ** 2), 0.0))
    lam1 = 0.5 * (j_ff + j_tt + delta)
    lam2 = 0.5 * (j_ff + j_tt - delta)
    coherence = (lam1 - lam2) / np.maximum(lam1 + lam2, 1e-6)
    energy = lam1 + lam2
    return coherence.astype(np.float32), energy.astype(np.float32)


def multi_scale_structure_tensor_gate(sxx_db_local: np.ndarray, patch_h: int, patch_w: int, scales: tuple[float, ...] = (0.8, 1.6, 3.2)):
    residual_db, residual_n, background = residual_background_spectrogram(sxx_db_local)
    gate_stack = []
    coherence_stack = []
    energy_stack = []
    for grad_sigma in tuple(float(v) for v in scales):
        coherence, energy = _structure_tensor_components(residual_n, grad_sigma=grad_sigma, integ_sigma=max(1.0, 1.8 * grad_sigma))
        coherence_n = _normalize_map01_local(coherence, 5.0, 99.0)
        energy_n = _normalize_map01_local(energy, 5.0, 99.0)
        gate_stack.append((coherence_n * np.sqrt(np.maximum(energy_n, 0.0))).astype(np.float32))
        coherence_stack.append(coherence_n)
        energy_stack.append(energy_n)
    coherence_px = np.max(np.stack(coherence_stack, axis=0), axis=0)
    energy_px = np.max(np.stack(energy_stack, axis=0), axis=0)
    gate_px = _normalize_map01_local(np.max(np.stack(gate_stack, axis=0), axis=0), 5.0, 99.0)
    return {
        "background_db": background.astype(np.float32),
        "residual_db": residual_db.astype(np.float32),
        "residual_n": residual_n.astype(np.float32),
        "coherence_px": coherence_px.astype(np.float32),
        "energy_px": energy_px.astype(np.float32),
        "gate_px": gate_px.astype(np.float32),
        "coherence_patch": patch_mean_map(coherence_px, patch_h, patch_w),
        "energy_patch": patch_mean_map(energy_px, patch_h, patch_w),
        "gate_patch": patch_mean_map(gate_px, patch_h, patch_w),
        "residual_patch": patch_mean_map(residual_n, patch_h, patch_w),
    }


def _soft_gate_dino_score(dino_score_map: np.ndarray, gate_patch: np.ndarray, floor: float = 0.25):
    dino_n = _normalize_map01_local(dino_score_map, 5.0, 95.0)
    gate_n = _normalize_map01_local(gate_patch, 5.0, 95.0)
    score = dino_n * (float(floor) + (1.0 - float(floor)) * gate_n)
    return _normalize_map01_local(score, 5.0, 95.0).astype(np.float32), dino_n, gate_n


def apply_coherence_gate_to_dino_result(dino_group: dict[str, Any], sxx_db_local: np.ndarray, gate_floor: float = 0.25, min_component_size: int = 3):
    patch_h, patch_w = tuple(dino_group["shape"])
    coherence = multi_scale_structure_tensor_gate(sxx_db_local, patch_h, patch_w)
    raw_score = np.asarray(dino_group["score"], dtype=np.float32)
    gated_score, _, gate_patch = _soft_gate_dino_score(raw_score, coherence["gate_patch"], floor=gate_floor)
    thr = float(np.quantile(gated_score, 0.60))
    mask = _smooth_binary_label_map((gated_score >= thr).astype(np.uint8), iters=1, min_component_size=min_component_size)
    out = dict(dino_group)
    out["score"] = gated_score.astype(np.float32)
    out["mask"] = mask.astype(np.uint8)
    out["threshold"] = thr
    out["coherence_gate_px"] = coherence["gate_px"].astype(np.float32)
    out["coherence_energy_px"] = coherence["energy_px"].astype(np.float32)
    out["coherence_residual_px"] = coherence["residual_n"].astype(np.float32)
    out["coherence_gate_patch"] = gate_patch.astype(np.float32)
    out["coherence_energy_patch"] = coherence["energy_patch"].astype(np.float32)
    out["coherence_residual_patch"] = coherence["residual_patch"].astype(np.float32)
    return out


def burst_companion_gate(
    sxx_db_local: np.ndarray,
    coherence_gate_px: np.ndarray,
    residual_px: np.ndarray,
    patch_h: int,
    patch_w: int,
    residual_gain: float = 0.45,
    temporal_contrast_gain: float = 0.35,
    coherence_gain: float = 0.20,
):
    residual_px = _normalize_map01_local(np.asarray(residual_px, dtype=np.float32), 5.0, 99.0)
    coherence_gate_px = _normalize_map01_local(np.asarray(coherence_gate_px, dtype=np.float32), 5.0, 99.0)
    time_window = max(5, int(2 * (max(1, sxx_db_local.shape[1] // 32)) + 1))
    time_baseline = ndimage.uniform_filter(residual_px, size=(1, time_window), mode="nearest")
    temporal_contrast_px = np.clip(residual_px - time_baseline, 0.0, None).astype(np.float32)
    temporal_contrast_px = _normalize_map01_local(temporal_contrast_px, 5.0, 99.0)
    companion_px = (
        float(residual_gain) * residual_px
        + float(temporal_contrast_gain) * temporal_contrast_px
        + float(coherence_gain) * coherence_gate_px
    ).astype(np.float32)
    companion_px = _normalize_map01_local(companion_px, 5.0, 99.0)
    return {
        "companion_px": companion_px.astype(np.float32),
        "residual_px": residual_px.astype(np.float32),
        "temporal_contrast_px": temporal_contrast_px.astype(np.float32),
        "companion_patch": patch_mean_map(companion_px, patch_h, patch_w),
    }


def power_prior_patch_map(sxx_db_local: np.ndarray, patch_h: int, patch_w: int) -> np.ndarray:
    p_lin = np.power(10.0, np.asarray(sxx_db_local, dtype=np.float32) / 10.0)
    p_floor = max(float(np.percentile(p_lin, 30.0)), 1e-20)
    rel_db = 10.0 * np.log10(np.maximum(p_lin, 1e-20) / p_floor)
    rel_db = np.clip(rel_db, -5.0, 25.0)
    return patch_mean_map(rel_db, patch_h, patch_w)


def _resize_patch_map_to_pixels(map_patch: np.ndarray, rows: int, cols: int, resample: int) -> np.ndarray:
    img = Image.fromarray(np.asarray(map_patch, dtype=np.float32), mode="F")
    out = img.resize((int(cols), int(rows)), resample=resample)
    return np.asarray(out, dtype=np.float32)


def _resize_patch_mask_to_pixels(mask_patch: np.ndarray, rows: int, cols: int) -> np.ndarray:
    resized = _resize_patch_map_to_pixels(mask_patch.astype(np.float32), rows, cols, Image.NEAREST)
    return resized >= 0.5


def run_chunk_detector(
    sxx_db_chunk: np.ndarray,
    model,
    patch_size: int,
    device: str,
    cfg: WidebandChunkConfig,
) -> dict[str, Any]:
    t0 = time.perf_counter()
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
    t1 = time.perf_counter()
    dino_gated = apply_coherence_gate_to_dino_result(
        dino_group,
        sxx_db_chunk,
        gate_floor=cfg.dino_coherence_gate_floor,
        min_component_size=max(3, cfg.min_component_size // 2),
    )
    t2 = time.perf_counter()
    patch_h, patch_w = tuple(dino_gated["shape"])
    power_patch = _normalize_map01_local(power_prior_patch_map(sxx_db_chunk, patch_h, patch_w), 5.0, 95.0)
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
    fused_patch = _normalize_map01_local((1.0 - cfg.power_fusion_gain) * dino_gated["score"] + cfg.power_fusion_gain * power_patch, 5.0, 95.0)
    strict_patch = _normalize_map01_local(
        fused_patch * (float(cfg.merge_companion_floor) + (1.0 - float(cfg.merge_companion_floor)) * burst_gate["companion_patch"]),
        5.0,
        95.0,
    )
    support_patch = _normalize_map01_local(0.70 * dino_gated["score"] + 0.30 * burst_gate["companion_patch"], 5.0, 95.0)
    support_thr = float(np.quantile(support_patch, float(np.clip(cfg.merge_support_q, 0.50, 0.95))))
    support_mask = _smooth_binary_label_map((support_patch >= support_thr).astype(np.uint8), iters=1, min_component_size=max(3, cfg.min_component_size // 2))
    fused_thr = float(np.quantile(strict_patch, cfg.final_score_q))
    fused_mask = _smooth_binary_label_map((strict_patch >= fused_thr).astype(np.uint8), iters=1, min_component_size=cfg.min_component_size)
    score_px = _resize_patch_map_to_pixels(strict_patch, sxx_db_chunk.shape[0], sxx_db_chunk.shape[1], Image.BILINEAR)
    mask_px = _resize_patch_mask_to_pixels(fused_mask, sxx_db_chunk.shape[0], sxx_db_chunk.shape[1])
    support_px = _resize_patch_mask_to_pixels(support_mask, sxx_db_chunk.shape[0], sxx_db_chunk.shape[1])
    companion_px = _resize_patch_map_to_pixels(burst_gate["companion_px"], sxx_db_chunk.shape[0], sxx_db_chunk.shape[1], Image.BILINEAR)
    t3 = time.perf_counter()
    return {
        "dino_group": dino_group,
        "dino_gated": dino_gated,
        "burst_gate": burst_gate,
        "power_patch": power_patch.astype(np.float32),
        "fused_patch": fused_patch.astype(np.float32),
        "strict_patch": strict_patch.astype(np.float32),
        "support_patch": support_patch.astype(np.float32),
        "support_threshold": support_thr,
        "support_mask_patch": support_mask.astype(np.uint8),
        "fused_mask_patch": fused_mask.astype(np.uint8),
        "fused_threshold": fused_thr,
        "score_px": score_px.astype(np.float32),
        "mask_px": mask_px.astype(bool),
        "support_px": support_px.astype(bool),
        "companion_px": companion_px.astype(np.float32),
        "timing_ms": {
            "dino_group_ms": (t1 - t0) * 1000.0,
            "dino_coherence_ms": (t2 - t1) * 1000.0,
            "fuse_ms": (t3 - t2) * 1000.0,
            "total_ms": (t3 - t0) * 1000.0,
        },
    }


def _chunk_blend_weights(length: int) -> np.ndarray:
    if length <= 2:
        return np.ones(length, dtype=np.float32)
    base = np.hanning(length).astype(np.float32)
    if float(np.max(base)) <= 0.0:
        return np.ones(length, dtype=np.float32)
    base = base / float(np.max(base))
    return (0.2 + 0.8 * base).astype(np.float32)


def merge_chunk_results(global_shape: tuple[int, int], chunk_results: list[dict[str, Any]], final_score_q: float = 0.90, min_component_size: int = 6):
    merged_base_score = np.zeros(global_shape, dtype=np.float32)
    merged_companion = np.zeros(global_shape, dtype=np.float32)
    merged_support = np.zeros(global_shape, dtype=bool)
    for chunk in chunk_results:
        row_start = int(chunk["row_start"])
        row_stop = int(chunk["row_stop"])
        chunk_weights = _chunk_blend_weights(row_stop - row_start)[:, None]
        weighted_score = chunk["score_px"] * chunk_weights
        weighted_companion = chunk["companion_px"] * chunk_weights
        merged_base_score[row_start:row_stop, :] = np.maximum(merged_base_score[row_start:row_stop, :], weighted_score)
        merged_companion[row_start:row_stop, :] = np.maximum(merged_companion[row_start:row_stop, :], weighted_companion)
        merged_support[row_start:row_stop, :] |= np.asarray(chunk["support_px"], dtype=bool)
    merged_score = _normalize_map01_local(
        merged_base_score * (0.30 + 0.70 * _normalize_map01_local(merged_companion, 5.0, 95.0)),
        5.0,
        95.0,
    )
    valid_row_mask = np.ones(global_shape[0], dtype=bool)
    if chunk_results:
        candidate_mask = chunk_results[0].get("valid_row_mask")
        if candidate_mask is not None:
            valid_row_mask = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    valid_scores = merged_score[np.logical_and(valid_row_mask[:, None], merged_support)]
    threshold = _robust_high_quantile_threshold(valid_scores, final_score_q) if valid_scores.size else 1.0
    merged_mask = _smooth_binary_label_map(
        np.logical_and(merged_score >= threshold, merged_support).astype(np.uint8),
        iters=1,
        min_component_size=min_component_size,
    )
    merged_mask[~valid_row_mask, :] = 0
    merged_score[~valid_row_mask, :] = 0.0
    merged_support[~valid_row_mask, :] = False
    return {
        "merged_score": merged_score.astype(np.float32),
        "merged_mask": merged_mask.astype(bool),
        "merged_threshold": threshold,
        "valid_row_mask": valid_row_mask.astype(bool),
        "merged_support": merged_support.astype(bool),
        "merged_base_score": merged_base_score.astype(np.float32),
        "merged_companion": merged_companion.astype(np.float32),
    }


def run_chunked_pipeline(
    input_record: dict[str, Any],
    model,
    patch_size: int,
    device: str,
    cfg: WidebandChunkConfig,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    ignore_info = compute_ignore_sideband_rows(
        input_record["freq_axis_hz"],
        ignore_sideband_hz=cfg.ignore_sideband_hz,
        min_keep_rows=max(int(patch_size), 16),
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
        detection = run_chunk_detector(corrected_sxx_db[row_slice, :], model, patch_size, device, cfg)
        chunk_results.append({**chunk, **detection, "valid_row_mask": valid_row_mask})
    merged = merge_chunk_results(
        corrected_sxx_db.shape,
        chunk_results,
        final_score_q=cfg.final_score_q,
        min_component_size=cfg.min_component_size,
    )
    t1 = time.perf_counter()
    return {
        "input_record": input_record,
        "config": cfg,
        "frontend": correction,
        "corrected_sxx_db": corrected_sxx_db,
        "ignore_sideband": ignore_info,
        "chunk_plan": chunk_plan,
        "chunk_results": chunk_results,
        **merged,
        "total_runtime_ms": (t1 - t0) * 1000.0,
    }


def _display_db_window(sxx_db: np.ndarray, low_q: float = 1.0, high_q: float = 99.0):
    vals = np.asarray(sxx_db, dtype=np.float32)
    return float(np.percentile(vals, low_q)), float(np.percentile(vals, high_q))


def plot_frontend_overview(pipeline_result: dict[str, Any], figsize: tuple[int, int] = (18, 12)):
    record = pipeline_result["input_record"]
    corrected_sxx_db = pipeline_result["corrected_sxx_db"]
    frontend = pipeline_result["frontend"]
    ignore_info = pipeline_result.get("ignore_sideband", {})
    valid_row_mask = np.asarray(ignore_info.get("valid_row_mask", np.ones(corrected_sxx_db.shape[0], dtype=bool)), dtype=bool)
    freq_axis_hz = np.asarray(record["freq_axis_hz"], dtype=np.float32)
    raw_vmin, raw_vmax = _display_db_window(record["sxx_db"])
    corrected_vmin, corrected_vmax = _display_db_window(corrected_sxx_db)
    fig, axes = plt.subplots(3, 1, figsize=figsize, constrained_layout=True)
    axes[0].imshow(record["sxx_db"], aspect="auto", origin="lower", cmap="viridis", vmin=raw_vmin, vmax=raw_vmax)
    axes[0].set_title("Wideband input spectrogram")
    axes[0].set_ylabel("Frequency row")
    axes[1].imshow(corrected_sxx_db, aspect="auto", origin="lower", cmap="viridis", vmin=corrected_vmin, vmax=corrected_vmax)
    axes[1].set_title("Globally corrected spectrogram")
    axes[1].set_ylabel("Frequency row")
    axes[2].plot(frontend["row_floor_db"], freq_axis_hz, label="Row floor")
    axes[2].plot(frontend["response_db"], freq_axis_hz, label="Smoothed response")
    axes[2].axvline(frontend["reference_db"], color="tab:green", linestyle="--", label="Reference")
    for chunk in pipeline_result["chunk_plan"]:
        axes[1].axhline(chunk["row_start"], color="white", alpha=0.15, linewidth=0.8)
        axes[1].axhline(chunk["row_stop"] - 1, color="white", alpha=0.15, linewidth=0.8)
    ignored_rows = np.flatnonzero(~valid_row_mask)
    if ignored_rows.size > 0:
        low_block = ignored_rows[ignored_rows < (corrected_sxx_db.shape[0] // 2)]
        high_block = ignored_rows[ignored_rows >= (corrected_sxx_db.shape[0] // 2)]
        if low_block.size > 0:
            for ax in axes[:2]:
                ax.axhspan(low_block[0], low_block[-1], color="black", alpha=0.18)
        if high_block.size > 0:
            for ax in axes[:2]:
                ax.axhspan(high_block[0], high_block[-1], color="black", alpha=0.18)
    axes[2].set_title("Global frontend correction profile")
    axes[2].set_xlabel("Level (dB)")
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].legend(loc="best")
    return fig, axes


def plot_chunk_examples(pipeline_result: dict[str, Any], max_chunks: int = 4, figsize: tuple[int, int] = (22, 5)):
    chunk_results = pipeline_result["chunk_results"][: max(1, int(max_chunks))]
    fig, axes = plt.subplots(len(chunk_results), 4, figsize=(figsize[0], figsize[1] * len(chunk_results)), constrained_layout=True)
    if len(chunk_results) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_idx, chunk in enumerate(chunk_results):
        tile = pipeline_result["corrected_sxx_db"][chunk["row_start"]:chunk["row_stop"], :]
        dino = chunk["dino_gated"]
        axes[row_idx][0].imshow(tile, aspect="auto", origin="lower", cmap="viridis")
        axes[row_idx][0].set_title(f"Chunk {chunk['chunk_index']} corrected tile")
        axes[row_idx][1].imshow(dino["input_img"])
        axes[row_idx][1].set_title("DINO input")
        axes[row_idx][2].imshow(chunk["strict_patch"], cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[row_idx][2].set_title("Chunk strict patch score")
        axes[row_idx][3].imshow(chunk["mask_px"], cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_idx][3].set_title("Chunk pixel mask")
        for ax in axes[row_idx]:
            ax.set_xticks([])
            ax.set_yticks([])
    return fig, axes


def plot_merged_detection(pipeline_result: dict[str, Any], figsize: tuple[int, int] = (20, 6)):
    corrected_sxx_db = pipeline_result["corrected_sxx_db"]
    merged_score = pipeline_result["merged_score"]
    merged_mask = pipeline_result["merged_mask"]
    merged_support = np.asarray(pipeline_result.get("merged_support", np.ones_like(merged_mask, dtype=bool)), dtype=bool)
    valid_row_mask = np.asarray(pipeline_result.get("valid_row_mask", np.ones(corrected_sxx_db.shape[0], dtype=bool)), dtype=bool)
    vmin, vmax = _display_db_window(corrected_sxx_db)
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    axes[0].imshow(corrected_sxx_db, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Corrected wideband spectrogram")
    axes[1].imshow(merged_score, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].imshow(np.where(merged_support, 1.0, np.nan), aspect="auto", origin="lower", cmap="winter", alpha=0.18)
    axes[1].set_title("Merged strict score + support")
    axes[2].imshow(corrected_sxx_db, aspect="auto", origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    axes[2].imshow(np.where(merged_mask, 1.0, np.nan), aspect="auto", origin="lower", cmap="autumn", alpha=0.55)
    axes[2].set_title("Merged detection overlay")
    ignored_rows = np.flatnonzero(~valid_row_mask)
    if ignored_rows.size > 0:
        low_block = ignored_rows[ignored_rows < (corrected_sxx_db.shape[0] // 2)]
        high_block = ignored_rows[ignored_rows >= (corrected_sxx_db.shape[0] // 2)]
        if low_block.size > 0:
            for ax in axes:
                ax.axhspan(low_block[0], low_block[-1], color="black", alpha=0.12)
        if high_block.size > 0:
            for ax in axes:
                ax.axhspan(high_block[0], high_block[-1], color="black", alpha=0.12)
    for ax in axes:
        ax.set_xlabel("Time bin")
        ax.set_ylabel("Frequency row")
    return fig, axes


def plot_merged_debug(pipeline_result: dict[str, Any], figsize: tuple[int, int] = (20, 12)):
    corrected_sxx_db = pipeline_result["corrected_sxx_db"]
    merged_base_score = np.asarray(pipeline_result.get("merged_base_score", np.zeros_like(corrected_sxx_db)), dtype=np.float32)
    merged_companion = np.asarray(pipeline_result.get("merged_companion", np.zeros_like(corrected_sxx_db)), dtype=np.float32)
    merged_support = np.asarray(pipeline_result.get("merged_support", np.zeros_like(corrected_sxx_db, dtype=bool)), dtype=bool)
    merged_score = np.asarray(pipeline_result["merged_score"], dtype=np.float32)
    merged_mask = np.asarray(pipeline_result["merged_mask"], dtype=bool)
    merged_threshold = float(pipeline_result.get("merged_threshold", 0.0))
    valid_row_mask = np.asarray(pipeline_result.get("valid_row_mask", np.ones(corrected_sxx_db.shape[0], dtype=bool)), dtype=bool)

    vmin, vmax = _display_db_window(corrected_sxx_db)
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    axes[0][0].imshow(merged_base_score, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
    axes[0][0].set_title("Merged base score")

    axes[0][1].imshow(merged_companion, aspect="auto", origin="lower", cmap="cividis", vmin=0.0, vmax=1.0)
    axes[0][1].set_title("Merged companion gate")

    axes[1][0].imshow(corrected_sxx_db, aspect="auto", origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    axes[1][0].imshow(np.where(merged_support, 1.0, np.nan), aspect="auto", origin="lower", cmap="winter", alpha=0.50)
    axes[1][0].set_title("Merged support region")

    axes[1][1].imshow(corrected_sxx_db, aspect="auto", origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    axes[1][1].imshow(np.where(merged_score >= merged_threshold, 1.0, np.nan), aspect="auto", origin="lower", cmap="plasma", alpha=0.22)
    axes[1][1].imshow(np.where(merged_mask, 1.0, np.nan), aspect="auto", origin="lower", cmap="autumn", alpha=0.55)
    axes[1][1].set_title(f"Final overlay | threshold={merged_threshold:.3f}")

    ignored_rows = np.flatnonzero(~valid_row_mask)
    if ignored_rows.size > 0:
        low_block = ignored_rows[ignored_rows < (corrected_sxx_db.shape[0] // 2)]
        high_block = ignored_rows[ignored_rows >= (corrected_sxx_db.shape[0] // 2)]
        for row in axes:
            for ax in row:
                if low_block.size > 0:
                    ax.axhspan(low_block[0], low_block[-1], color="black", alpha=0.12)
                if high_block.size > 0:
                    ax.axhspan(high_block[0], high_block[-1], color="black", alpha=0.12)

    for row in axes:
        for ax in row:
            ax.set_xlabel("Time bin")
            ax.set_ylabel("Frequency row")

    return fig, axes