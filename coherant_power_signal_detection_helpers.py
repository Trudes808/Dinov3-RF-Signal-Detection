from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from scipy import ndimage, signal


@dataclass
class CoherentPowerConfig:
    chunk_bandwidth_hz: float = 25e6
    chunk_overlap_hz: float = 6.25e6
    uncalibrated_chunk_fraction: float = 0.40
    uncalibrated_overlap_fraction: float = 0.20
    ignore_sideband_percent: float = 0.0
    ignore_sideband_hz: float | None = 7.0e6
    frontend_row_q: float = 25.0
    frontend_reference_q: float = 75.0
    frontend_smooth_sigma: float = 12.0
    frontend_max_boost_db: float = 12.0
    coherence_weight: float = 0.55
    power_weight: float = 0.45
    coherence_power_support_q: float = 0.82
    coherence_power_q: float = 0.92
    min_component_size: int = 6
    grouping_seed_score_q: float = 0.72
    grouping_bridge_freq_px: int = 33
    grouping_bridge_time_px: int = 5
    grouping_min_component_size: int = 24
    grouping_min_freq_span_px: int = 18
    grouping_min_time_span_px: int = 2
    grouping_min_density: float = 0.06
    grouping_time_continuity_ratio: float = 0.85


def infer_input_kind(input_path: str | Path, explicit_kind: str = "auto") -> str:
    if explicit_kind in {"pgm", "sigmf", "tensor_npy", "npy"}:
        return "tensor_npy" if explicit_kind == "npy" else explicit_kind
    suffix = Path(input_path).suffix.lower()
    if suffix == ".pgm":
        return "pgm"
    if suffix == ".npy":
        return "tensor_npy"
    if suffix == ".sigmf-meta":
        return "sigmf"
    raise ValueError(f"Unsupported input type for {input_path}")


def input_kind_requires_display_transpose(input_kind: str | None) -> bool:
    return input_kind in {"pgm", "tensor_npy"}


def has_calibrated_frequency_axis(input_record: dict[str, Any]) -> bool:
    calibrated = input_record.get("frequency_axis_calibrated")
    if calibrated is not None:
        return bool(calibrated)
    sample_rate_hz = input_record.get("sample_rate_hz")
    return sample_rate_hz is not None and float(sample_rate_hz) > 0.0


def read_pgm_raw(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as file:
        magic = file.readline().strip()
        if magic != b"P5":
            raise ValueError(f"{path.name}: unsupported PGM magic {magic!r}")

        header_tokens: list[bytes] = []
        while len(header_tokens) < 3:
            line = file.readline()
            if not line:
                raise ValueError(f"{path.name}: truncated PGM header")
            line = line.strip()
            if not line or line.startswith(b"#"):
                continue
            header_tokens.extend(line.split())

        cols, rows, maxval = map(int, header_tokens[:3])
        if maxval > 255:
            raise ValueError(f"{path.name}: only 8-bit PGM supported (maxval={maxval})")

        data = file.read(rows * cols)
        if len(data) != rows * cols:
            raise ValueError(f"{path.name}: unexpected payload length {len(data)}")

    image = np.frombuffer(data, dtype=np.uint8).reshape(rows, cols).astype(np.float32)
    return np.ascontiguousarray(image)


def read_complex_tensor_npy(path: str | Path) -> np.ndarray:
    path = Path(path)
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(f"{path.name}: expected a 2D tensor snapshot, got shape {array.shape}")
    if not np.iscomplexobj(array):
        raise ValueError(f"{path.name}: expected complex64 tensor data")
    return np.ascontiguousarray(array.astype(np.complex64, copy=False))


def resize_float_image(image: np.ndarray, width: int, height: int, resample: int = Image.BILINEAR) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D float image, got shape {image.shape}")
    width = max(1, int(width))
    height = max(1, int(height))
    pil_image = Image.fromarray(image, mode="F")
    resized = pil_image.resize((width, height), resample=resample)
    return np.asarray(resized, dtype=np.float32)


def read_sigmf_meta(meta_path: str | Path):
    meta_path = Path(meta_path)
    with meta_path.open("r") as file:
        meta = json.load(file)
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


def _load_sigmf_iq(
    data_path: str | Path,
    dtype,
    is_complex: bool,
    start_sample: int,
    count: int | None,
    num_channels: int = 1,
    channel: int = 0,
):
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


def load_sigmf_samples(
    meta_path: str | Path,
    start_s: float = 0.0,
    duration_s: float | None = 1.0,
    capture_index: int = 0,
    channel: int = 0,
):
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


def generate_spectrogram(
    iq_data: np.ndarray,
    sample_rate_hz: float,
    fft_size: int = 1024,
    noverlap: int = 512,
    center_frequency_hz: float | None = None,
):
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
    tensor_target_height: int | None = None,
    tensor_target_width: int | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    resolved_kind = infer_input_kind(input_path, input_kind)
    if resolved_kind == "pgm":
        pgm_img = read_pgm_raw(input_path)
        sxx_db = np.ascontiguousarray(pgm_img.T)
        time_axis_s = np.arange(pgm_img.shape[0], dtype=np.float32)
        center_frequency_hz = None
        sample_rate_hz = None
        freq_axis_hz = np.arange(sxx_db.shape[0], dtype=np.float32)
        return {
            "input_kind": "pgm",
            "input_path": str(input_path),
            "sxx_db": sxx_db.astype(np.float32),
            "display_sxx_db": pgm_img.astype(np.float32),
            "display_transposed": True,
            "frequency_axis_calibrated": False,
            "freq_axis_hz": freq_axis_hz.astype(np.float32),
            "time_axis_s": time_axis_s,
            "center_frequency_hz": center_frequency_hz,
            "sample_rate_hz": sample_rate_hz,
            "annotations": [],
        }

    if resolved_kind == "tensor_npy":
        complex_tensor = read_complex_tensor_npy(input_path)
        power_db = (10.0 * np.log10(np.maximum(np.abs(complex_tensor) ** 2, 1e-12))).astype(np.float32)
        display_power_db = power_db
        if tensor_target_height is not None and tensor_target_width is not None:
            display_power_db = resize_float_image(
                display_power_db,
                width=int(tensor_target_width),
                height=int(tensor_target_height),
                resample=Image.BILINEAR,
            )
        sxx_db = np.ascontiguousarray(display_power_db.T)
        time_axis_s = np.arange(display_power_db.shape[0], dtype=np.float32)
        center_frequency_hz = None
        sample_rate_hz = None
        freq_axis_hz = np.arange(sxx_db.shape[0], dtype=np.float32)
        return {
            "input_kind": "tensor_npy",
            "input_path": str(input_path),
            "sxx_db": sxx_db.astype(np.float32),
            "display_sxx_db": display_power_db.astype(np.float32),
            "display_transposed": True,
            "frequency_axis_calibrated": False,
            "freq_axis_hz": freq_axis_hz.astype(np.float32),
            "time_axis_s": time_axis_s,
            "center_frequency_hz": center_frequency_hz,
            "sample_rate_hz": sample_rate_hz,
            "raw_tensor_shape": tuple(int(v) for v in complex_tensor.shape),
            "resized_tensor_shape": tuple(int(v) for v in display_power_db.shape),
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
        "display_transposed": False,
        "frequency_axis_calibrated": True,
        "freq_axis_hz": freq_axis_hz,
        "time_axis_s": time_axis_s,
        "center_frequency_hz": meta["center_frequency_hz"],
        "sample_rate_hz": meta["sample_rate_hz"],
        "annotations": meta["annotations"],
    }


def adapt_chunk_config_for_input_record(
    input_record: dict[str, Any],
    cfg: CoherentPowerConfig,
    target_chunk_rows: int = 1024,
    target_overlap_rows: int = 256,
) -> CoherentPowerConfig:
    freq_axis_hz = np.asarray(input_record.get("freq_axis_hz", []), dtype=np.float32).reshape(-1)
    if freq_axis_hz.size < 2:
        return cfg

    bin_hz = float(np.median(np.abs(np.diff(freq_axis_hz))))
    if not np.isfinite(bin_hz) or bin_hz <= 0.0:
        return cfg

    if input_record.get("input_kind") != "tensor_npy":
        return cfg

    calibrated_axis = has_calibrated_frequency_axis(input_record)
    num_rows = int(freq_axis_hz.size)
    target_chunk_rows = int(min(num_rows, max(32, target_chunk_rows)))
    target_overlap_rows = int(min(target_chunk_rows - 1, max(0, target_overlap_rows)))

    return replace(
        cfg,
        chunk_bandwidth_hz=float(target_chunk_rows * bin_hz),
        chunk_overlap_hz=float(target_overlap_rows * bin_hz),
        ignore_sideband_percent=0.0,
        ignore_sideband_hz=(cfg.ignore_sideband_hz if cfg.ignore_sideband_hz is not None else 7.0e6) if calibrated_axis else None,
    )


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
    response_db = ndimage.gaussian_filter1d(
        row_floor_db,
        sigma=max(float(smooth_sigma), 1.0),
        mode="nearest",
    ).astype(np.float32)
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


def compute_ignore_sideband_rows(
    freq_axis_hz: np.ndarray,
    ignore_sideband_percent: float = 0.10,
    min_keep_rows: int = 16,
    ignore_sideband_hz: float | None = None,
) -> dict[str, float | int | np.ndarray]:
    freq_axis_hz = np.asarray(freq_axis_hz, dtype=np.float32).reshape(-1)
    num_rows = int(freq_axis_hz.size)
    clipped_percent = float(np.clip(ignore_sideband_percent, 0.0, 0.49))
    info: dict[str, float | int | np.ndarray] = {
        "requested_percent": clipped_percent,
        "applied_percent": 0.0,
        "requested_hz": float(max(0.0, ignore_sideband_hz or 0.0)),
        "requested_bins": 0,
        "applied_hz": 0.0,
        "applied_bins": 0,
        "bin_hz": 0.0,
        "valid_row_mask": np.ones(num_rows, dtype=bool),
    }
    if num_rows < 2:
        return info

    bin_hz = float(np.median(np.abs(np.diff(freq_axis_hz))))
    if not np.isfinite(bin_hz) or bin_hz <= 0.0:
        return info

    max_bins = max(0, (num_rows - int(max(1, min_keep_rows))) // 2)
    if clipped_percent > 0.0:
        requested_bins = int(np.ceil(num_rows * clipped_percent))
        requested_hz = float(requested_bins * bin_hz)
    else:
        requested_hz = float(max(0.0, ignore_sideband_hz or 0.0))
        requested_bins = int(np.ceil(requested_hz / bin_hz)) if requested_hz > 0.0 else 0
    applied_bins = int(np.clip(requested_bins, 0, max_bins))
    valid_row_mask = np.ones(num_rows, dtype=bool)
    if applied_bins > 0:
        valid_row_mask[:applied_bins] = False
        valid_row_mask[-applied_bins:] = False

    info.update(
        {
            "requested_percent": clipped_percent,
            "applied_percent": float(applied_bins / max(num_rows, 1)),
            "requested_hz": requested_hz,
            "requested_bins": int(requested_bins),
            "applied_hz": float(applied_bins * bin_hz),
            "applied_bins": int(applied_bins),
            "bin_hz": bin_hz,
            "valid_row_mask": valid_row_mask,
        }
    )
    return info


def build_frequency_chunks(
    freq_axis_hz: np.ndarray,
    chunk_bandwidth_hz: float,
    chunk_overlap_hz: float,
    min_rows: int = 16,
    valid_row_mask: np.ndarray | None = None,
    uncalibrated_chunk_fraction: float = 0.40,
    uncalibrated_overlap_fraction: float = 0.20,
) -> list[dict[str, Any]]:
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

    freq_span = float(freq_max - freq_min)
    if freq_span <= 0.0 or chunk_bandwidth_hz >= freq_span:
        valid_count = int(valid_idx.size)
        chunk_fraction = float(np.clip(uncalibrated_chunk_fraction, 0.10, 1.0))
        overlap_fraction = float(np.clip(uncalibrated_overlap_fraction, 0.0, 0.95))
        chunk_rows = int(np.clip(round(valid_count * chunk_fraction), min_rows, valid_count))
        if chunk_rows >= valid_count:
            return [{
                "chunk_index": 0,
                "row_start": int(valid_idx[0]),
                "row_stop": int(valid_idx[-1]) + 1,
                "freq_start_hz": float(freq_axis_hz[valid_idx[0]]),
                "freq_stop_hz": float(freq_axis_hz[valid_idx[-1]]),
            }]
        overlap_rows = int(np.clip(round(chunk_rows * overlap_fraction), 0, chunk_rows - 1))
        step_rows = max(1, chunk_rows - overlap_rows)
        chunks: list[dict[str, Any]] = []
        chunk_index = 0
        start_pos = 0
        while start_pos < valid_count:
            stop_pos = min(start_pos + chunk_rows, valid_count)
            in_chunk = valid_idx[start_pos:stop_pos]
            if in_chunk.size >= int(min_rows):
                chunks.append({
                    "chunk_index": chunk_index,
                    "row_start": int(in_chunk[0]),
                    "row_stop": int(in_chunk[-1]) + 1,
                    "freq_start_hz": float(freq_axis_hz[in_chunk[0]]),
                    "freq_stop_hz": float(freq_axis_hz[in_chunk[-1]]),
                })
                chunk_index += 1
            if stop_pos >= valid_count:
                break
            start_pos += step_rows
        return chunks

    chunks: list[dict[str, Any]] = []
    chunk_start_hz = freq_min
    chunk_index = 0
    while chunk_start_hz < freq_max + 1e-6:
        chunk_stop_hz = min(chunk_start_hz + chunk_bandwidth_hz, freq_max)
        in_chunk = valid_idx[(valid_freq_axis_hz >= chunk_start_hz) & (valid_freq_axis_hz <= chunk_stop_hz)]
        if in_chunk.size >= int(min_rows):
            chunks.append({
                "chunk_index": chunk_index,
                "row_start": int(in_chunk[0]),
                "row_stop": int(in_chunk[-1]) + 1,
                "freq_start_hz": float(freq_axis_hz[in_chunk[0]]),
                "freq_stop_hz": float(freq_axis_hz[in_chunk[-1]]),
            })
            chunk_index += 1
        if chunk_stop_hz >= freq_max:
            break
        chunk_start_hz += step_hz
    return chunks


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


def _normalize_map01_masked(
    x: np.ndarray,
    mask: np.ndarray,
    low_q: float = 5.0,
    high_q: float = 95.0,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if x.shape != mask.shape:
        raise ValueError("x and mask must share the same shape")
    vals = x[np.logical_and(mask, np.isfinite(x))]
    out = np.zeros_like(x, dtype=np.float32)
    if vals.size == 0:
        return out
    lo = float(np.percentile(vals, low_q))
    hi = float(np.percentile(vals, high_q))
    if hi <= lo:
        hi = lo + 1e-6
    out[mask] = np.clip((x[mask] - lo) / (hi - lo), 0.0, 1.0)
    return out.astype(np.float32)


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


def _local_relative_power_support_map(
    sxx_db_local: np.ndarray,
    valid_row_mask: np.ndarray | None = None,
    floor_q: float = 30.0,
    freq_window: int = 9,
    time_window: int = 33,
) -> np.ndarray:
    x_db = np.asarray(sxx_db_local, dtype=np.float32)
    p_lin = np.power(10.0, x_db / 10.0)
    if valid_row_mask is None:
        valid_values = p_lin.reshape(-1)
    else:
        valid_row_mask = np.asarray(valid_row_mask, dtype=bool).reshape(-1)
        if valid_row_mask.shape[0] != p_lin.shape[0]:
            raise ValueError("valid_row_mask length must match the number of spectrogram rows")
        valid_values = p_lin[valid_row_mask, :].reshape(-1)
        if valid_values.size == 0:
            valid_values = p_lin.reshape(-1)
    p_floor = max(float(np.percentile(valid_values, floor_q)), 1e-20)
    rel_db = 10.0 * np.log10(np.maximum(p_lin, 1e-20) / p_floor)
    rel_db = np.clip(rel_db, -5.0, 25.0).astype(np.float32)
    freq_window = max(3, int(freq_window) | 1)
    time_window = max(5, int(time_window) | 1)
    local_baseline = ndimage.uniform_filter(rel_db, size=(freq_window, time_window), mode="nearest")
    local_support = np.clip(rel_db - local_baseline, 0.0, None).astype(np.float32)
    if valid_row_mask is not None:
        local_support = local_support.copy()
        local_support[~valid_row_mask, :] = 0.0
    return local_support


def residual_background_spectrogram(sxx_db_local: np.ndarray):
    x_db = np.asarray(sxx_db_local, dtype=np.float32)
    bg_freq = max(9, int(2 * max(1, x_db.shape[0] // 24) + 1))
    bg_time = max(9, int(2 * max(1, x_db.shape[1] // 24) + 1))
    background = ndimage.uniform_filter(x_db, size=(bg_freq, bg_time), mode="nearest").astype(np.float32)
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


def multi_scale_structure_tensor_gate(
    sxx_db_local: np.ndarray,
    scales: tuple[float, ...] = (0.8, 1.6, 3.2),
    max_height_px: int | None = None,
    max_width_px: int | None = None,
):
    sxx_db_local = np.asarray(sxx_db_local, dtype=np.float32)
    work_rows = int(sxx_db_local.shape[0])
    work_cols = int(sxx_db_local.shape[1])
    if max_height_px is not None:
        work_rows = min(work_rows, int(max_height_px))
    if max_width_px is not None:
        work_cols = min(work_cols, int(max_width_px))

    work_sxx_db = sxx_db_local
    if work_rows < int(sxx_db_local.shape[0]) or work_cols < int(sxx_db_local.shape[1]):
        work_sxx_db = resize_float_image(sxx_db_local, width=work_cols, height=work_rows, resample=Image.BILINEAR)

    residual_db, residual_n, background = residual_background_spectrogram(work_sxx_db)
    gate_stack = []
    coherence_stack = []
    energy_stack = []
    for grad_sigma in tuple(float(value) for value in scales):
        coherence, energy = _structure_tensor_components(
            residual_n,
            grad_sigma=grad_sigma,
            integ_sigma=max(1.0, 1.8 * grad_sigma),
        )
        coherence_n = _normalize_map01_local(coherence, 5.0, 99.0)
        energy_n = _normalize_map01_local(energy, 5.0, 99.0)
        gate_stack.append((coherence_n * np.sqrt(np.maximum(energy_n, 0.0))).astype(np.float32))
        coherence_stack.append(coherence_n)
        energy_stack.append(energy_n)

    coherence_px = np.max(np.stack(coherence_stack, axis=0), axis=0).astype(np.float32)
    energy_px = np.max(np.stack(energy_stack, axis=0), axis=0).astype(np.float32)
    gate_px = _normalize_map01_local(np.max(np.stack(gate_stack, axis=0), axis=0), 5.0, 99.0).astype(np.float32)

    if work_sxx_db.shape != sxx_db_local.shape:
        target_rows = int(sxx_db_local.shape[0])
        target_cols = int(sxx_db_local.shape[1])
        background = resize_float_image(background, width=target_cols, height=target_rows, resample=Image.BILINEAR)
        residual_db = resize_float_image(residual_db, width=target_cols, height=target_rows, resample=Image.BILINEAR)
        residual_n = resize_float_image(residual_n, width=target_cols, height=target_rows, resample=Image.BILINEAR)
        coherence_px = resize_float_image(coherence_px, width=target_cols, height=target_rows, resample=Image.BILINEAR)
        energy_px = resize_float_image(energy_px, width=target_cols, height=target_rows, resample=Image.BILINEAR)
        gate_px = resize_float_image(gate_px, width=target_cols, height=target_rows, resample=Image.BILINEAR)

    return {
        "background_db": background.astype(np.float32),
        "residual_db": residual_db.astype(np.float32),
        "residual_n": residual_n.astype(np.float32),
        "coherence_px": coherence_px.astype(np.float32),
        "energy_px": energy_px.astype(np.float32),
        "gate_px": gate_px.astype(np.float32),
    }


def detect_chunk_coherent_power(
    corrected_chunk: np.ndarray,
    cfg: CoherentPowerConfig,
    valid_row_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    corrected_chunk = np.asarray(corrected_chunk, dtype=np.float32)
    if valid_row_mask is None:
        valid_row_mask = np.ones(corrected_chunk.shape[0], dtype=bool)
    else:
        valid_row_mask = np.asarray(valid_row_mask, dtype=bool).reshape(-1)
        if valid_row_mask.shape[0] != corrected_chunk.shape[0]:
            raise ValueError("valid_row_mask length must match the chunk rows")

    coherence_maps = multi_scale_structure_tensor_gate(corrected_chunk)
    coherence_px = _normalize_map01_local(np.asarray(coherence_maps["coherence_px"], dtype=np.float32), 5.0, 99.0)
    power_px = _normalize_map01_local(
        _local_relative_power_support_map(corrected_chunk, valid_row_mask=valid_row_mask, floor_q=30.0),
        5.0,
        95.0,
    )
    combined_score = _normalize_map01_local(
        float(cfg.coherence_weight) * coherence_px + float(cfg.power_weight) * power_px,
        5.0,
        95.0,
    )

    valid_score_mask = np.repeat(valid_row_mask[:, None], corrected_chunk.shape[1], axis=1)
    valid_scores = combined_score[valid_score_mask]
    support_threshold = _robust_high_quantile_threshold(valid_scores, cfg.coherence_power_support_q) if valid_scores.size else 1.0
    support_px = _smooth_binary_label_map(
        np.logical_and(combined_score >= support_threshold, valid_score_mask).astype(np.uint8),
        iters=1,
        min_component_size=max(3, cfg.min_component_size // 2),
    ).astype(bool)

    final_mask_source = np.logical_and(valid_score_mask, support_px)
    final_scores = combined_score[final_mask_source]
    final_threshold = _robust_high_quantile_threshold(final_scores, cfg.coherence_power_q) if final_scores.size else support_threshold
    mask_px = _smooth_binary_label_map(
        np.logical_and.reduce((combined_score >= final_threshold, valid_score_mask, support_px)).astype(np.uint8),
        iters=1,
        min_component_size=cfg.min_component_size,
    ).astype(bool)

    coherence_px[~valid_score_mask] = 0.0
    power_px[~valid_score_mask] = 0.0
    combined_score[~valid_score_mask] = 0.0
    support_px[~valid_score_mask] = False
    mask_px[~valid_score_mask] = False
    t1 = time.perf_counter()

    return {
        "coherence_px": coherence_px.astype(np.float32),
        "power_px": power_px.astype(np.float32),
        "score_px": combined_score.astype(np.float32),
        "support_px": support_px.astype(bool),
        "mask_px": mask_px.astype(bool),
        "support_threshold": float(support_threshold),
        "score_threshold": float(final_threshold),
        "valid_score_mask": valid_score_mask.astype(bool),
        "timing_ms": {
            "coherence_power_ms": (t1 - t0) * 1000.0,
            "total_ms": (t1 - t0) * 1000.0,
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


def _fill_nearly_continuous_time_gaps(
    mask: np.ndarray,
    max_gap_px: int,
    min_continuity_ratio: float = 0.85,
) -> np.ndarray:
    filled = np.asarray(mask, dtype=bool).copy()
    max_gap_px = max(0, int(max_gap_px))
    if max_gap_px <= 0:
        return filled

    min_continuity_ratio = float(np.clip(min_continuity_ratio, 0.0, 1.0))
    for row_index in range(filled.shape[0]):
        row = filled[row_index]
        active_cols = np.flatnonzero(row)
        if active_cols.size < 2:
            continue

        run_starts = [int(active_cols[0])]
        run_stops: list[int] = []
        previous_col = int(active_cols[0])
        for current_col in (int(value) for value in active_cols[1:]):
            if current_col != previous_col + 1:
                run_stops.append(previous_col + 1)
                run_starts.append(current_col)
            previous_col = current_col
        run_stops.append(previous_col + 1)

        for left_start, left_stop, right_start, right_stop in zip(
            run_starts,
            run_stops,
            run_starts[1:],
            run_stops[1:],
        ):
            gap_width = int(right_start - left_stop)
            if gap_width <= 0 or gap_width > max_gap_px:
                continue
            left_width = int(left_stop - left_start)
            right_width = int(right_stop - right_start)
            continuity_ratio = float(left_width + right_width) / float(left_width + gap_width + right_width)
            if continuity_ratio >= min_continuity_ratio:
                row[left_stop:right_start] = True
    return filled


def _true_runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    mask_1d = np.asarray(mask_1d, dtype=bool).reshape(-1)
    if mask_1d.size == 0:
        return []

    padded = np.pad(mask_1d.astype(np.int8), (1, 1), mode="constant")
    transitions = np.diff(padded)
    run_starts = np.flatnonzero(transitions == 1)
    run_stops = np.flatnonzero(transitions == -1)
    return [(int(start), int(stop)) for start, stop in zip(run_starts, run_stops)]


def _split_component_candidate_masks(
    component_mask_local: np.ndarray,
    min_freq_span_px: int,
    min_time_span_px: int,
) -> list[dict[str, Any]]:
    component_mask_local = np.asarray(component_mask_local, dtype=bool)
    if component_mask_local.ndim != 2 or not np.any(component_mask_local):
        return [{"mask": component_mask_local.astype(bool), "split_role": "unsplit", "split_applied": False}]

    active_cols = np.flatnonzero(np.any(component_mask_local, axis=0))
    if active_cols.size < max(6, 2 * int(min_time_span_px)):
        return [{"mask": component_mask_local.astype(bool), "split_role": "unsplit", "split_applied": False}]

    col_span = np.zeros(component_mask_local.shape[1], dtype=np.int32)
    for col in active_cols:
        rows = np.flatnonzero(component_mask_local[:, col])
        if rows.size:
            col_span[col] = int(rows.max() - rows.min() + 1)

    active_spans = col_span[active_cols]
    if active_spans.size < max(6, 2 * int(min_time_span_px)):
        return [{"mask": component_mask_local.astype(bool), "split_role": "unsplit", "split_applied": False}]

    global_rows = np.flatnonzero(np.any(component_mask_local, axis=1))
    if global_rows.size == 0:
        return [{"mask": component_mask_local.astype(bool), "split_role": "unsplit", "split_applied": False}]

    baseline_span = float(np.quantile(active_spans.astype(np.float32), 0.35))
    global_span = int(global_rows.max() - global_rows.min() + 1)
    burst_span_threshold = max(
        int(np.ceil(baseline_span * 1.8)),
        int(np.ceil(baseline_span + max(4.0, float(min_freq_span_px) * 0.5))),
        int(min_freq_span_px),
    )
    if burst_span_threshold >= global_span:
        return [{"mask": component_mask_local.astype(bool), "split_role": "unsplit", "split_applied": False}]

    burst_cols_mask = np.zeros(component_mask_local.shape[1], dtype=bool)
    burst_cols_mask[active_cols] = active_spans >= burst_span_threshold
    burst_runs = [
        (start, stop)
        for start, stop in _true_runs(burst_cols_mask)
        if (stop - start) >= int(min_time_span_px)
        and (stop - start) < max(int(active_cols.size * 0.7), int(min_time_span_px) + 1)
    ]
    if not burst_runs:
        return [{"mask": component_mask_local.astype(bool), "split_role": "unsplit", "split_applied": False}]

    non_burst_cols_mask = np.zeros(component_mask_local.shape[1], dtype=bool)
    non_burst_cols_mask[active_cols] = True
    non_burst_cols_mask[burst_cols_mask] = False
    non_burst_cols = np.flatnonzero(non_burst_cols_mask)
    if non_burst_cols.size < max(4, 2 * int(min_time_span_px)):
        return [{"mask": component_mask_local.astype(bool), "split_role": "unsplit", "split_applied": False}]

    row_hits = np.count_nonzero(component_mask_local[:, non_burst_cols], axis=1)
    min_row_hits = max(2, int(np.ceil(non_burst_cols.size * 0.45)))
    carrier_row_runs = _true_runs(row_hits >= min_row_hits)
    if not carrier_row_runs:
        return [{"mask": component_mask_local.astype(bool), "split_role": "unsplit", "split_applied": False}]

    carrier_freq_start, carrier_freq_stop = max(carrier_row_runs, key=lambda run: run[1] - run[0])
    carrier_freq_span = int(carrier_freq_stop - carrier_freq_start)
    if carrier_freq_span < max(2, int(np.floor(baseline_span))) or carrier_freq_span >= burst_span_threshold:
        return [{"mask": component_mask_local.astype(bool), "split_role": "unsplit", "split_applied": False}]

    carrier_mask = np.zeros_like(component_mask_local, dtype=bool)
    carrier_mask[carrier_freq_start:carrier_freq_stop, :] = component_mask_local[carrier_freq_start:carrier_freq_stop, :]
    if np.count_nonzero(carrier_mask) < max(2, int(min_time_span_px) * 2):
        return [{"mask": component_mask_local.astype(bool), "split_role": "unsplit", "split_applied": False}]

    candidate_masks: list[dict[str, Any]] = [
        {"mask": carrier_mask.astype(bool), "split_role": "persistent_carrier", "split_applied": True}
    ]
    for start, stop in burst_runs:
        burst_mask = np.zeros_like(component_mask_local, dtype=bool)
        burst_mask[:, start:stop] = component_mask_local[:, start:stop]
        if np.count_nonzero(burst_mask) == 0:
            continue
        candidate_masks.append({
            "mask": burst_mask.astype(bool),
            "split_role": "transient_wideband_burst",
            "split_applied": True,
        })

    if len(candidate_masks) < 2:
        return [{"mask": component_mask_local.astype(bool), "split_role": "unsplit", "split_applied": False}]

    return candidate_masks


def _component_envelope_area(component_mask: np.ndarray) -> int:
    component_mask = np.asarray(component_mask, dtype=bool)
    if component_mask.ndim != 2 or not np.any(component_mask):
        return 0

    envelope_area = 0
    active_cols = np.flatnonzero(np.any(component_mask, axis=0))
    for col in active_cols:
        rows = np.flatnonzero(component_mask[:, col])
        if rows.size == 0:
            continue
        envelope_area += int(rows.max() - rows.min() + 1)
    return int(envelope_area)


def group_signal_mask_regions(
    mask: np.ndarray,
    score_map: np.ndarray | None = None,
    valid_row_mask: np.ndarray | None = None,
    bridge_freq_px: int = 21,
    bridge_time_px: int = 3,
    min_component_size: int = 24,
    min_freq_span_px: int = 12,
    min_time_span_px: int = 1,
    min_density: float = 0.08,
    time_continuity_ratio: float = 0.85,
) -> dict[str, Any]:
    raw_mask = np.asarray(mask, dtype=bool)
    if raw_mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {raw_mask.shape}")

    working_mask = raw_mask.copy()
    if valid_row_mask is not None:
        valid_row_mask = np.asarray(valid_row_mask, dtype=bool).reshape(-1)
        if valid_row_mask.shape[0] != working_mask.shape[0]:
            raise ValueError("valid_row_mask length must match mask rows")
        working_mask[~valid_row_mask, :] = False

    bridged_mask = working_mask.copy()
    if int(bridge_freq_px) > 1:
        bridged_mask = ndimage.binary_closing(
            bridged_mask,
            structure=np.ones((max(1, int(bridge_freq_px)), 1), dtype=bool),
        )
    bridged_mask = _fill_nearly_continuous_time_gaps(
        bridged_mask,
        max_gap_px=bridge_time_px,
        min_continuity_ratio=time_continuity_ratio,
    )

    component_labels, n_components = ndimage.label(bridged_mask)
    candidate_component_labels = np.zeros_like(component_labels, dtype=np.int32)
    grouped_mask = np.zeros_like(working_mask, dtype=bool)
    boxes: list[dict[str, int | float]] = []
    component_rows: list[dict[str, int | float]] = []
    output_component_id = 0

    score_map_arr = None if score_map is None else np.asarray(score_map, dtype=np.float32)
    active_scores = None
    if score_map_arr is not None and np.any(working_mask):
        active_scores = score_map_arr[working_mask]
    peak_score_floor = float(np.quantile(active_scores, 0.50)) if active_scores is not None and active_scores.size else 0.0

    for component_id in range(1, int(n_components) + 1):
        component_mask = component_labels == component_id
        if not np.any(component_mask):
            continue

        row_coords, col_coords = np.nonzero(component_mask)
        parent_freq_start = int(row_coords.min())
        parent_freq_stop = int(row_coords.max()) + 1
        parent_time_start = int(col_coords.min())
        parent_time_stop = int(col_coords.max()) + 1
        component_mask_local = component_mask[parent_freq_start:parent_freq_stop, parent_time_start:parent_time_stop]

        candidate_masks = _split_component_candidate_masks(
            component_mask_local,
            min_freq_span_px=int(min_freq_span_px),
            min_time_span_px=int(min_time_span_px),
        )

        for candidate in candidate_masks:
            candidate_mask_local = np.asarray(candidate["mask"], dtype=bool)
            if not np.any(candidate_mask_local):
                continue

            local_row_coords, local_col_coords = np.nonzero(candidate_mask_local)
            local_freq_start = int(local_row_coords.min())
            local_freq_stop = int(local_row_coords.max()) + 1
            local_time_start = int(local_col_coords.min())
            local_time_stop = int(local_col_coords.max()) + 1

            freq_start = int(parent_freq_start + local_freq_start)
            freq_stop = int(parent_freq_start + local_freq_stop)
            time_start = int(parent_time_start + local_time_start)
            time_stop = int(parent_time_start + local_time_stop)
            freq_span = int(freq_stop - freq_start)
            time_span = int(time_stop - time_start)
            cropped_candidate_mask = candidate_mask_local[local_freq_start:local_freq_stop, local_time_start:local_time_stop]
            bbox_area = max(freq_span * time_span, 1)
            envelope_area = max(_component_envelope_area(cropped_candidate_mask), 1)
            filled_area = int(np.count_nonzero(cropped_candidate_mask))
            bbox_density = float(filled_area / bbox_area)
            envelope_density = float(filled_area / envelope_area)
            density = envelope_density

            if score_map_arr is not None:
                component_scores = score_map_arr[freq_start:freq_stop, time_start:time_stop][cropped_candidate_mask]
                score_peak = float(np.max(component_scores)) if component_scores.size else 0.0
                score_mean = float(np.mean(component_scores)) if component_scores.size else 0.0
            else:
                score_peak = 0.0
                score_mean = 0.0
            meets_min_component_size = bool(filled_area >= int(min_component_size))
            meets_min_freq_span = bool(freq_span >= int(min_freq_span_px))
            meets_min_time_span = bool(time_span >= int(min_time_span_px))
            meets_min_density = bool(density >= float(min_density))
            meets_peak_score_floor = bool(score_peak >= peak_score_floor)
            keep_component = (
                meets_min_component_size
                and meets_min_freq_span
                and meets_min_time_span
                and meets_min_density
                and meets_peak_score_floor
            )
            failed_reasons = []
            if not meets_min_component_size:
                failed_reasons.append("min_component_size")
            if not meets_min_freq_span:
                failed_reasons.append("min_freq_span_px")
            if not meets_min_time_span:
                failed_reasons.append("min_time_span_px")
            if not meets_min_density:
                failed_reasons.append("min_density")
            if not meets_peak_score_floor:
                failed_reasons.append("peak_score_floor")

            output_component_id += 1
            candidate_label_view = candidate_component_labels[freq_start:freq_stop, time_start:time_stop]
            candidate_label_view[np.logical_and(cropped_candidate_mask, candidate_label_view == 0)] = int(output_component_id)

            component_rows.append({
                "component_id": int(output_component_id),
                "parent_component_id": int(component_id),
                "split_role": str(candidate.get("split_role", "unsplit")),
                "split_applied": bool(candidate.get("split_applied", False)),
                "freq_start": freq_start,
                "freq_stop": freq_stop,
                "time_start": time_start,
                "time_stop": time_stop,
                "size_px": filled_area,
                "freq_span": freq_span,
                "freq_span_px": freq_span,
                "time_span": time_span,
                "time_span_px": time_span,
                "filled_area": filled_area,
                "density": density,
                "bbox_area": bbox_area,
                "bbox_density": bbox_density,
                "envelope_area": envelope_area,
                "envelope_density": envelope_density,
                "score_mean": score_mean,
                "score_peak": score_peak,
                "score_peak_minus_floor": float(score_peak - peak_score_floor),
                "min_component_size_threshold": int(min_component_size),
                "min_freq_span_threshold_px": int(min_freq_span_px),
                "min_time_span_threshold_px": int(min_time_span_px),
                "min_density_threshold": float(min_density),
                "peak_score_floor_value": peak_score_floor,
                "min_component_size": meets_min_component_size,
                "min_freq_span_px": meets_min_freq_span,
                "min_time_span_px": meets_min_time_span,
                "min_density": meets_min_density,
                "peak_score_floor": meets_peak_score_floor,
                "failed_reasons": failed_reasons,
                "primary_failed_reason": failed_reasons[0] if failed_reasons else None,
                "accepted": bool(keep_component),
                "kept": bool(keep_component),
            })

            if not keep_component:
                continue

            grouped_mask[freq_start:freq_stop, time_start:time_stop] |= cropped_candidate_mask
            boxes.append({
                "freq_start": freq_start,
                "freq_stop": freq_stop,
                "time_start": time_start,
                "time_stop": time_stop,
                "freq_span": freq_span,
                "time_span": time_span,
                "filled_area": filled_area,
                "density": density,
                "bbox_density": bbox_density,
                "envelope_density": envelope_density,
                "score_mean": score_mean,
                "score_peak": score_peak,
                "split_role": str(candidate.get("split_role", "unsplit")),
                "split_applied": bool(candidate.get("split_applied", False)),
                "parent_component_id": int(component_id),
            })

    if valid_row_mask is not None:
        grouped_mask[~valid_row_mask, :] = False

    return {
        "seed_mask": working_mask.astype(bool),
        "bridged_mask": bridged_mask.astype(bool),
        "component_labels": candidate_component_labels.astype(np.int32),
        "grouped_mask": grouped_mask.astype(bool),
        "boxes": boxes,
        "components": component_rows,
        "peak_score_floor": peak_score_floor,
    }


def build_grouped_detection_regions(
    merged_score: np.ndarray,
    merged_mask: np.ndarray,
    merged_support: np.ndarray,
    valid_row_mask: np.ndarray | None = None,
    seed_score_q: float = 0.72,
    bridge_freq_px: int = 33,
    bridge_time_px: int = 5,
    min_component_size: int = 24,
    min_freq_span_px: int = 18,
    min_time_span_px: int = 2,
    min_density: float = 0.06,
    time_continuity_ratio: float = 0.85,
) -> dict[str, Any]:
    merged_score = np.asarray(merged_score, dtype=np.float32)
    merged_mask = np.asarray(merged_mask, dtype=bool)
    merged_support = np.asarray(merged_support, dtype=bool)
    if merged_score.shape != merged_mask.shape or merged_score.shape != merged_support.shape:
        raise ValueError("merged_score, merged_mask, and merged_support must share the same shape")

    if valid_row_mask is None:
        valid_row_mask = np.ones(merged_score.shape[0], dtype=bool)
    else:
        valid_row_mask = np.asarray(valid_row_mask, dtype=bool).reshape(-1)
        if valid_row_mask.shape[0] != merged_score.shape[0]:
            raise ValueError("valid_row_mask length must match the merged map rows")

    valid_seed_mask = np.logical_and(valid_row_mask[:, None], merged_support)
    valid_seed_scores = merged_score[valid_seed_mask]
    seed_threshold = _robust_high_quantile_threshold(valid_seed_scores, seed_score_q) if valid_seed_scores.size else 1.0
    seed_mask = np.logical_or(merged_mask, np.logical_and(merged_support, merged_score >= seed_threshold))
    region_groups = group_signal_mask_regions(
        seed_mask,
        score_map=merged_score,
        valid_row_mask=valid_row_mask,
        bridge_freq_px=bridge_freq_px,
        bridge_time_px=bridge_time_px,
        min_component_size=min_component_size,
        min_freq_span_px=min_freq_span_px,
        min_time_span_px=min_time_span_px,
        min_density=min_density,
        time_continuity_ratio=time_continuity_ratio,
    )
    region_groups["seed_threshold"] = float(seed_threshold)
    region_groups["seed_score_q"] = float(seed_score_q)
    region_groups["raw_mask_fraction"] = float(np.mean(merged_mask))
    region_groups["grouped_mask_fraction"] = float(np.mean(np.asarray(region_groups["grouped_mask"], dtype=bool)))
    return region_groups


def _boxes_overlap(box_a: dict[str, Any], box_b: dict[str, Any]) -> bool:
    return (
        int(box_a["freq_start"]) < int(box_b["freq_stop"])
        and int(box_b["freq_start"]) < int(box_a["freq_stop"])
        and int(box_a["time_start"]) < int(box_b["time_stop"])
        and int(box_b["time_start"]) < int(box_a["time_stop"])
    )


def _boxes_should_merge(box_a: dict[str, Any], box_b: dict[str, Any]) -> bool:
    if not _boxes_overlap(box_a, box_b):
        return False

    role_a = str(box_a.get("split_role", "unsplit"))
    role_b = str(box_b.get("split_role", "unsplit"))
    if {role_a, role_b} == {"persistent_carrier", "transient_wideband_burst"}:
        return False
    return True


def _merge_box_cluster(cluster: list[dict[str, Any]]) -> dict[str, Any]:
    if not cluster:
        raise ValueError("Cannot merge an empty box cluster")

    freq_start = min(int(box["freq_start"]) for box in cluster)
    freq_stop = max(int(box["freq_stop"]) for box in cluster)
    time_start = min(int(box["time_start"]) for box in cluster)
    time_stop = max(int(box["time_stop"]) for box in cluster)
    filled_area = int(sum(int(box.get("filled_area", 0)) for box in cluster))
    bbox_area = max(int(freq_stop - freq_start) * int(time_stop - time_start), 1)
    score_weight = max(filled_area, 1)
    score_mean = float(
        sum(float(box.get("score_mean", 0.0)) * max(int(box.get("filled_area", 0)), 1) for box in cluster)
        / float(score_weight)
    )
    split_roles = sorted({str(box.get("split_role", "unsplit")) for box in cluster})
    split_role = split_roles[0] if len(split_roles) == 1 else "mixed"
    source_chunk_indices = sorted({
        int(chunk_index)
        for box in cluster
        for chunk_index in box.get("source_chunk_indices", [])
    })
    parent_component_ids = sorted({
        int(parent_component_id)
        for box in cluster
        if box.get("parent_component_id") is not None
        for parent_component_id in [box.get("parent_component_id")]
    })
    return {
        "freq_start": freq_start,
        "freq_stop": freq_stop,
        "time_start": time_start,
        "time_stop": time_stop,
        "freq_span": int(freq_stop - freq_start),
        "time_span": int(time_stop - time_start),
        "filled_area": filled_area,
        "density": float(filled_area / bbox_area),
        "score_mean": score_mean,
        "score_peak": float(max(float(box.get("score_peak", 0.0)) for box in cluster)),
        "split_role": split_role,
        "split_roles": split_roles,
        "split_applied": bool(any(bool(box.get("split_applied", False)) for box in cluster)),
        "source_box_count": len(cluster),
        "source_chunk_indices": source_chunk_indices,
        "parent_component_ids": parent_component_ids,
    }


def _boxes_to_mask(
    shape: tuple[int, int],
    boxes: list[dict[str, Any]],
    valid_row_mask: np.ndarray | None = None,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for box in boxes:
        freq_start = max(0, min(shape[0], int(box["freq_start"])))
        freq_stop = max(freq_start, min(shape[0], int(box["freq_stop"])))
        time_start = max(0, min(shape[1], int(box["time_start"])))
        time_stop = max(time_start, min(shape[1], int(box["time_stop"])))
        if freq_stop <= freq_start or time_stop <= time_start:
            continue
        mask[freq_start:freq_stop, time_start:time_stop] = True
    if valid_row_mask is not None:
        valid_row_mask = np.asarray(valid_row_mask, dtype=bool).reshape(-1)
        if valid_row_mask.shape[0] != shape[0]:
            raise ValueError("valid_row_mask length must match mask rows")
        mask[~valid_row_mask, :] = False
    return mask


def _project_chunk_boxes_to_global(
    chunk_results: list[dict[str, Any]],
    global_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    projected_boxes: list[dict[str, Any]] = []
    for chunk in chunk_results:
        row_start = int(chunk["row_start"])
        row_stop = int(chunk["row_stop"])
        chunk_index = int(chunk["chunk_index"])
        local_boxes = list(chunk.get("grouped_boxes", []))
        if not local_boxes:
            continue
        for box in local_boxes:
            freq_start = max(0, min(global_shape[0], row_start + int(box["freq_start"])))
            freq_stop = max(freq_start, min(global_shape[0], row_start + int(box["freq_stop"])))
            time_start = max(0, min(global_shape[1], int(box["time_start"])))
            time_stop = max(time_start, min(global_shape[1], int(box["time_stop"])))
            if freq_stop <= freq_start or time_stop <= time_start:
                continue
            projected_boxes.append({
                "freq_start": freq_start,
                "freq_stop": freq_stop,
                "time_start": time_start,
                "time_stop": time_stop,
                "freq_span": int(freq_stop - freq_start),
                "time_span": int(time_stop - time_start),
                "filled_area": int(box.get("filled_area", 0)),
                "density": float(box.get("density", 0.0)),
                "score_mean": float(box.get("score_mean", 0.0)),
                "score_peak": float(box.get("score_peak", 0.0)),
                "split_role": str(box.get("split_role", "unsplit")),
                "split_applied": bool(box.get("split_applied", False)),
                "parent_component_id": box.get("parent_component_id"),
                "source_chunk_indices": [chunk_index],
                "source_row_start": row_start,
                "source_row_stop": row_stop,
            })
    return projected_boxes


def _project_chunk_grouped_masks_to_global(
    chunk_results: list[dict[str, Any]],
    global_shape: tuple[int, int],
    valid_row_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    source_mask = np.zeros(global_shape, dtype=bool)
    projected_boxes = _project_chunk_boxes_to_global(chunk_results, global_shape)
    for chunk in chunk_results:
        row_start = int(chunk["row_start"])
        row_stop = int(chunk["row_stop"])
        local_mask = np.asarray(chunk.get("grouped_mask", chunk.get("mask_px")), dtype=bool)
        expected_shape = (row_stop - row_start, global_shape[1])
        if local_mask.shape != expected_shape:
            raise ValueError(
                f"Projected subsection mask shape {local_mask.shape} does not match expected {expected_shape}"
            )
        source_mask[row_start:row_stop, :] |= local_mask
    if valid_row_mask is not None:
        valid_row_mask = np.asarray(valid_row_mask, dtype=bool).reshape(-1)
        if valid_row_mask.shape[0] != global_shape[0]:
            raise ValueError("valid_row_mask length must match mask rows")
        source_mask[~valid_row_mask, :] = False
    return source_mask.astype(bool), projected_boxes


def _merge_projected_subsection_boxes(
    global_shape: tuple[int, int],
    chunk_results: list[dict[str, Any]],
    merged_score: np.ndarray,
    valid_row_mask: np.ndarray | None = None,
    bridge_freq_px: int = 33,
    bridge_time_px: int = 5,
    min_component_size: int = 24,
    min_freq_span_px: int = 18,
    min_time_span_px: int = 2,
    min_density: float = 0.06,
    time_continuity_ratio: float = 0.85,
) -> dict[str, Any]:
    merged_score = np.asarray(merged_score, dtype=np.float32)
    source_mask, projected_boxes = _project_chunk_grouped_masks_to_global(
        chunk_results,
        global_shape,
        valid_row_mask=valid_row_mask,
    )
    if not np.any(source_mask):
        return {
            "boxes": [],
            "grouped_mask": np.zeros(global_shape, dtype=bool),
            "source_boxes": [],
            "source_mask": source_mask.astype(bool),
        }

    if any(bool(box.get("split_applied", False)) for box in projected_boxes):
        merged_boxes: list[dict[str, Any]] = []
        visited = [False] * len(projected_boxes)
        for start_index in range(len(projected_boxes)):
            if visited[start_index]:
                continue
            pending = [start_index]
            visited[start_index] = True
            cluster_indices: list[int] = []
            while pending:
                current_index = pending.pop()
                cluster_indices.append(current_index)
                current_box = projected_boxes[current_index]
                for other_index, other_box in enumerate(projected_boxes):
                    if visited[other_index]:
                        continue
                    if _boxes_should_merge(current_box, other_box):
                        visited[other_index] = True
                        pending.append(other_index)

            cluster = [projected_boxes[index] for index in cluster_indices]
            merged_boxes.append(_merge_box_cluster(cluster))

        grouped_mask = _boxes_to_mask(global_shape, merged_boxes, valid_row_mask=valid_row_mask)
        return {
            "boxes": merged_boxes,
            "grouped_mask": grouped_mask.astype(bool),
            "source_boxes": projected_boxes,
            "source_mask": source_mask.astype(bool),
            "grouping": None,
        }

    grouping = group_signal_mask_regions(
        source_mask,
        score_map=merged_score,
        valid_row_mask=valid_row_mask,
        bridge_freq_px=bridge_freq_px,
        bridge_time_px=bridge_time_px,
        min_component_size=min_component_size,
        min_freq_span_px=min_freq_span_px,
        min_time_span_px=min_time_span_px,
        min_density=min_density,
        time_continuity_ratio=time_continuity_ratio,
    )
    merged_boxes: list[dict[str, Any]] = []
    for box in grouping["boxes"]:
        overlapping_source_boxes = [
            source_box
            for source_box in projected_boxes
            if _boxes_overlap(box, source_box)
        ]
        merged_boxes.append({
            **box,
            "source_box_count": len(overlapping_source_boxes),
            "source_chunk_indices": sorted({
                int(chunk_index)
                for source_box in overlapping_source_boxes
                for chunk_index in source_box.get("source_chunk_indices", [])
            }),
        })

    grouped_mask = np.asarray(grouping["grouped_mask"], dtype=bool)
    return {
        "boxes": merged_boxes,
        "grouped_mask": grouped_mask.astype(bool),
        "source_boxes": projected_boxes,
        "source_mask": source_mask.astype(bool),
        "grouping": grouping,
    }


def merge_chunk_results(
    global_shape: tuple[int, int],
    chunk_results: list[dict[str, Any]],
    final_score_q: float = 0.92,
    min_component_size: int = 6,
    global_valid_row_mask: np.ndarray | None = None,
    coherence_weight: float = 0.55,
    power_weight: float = 0.45,
    grouping_seed_score_q: float = 0.72,
    grouping_bridge_freq_px: int = 33,
    grouping_bridge_time_px: int = 5,
    grouping_min_component_size: int = 24,
    grouping_min_freq_span_px: int = 18,
    grouping_min_time_span_px: int = 2,
    grouping_min_density: float = 0.06,
    grouping_time_continuity_ratio: float = 0.85,
) -> dict[str, Any]:
    merged_score_sum = np.zeros(global_shape, dtype=np.float32)
    merged_support = np.zeros(global_shape, dtype=bool)
    merged_coherence_sum = np.zeros(global_shape, dtype=np.float32)
    merged_power_sum = np.zeros(global_shape, dtype=np.float32)
    merged_weight = np.zeros(global_shape, dtype=np.float32)

    for chunk in chunk_results:
        row_start = int(chunk["row_start"])
        row_stop = int(chunk["row_stop"])
        chunk_weights = _chunk_blend_weights(row_stop - row_start)[:, None]
        valid_score_mask = np.asarray(chunk.get("valid_score_mask", np.ones((row_stop - row_start, global_shape[1]), dtype=bool)), dtype=bool)
        blend_weights = chunk_weights * valid_score_mask.astype(np.float32)
        score_px = np.asarray(chunk["score_px"], dtype=np.float32)
        coherence_px = np.asarray(chunk["coherence_px"], dtype=np.float32)
        power_px = np.asarray(chunk["power_px"], dtype=np.float32)
        merged_score_sum[row_start:row_stop, :] += score_px * blend_weights
        merged_coherence_sum[row_start:row_stop, :] += coherence_px * blend_weights
        merged_power_sum[row_start:row_stop, :] += power_px * blend_weights
        merged_weight[row_start:row_stop, :] += blend_weights
        merged_support[row_start:row_stop, :] |= np.asarray(chunk["support_px"], dtype=bool)

    valid_row_mask = np.ones(global_shape[0], dtype=bool)
    if global_valid_row_mask is not None:
        valid_row_mask = np.asarray(global_valid_row_mask, dtype=bool).reshape(-1)
        if valid_row_mask.shape[0] != global_shape[0]:
            raise ValueError("global_valid_row_mask length must match global_shape rows")

    merged_coherence = np.zeros(global_shape, dtype=np.float32)
    merged_power = np.zeros(global_shape, dtype=np.float32)
    overlap_mask = merged_weight > 0.0
    merged_coherence[overlap_mask] = merged_coherence_sum[overlap_mask] / merged_weight[overlap_mask]
    merged_power[overlap_mask] = merged_power_sum[overlap_mask] / merged_weight[overlap_mask]
    combined_score = (
        float(coherence_weight) * merged_coherence
        + float(power_weight) * merged_power
    ).astype(np.float32)
    merged_score = _normalize_map01_masked(
        combined_score,
        mask=np.logical_and(valid_row_mask[:, None], overlap_mask),
        low_q=5.0,
        high_q=95.0,
    )

    valid_scores = merged_score[np.logical_and(valid_row_mask[:, None], merged_support)]
    threshold = _robust_high_quantile_threshold(valid_scores, final_score_q) if valid_scores.size else 1.0
    raw_merged_mask = _smooth_binary_label_map(
        np.logical_and(merged_score >= threshold, np.logical_and(valid_row_mask[:, None], merged_support)).astype(np.uint8),
        iters=1,
        min_component_size=min_component_size,
    )
    raw_merged_mask[~valid_row_mask, :] = 0
    merged_score[~valid_row_mask, :] = 0.0
    merged_support[~valid_row_mask, :] = False
    merged_coherence[~valid_row_mask, :] = 0.0
    merged_power[~valid_row_mask, :] = 0.0

    merged_box_groups = _merge_projected_subsection_boxes(
        global_shape=global_shape,
        chunk_results=chunk_results,
        merged_score=merged_score,
        valid_row_mask=valid_row_mask,
        bridge_freq_px=grouping_bridge_freq_px,
        bridge_time_px=grouping_bridge_time_px,
        min_component_size=max(int(grouping_min_component_size), int(min_component_size)),
        min_freq_span_px=grouping_min_freq_span_px,
        min_time_span_px=grouping_min_time_span_px,
        min_density=grouping_min_density,
        time_continuity_ratio=grouping_time_continuity_ratio,
    )
    merged_boxes = list(merged_box_groups["boxes"])
    if merged_boxes:
        merged_mask = np.asarray(merged_box_groups["grouped_mask"], dtype=bool)
        merged_region_groups = None
    else:
        merged_region_groups = build_grouped_detection_regions(
            merged_score=merged_score,
            merged_mask=raw_merged_mask,
            merged_support=merged_support,
            valid_row_mask=valid_row_mask,
            seed_score_q=grouping_seed_score_q,
            bridge_freq_px=grouping_bridge_freq_px,
            bridge_time_px=grouping_bridge_time_px,
            min_component_size=max(int(grouping_min_component_size), int(min_component_size)),
            min_freq_span_px=grouping_min_freq_span_px,
            min_time_span_px=grouping_min_time_span_px,
            min_density=grouping_min_density,
            time_continuity_ratio=grouping_time_continuity_ratio,
        )
        merged_mask = np.asarray(merged_region_groups["grouped_mask"], dtype=bool)
        merged_boxes = list(merged_region_groups["boxes"])
    return {
        "merged_score": merged_score.astype(np.float32),
        "merged_mask": merged_mask.astype(bool),
        "raw_merged_mask": np.asarray(raw_merged_mask, dtype=bool),
        "merged_threshold": float(threshold),
        "valid_row_mask": valid_row_mask.astype(bool),
        "merged_support": merged_support.astype(bool),
        "merged_boxes": merged_boxes,
        "merged_box_groups": merged_box_groups,
        "merged_region_groups": merged_region_groups,
        "merged_coherence": merged_coherence.astype(np.float32),
        "merged_power": merged_power.astype(np.float32),
    }


def run_coherent_power_pipeline(
    input_record: dict[str, Any],
    cfg: CoherentPowerConfig,
    progress_callback: Callable[[str, int, int, dict[str, Any] | None], None] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    stage_timing_ms: dict[str, float] = {}

    def _emit_progress(stage: str, completed: int, total: int, info: dict[str, Any] | None = None) -> None:
        if progress_callback is None:
            return
        progress_callback(stage, int(completed), int(total), info)

    calibrated_axis = has_calibrated_frequency_axis(input_record)
    effective_ignore_sideband_hz = cfg.ignore_sideband_hz if calibrated_axis else None
    t_ignore_start = time.perf_counter()
    ignore_info = compute_ignore_sideband_rows(
        input_record["freq_axis_hz"],
        ignore_sideband_percent=cfg.ignore_sideband_percent,
        min_keep_rows=16,
        ignore_sideband_hz=effective_ignore_sideband_hz,
    )
    stage_timing_ms["ignore_sideband_ms"] = (time.perf_counter() - t_ignore_start) * 1000.0
    valid_row_mask = np.asarray(ignore_info["valid_row_mask"], dtype=bool)
    _emit_progress("ignore_sideband", 1, 1, {"applied_bins": int(ignore_info["applied_bins"])})

    t_frontend_start = time.perf_counter()
    correction = apply_global_frontend_correction(
        input_record["sxx_db"],
        row_q=cfg.frontend_row_q,
        reference_q=cfg.frontend_reference_q,
        smooth_sigma=cfg.frontend_smooth_sigma,
        max_boost_db=cfg.frontend_max_boost_db,
        valid_row_mask=valid_row_mask,
    )
    stage_timing_ms["frontend_ms"] = (time.perf_counter() - t_frontend_start) * 1000.0
    corrected_sxx_db = np.asarray(correction["corrected_sxx_db"], dtype=np.float32)
    _emit_progress("frontend", 1, 1, None)

    t_chunk_plan_start = time.perf_counter()
    chunk_plan = build_frequency_chunks(
        input_record["freq_axis_hz"],
        chunk_bandwidth_hz=cfg.chunk_bandwidth_hz,
        chunk_overlap_hz=cfg.chunk_overlap_hz,
        min_rows=16,
        valid_row_mask=valid_row_mask,
        uncalibrated_chunk_fraction=cfg.uncalibrated_chunk_fraction,
        uncalibrated_overlap_fraction=cfg.uncalibrated_overlap_fraction,
    )
    stage_timing_ms["chunk_plan_ms"] = (time.perf_counter() - t_chunk_plan_start) * 1000.0
    _emit_progress("chunk_plan", 1, 1, {"chunk_count": len(chunk_plan)})

    chunk_results: list[dict[str, Any]] = []
    chunk_detection_ms_total = 0.0
    chunk_grouping_ms_total = 0.0
    total_chunks = len(chunk_plan)
    for chunk_index, chunk in enumerate(chunk_plan, start=1):
        row_slice = slice(chunk["row_start"], chunk["row_stop"])
        chunk_valid_row_mask = valid_row_mask[row_slice]
        t_detection_start = time.perf_counter()
        detection = detect_chunk_coherent_power(
            corrected_sxx_db[row_slice, :],
            cfg,
            valid_row_mask=chunk_valid_row_mask,
        )
        detection_elapsed_ms = (time.perf_counter() - t_detection_start) * 1000.0
        chunk_detection_ms_total += detection_elapsed_ms
        _emit_progress(
            "chunk_detection",
            chunk_index,
            total_chunks,
            {"chunk_index": int(chunk["chunk_index"]), "rows": int(chunk["row_stop"] - chunk["row_start"]), "detection_ms": detection_elapsed_ms},
        )

        t_grouping_start = time.perf_counter()
        grouping = group_signal_mask_regions(
            np.asarray(detection["mask_px"], dtype=bool),
            score_map=np.asarray(detection["score_px"], dtype=np.float32),
            valid_row_mask=np.asarray(chunk_valid_row_mask, dtype=bool),
            bridge_freq_px=cfg.grouping_bridge_freq_px,
            bridge_time_px=cfg.grouping_bridge_time_px,
            min_component_size=max(int(cfg.grouping_min_component_size), int(cfg.min_component_size)),
            min_freq_span_px=cfg.grouping_min_freq_span_px,
            min_time_span_px=cfg.grouping_min_time_span_px,
            min_density=cfg.grouping_min_density,
        )
        grouping_elapsed_ms = (time.perf_counter() - t_grouping_start) * 1000.0
        chunk_grouping_ms_total += grouping_elapsed_ms
        _emit_progress(
            "chunk_grouping",
            chunk_index,
            total_chunks,
            {"chunk_index": int(chunk["chunk_index"]), "grouping_ms": grouping_elapsed_ms, "box_count": len(grouping["boxes"])},
        )
        timing_ms = dict(detection["timing_ms"])
        timing_ms["coherence_power_ms"] = float(detection_elapsed_ms)
        timing_ms["grouping_ms"] = float(grouping_elapsed_ms)
        timing_ms["total_ms"] = float(detection_elapsed_ms + grouping_elapsed_ms)
        chunk_results.append({
            **chunk,
            **detection,
            "valid_row_mask": chunk_valid_row_mask.astype(bool),
            "grouped_mask": np.asarray(grouping["grouped_mask"], dtype=bool),
            "grouped_boxes": list(grouping["boxes"]),
            "grouping": grouping,
            "timing_ms": timing_ms,
        })

    stage_timing_ms["chunk_detection_total_ms"] = float(chunk_detection_ms_total)
    stage_timing_ms["chunk_grouping_total_ms"] = float(chunk_grouping_ms_total)

    t_merge_start = time.perf_counter()
    merged = merge_chunk_results(
        corrected_sxx_db.shape,
        chunk_results,
        final_score_q=cfg.coherence_power_q,
        min_component_size=cfg.min_component_size,
        global_valid_row_mask=valid_row_mask,
        coherence_weight=cfg.coherence_weight,
        power_weight=cfg.power_weight,
        grouping_seed_score_q=cfg.grouping_seed_score_q,
        grouping_bridge_freq_px=cfg.grouping_bridge_freq_px,
        grouping_bridge_time_px=cfg.grouping_bridge_time_px,
        grouping_min_component_size=cfg.grouping_min_component_size,
        grouping_min_freq_span_px=cfg.grouping_min_freq_span_px,
        grouping_min_time_span_px=cfg.grouping_min_time_span_px,
        grouping_min_density=cfg.grouping_min_density,
        grouping_time_continuity_ratio=cfg.grouping_time_continuity_ratio,
    )
    stage_timing_ms["merge_ms"] = (time.perf_counter() - t_merge_start) * 1000.0
    _emit_progress("merge", 1, 1, {"merged_box_count": len(merged.get("merged_boxes", []))})
    t1 = time.perf_counter()
    stage_timing_ms["total_runtime_ms"] = (t1 - t0) * 1000.0
    return {
        "input_record": input_record,
        "config": cfg,
        "frontend": correction,
        "corrected_sxx_db": corrected_sxx_db,
        "ignore_sideband": ignore_info,
        "effective_ignore_sideband_hz": effective_ignore_sideband_hz,
        "frequency_axis_calibrated": calibrated_axis,
        "chunk_plan": chunk_plan,
        "chunk_results": chunk_results,
        "stage_timing_ms": stage_timing_ms,
        **merged,
        "total_runtime_ms": (t1 - t0) * 1000.0,
    }


def _display_db_window(sxx_db: np.ndarray, low_q: float = 1.0, high_q: float = 99.0):
    values = np.asarray(sxx_db, dtype=np.float32)
    return float(np.percentile(values, low_q)), float(np.percentile(values, high_q))


def _draw_signal_boxes(
    ax,
    boxes: list[dict[str, int | float]] | None,
    display_transposed: bool,
    edgecolor: str = "deepskyblue",
    linewidth: float = 1.4,
):
    if not boxes:
        return
    for box in boxes:
        freq_start = float(box["freq_start"])
        freq_stop = float(box["freq_stop"])
        time_start = float(box["time_start"])
        time_stop = float(box["time_stop"])
        if display_transposed:
            x0 = freq_start
            y0 = time_start
            width = max(freq_stop - freq_start, 1.0)
            height = max(time_stop - time_start, 1.0)
        else:
            x0 = time_start
            y0 = freq_start
            width = max(time_stop - time_start, 1.0)
            height = max(freq_stop - freq_start, 1.0)
        ax.add_patch(
            Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edgecolor,
                linewidth=linewidth,
            )
        )


def _show_debug_panel(ax, panel: np.ndarray, title: str, display_transposed: bool, cmap: str, vmin=None, vmax=None):
    display_panel = panel.T if display_transposed else panel
    ax.imshow(display_panel, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Time bin")


def _show_debug_overlay(
    ax,
    base: np.ndarray,
    overlay: np.ndarray,
    title: str,
    display_transposed: bool,
    base_vmin: float,
    base_vmax: float,
    overlay_cmap: str = "autumn",
    overlay_alpha: float = 0.5,
    boxes: list[dict[str, int | float]] | None = None,
    mask_cmap: str | None = None,
):
    if mask_cmap is not None:
        overlay_cmap = mask_cmap
    display_base = base.T if display_transposed else base
    display_overlay = overlay.T if display_transposed else overlay
    ax.imshow(display_base, aspect="auto", origin="lower", cmap="gray", vmin=base_vmin, vmax=base_vmax, interpolation="nearest")
    ax.imshow(np.where(display_overlay, 1.0, np.nan), aspect="auto", origin="lower", cmap=overlay_cmap, alpha=overlay_alpha, interpolation="nearest")
    _draw_signal_boxes(ax, boxes, display_transposed)
    ax.set_title(title)
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Time bin")


def plot_frontend_overview(pipeline_result: dict[str, Any], figsize: tuple[int, int] = (18, 10)):
    input_record = pipeline_result["input_record"]
    frontend = pipeline_result["frontend"]
    raw_sxx_db = np.asarray(input_record["sxx_db"], dtype=np.float32)
    corrected_sxx_db = np.asarray(pipeline_result["corrected_sxx_db"], dtype=np.float32)
    display_transposed = bool(input_record.get("display_transposed", input_kind_requires_display_transpose(input_record.get("input_kind"))))
    display_raw = raw_sxx_db.T if display_transposed else raw_sxx_db
    display_corrected = corrected_sxx_db.T if display_transposed else corrected_sxx_db
    raw_vmin, raw_vmax = _display_db_window(raw_sxx_db)
    corrected_vmin, corrected_vmax = _display_db_window(corrected_sxx_db)
    row_axis = np.arange(raw_sxx_db.shape[0], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes[0][0].imshow(display_raw, aspect="auto", origin="lower", cmap="viridis", vmin=raw_vmin, vmax=raw_vmax, interpolation="nearest")
    axes[0][0].set_title("Full spectrogram before correction")
    axes[0][1].imshow(display_corrected, aspect="auto", origin="lower", cmap="viridis", vmin=corrected_vmin, vmax=corrected_vmax, interpolation="nearest")
    axes[0][1].set_title("Full spectrogram after correction")
    axes[1][0].plot(row_axis, np.asarray(frontend["row_floor_db"], dtype=np.float32), label="Row floor")
    axes[1][0].plot(row_axis, np.asarray(frontend["response_db"], dtype=np.float32), label="Smoothed response")
    axes[1][0].axhline(float(frontend["reference_db"]), color="tab:green", linestyle="--", label="Reference")
    axes[1][0].set_title("Frontend response profile")
    axes[1][0].set_xlabel("Frequency row")
    axes[1][0].set_ylabel("dB")
    axes[1][0].legend(loc="best")
    axes[1][1].plot(row_axis, np.asarray(frontend["boost_db"], dtype=np.float32), color="tab:orange")
    axes[1][1].set_title("Frontend boost profile")
    axes[1][1].set_xlabel("Frequency row")
    axes[1][1].set_ylabel("Boost (dB)")
    for row_axes in axes:
        for ax in row_axes:
            if ax not in (axes[1][0], axes[1][1]):
                ax.set_xlabel("Frequency bin")
                ax.set_ylabel("Time bin")
    return fig, axes


def plot_chunk_plan(pipeline_result: dict[str, Any], figsize: tuple[int, int] = (18, 5)):
    corrected_sxx_db = np.asarray(pipeline_result["corrected_sxx_db"], dtype=np.float32)
    display_transposed = bool(
        pipeline_result["input_record"].get(
            "display_transposed",
            input_kind_requires_display_transpose(pipeline_result["input_record"].get("input_kind")),
        )
    )
    display_corrected = corrected_sxx_db.T if display_transposed else corrected_sxx_db
    valid_row_mask = np.asarray(pipeline_result.get("valid_row_mask", np.ones(corrected_sxx_db.shape[0], dtype=bool)), dtype=bool)
    vmin, vmax = _display_db_window(corrected_sxx_db)

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    ax.imshow(display_corrected, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    for chunk in pipeline_result["chunk_plan"]:
        if display_transposed:
            ax.axvline(chunk["row_start"], color="white", alpha=0.25, linewidth=0.9)
            ax.axvline(chunk["row_stop"] - 1, color="white", alpha=0.25, linewidth=0.9)
        else:
            ax.axhline(chunk["row_start"], color="white", alpha=0.25, linewidth=0.9)
            ax.axhline(chunk["row_stop"] - 1, color="white", alpha=0.25, linewidth=0.9)
    ignored_rows = np.flatnonzero(~valid_row_mask)
    if ignored_rows.size > 0:
        low_block = ignored_rows[ignored_rows < (corrected_sxx_db.shape[0] // 2)]
        high_block = ignored_rows[ignored_rows >= (corrected_sxx_db.shape[0] // 2)]
        if low_block.size > 0:
            if display_transposed:
                ax.axvspan(low_block[0], low_block[-1], color="black", alpha=0.12)
            else:
                ax.axhspan(low_block[0], low_block[-1], color="black", alpha=0.12)
        if high_block.size > 0:
            if display_transposed:
                ax.axvspan(high_block[0], high_block[-1], color="black", alpha=0.12)
            else:
                ax.axhspan(high_block[0], high_block[-1], color="black", alpha=0.12)
    ax.set_title("Corrected spectrogram with subsection boundaries")
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Time bin")
    return fig, ax


def plot_merged_detection(pipeline_result: dict[str, Any], figsize: tuple[int, int] = (20, 6)):
    corrected_sxx_db = np.asarray(pipeline_result["corrected_sxx_db"], dtype=np.float32)
    merged_score = np.asarray(pipeline_result["merged_score"], dtype=np.float32)
    merged_mask = np.asarray(pipeline_result["merged_mask"], dtype=bool)
    merged_support = np.asarray(pipeline_result.get("merged_support", np.ones_like(merged_mask, dtype=bool)), dtype=bool)
    merged_boxes = list(pipeline_result.get("merged_boxes", []))
    valid_row_mask = np.asarray(pipeline_result.get("valid_row_mask", np.ones(corrected_sxx_db.shape[0], dtype=bool)), dtype=bool)
    display_transposed = bool(
        pipeline_result["input_record"].get(
            "display_transposed",
            input_kind_requires_display_transpose(pipeline_result["input_record"].get("input_kind")),
        )
    )
    display_corrected = corrected_sxx_db.T if display_transposed else corrected_sxx_db
    display_score = merged_score.T if display_transposed else merged_score
    display_mask = merged_mask.T if display_transposed else merged_mask
    display_support = merged_support.T if display_transposed else merged_support
    vmin, vmax = _display_db_window(corrected_sxx_db)
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    axes[0].imshow(display_corrected, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0].set_title("Corrected wideband spectrogram")
    axes[1].imshow(display_score, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[1].imshow(np.where(display_support, 1.0, np.nan), aspect="auto", origin="lower", cmap="winter", alpha=0.18, interpolation="nearest")
    axes[1].set_title("Merged coherence-power score + support")
    axes[2].imshow(display_corrected, aspect="auto", origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[2].imshow(np.where(display_mask, 1.0, np.nan), aspect="auto", origin="lower", cmap="autumn", alpha=0.55, interpolation="nearest")
    axes[2].set_title(f"Grouped detection overlay ({len(merged_boxes)} boxes)")
    _draw_signal_boxes(axes[0], merged_boxes, display_transposed)
    _draw_signal_boxes(axes[1], merged_boxes, display_transposed)
    _draw_signal_boxes(axes[2], merged_boxes, display_transposed)
    ignored_rows = np.flatnonzero(~valid_row_mask)
    if ignored_rows.size > 0:
        low_block = ignored_rows[ignored_rows < (corrected_sxx_db.shape[0] // 2)]
        high_block = ignored_rows[ignored_rows >= (corrected_sxx_db.shape[0] // 2)]
        if low_block.size > 0:
            for ax in axes:
                if display_transposed:
                    ax.axvspan(low_block[0], low_block[-1], color="black", alpha=0.12)
                else:
                    ax.axhspan(low_block[0], low_block[-1], color="black", alpha=0.12)
        if high_block.size > 0:
            for ax in axes:
                if display_transposed:
                    ax.axvspan(high_block[0], high_block[-1], color="black", alpha=0.12)
                else:
                    ax.axhspan(high_block[0], high_block[-1], color="black", alpha=0.12)
    for ax in axes:
        ax.set_xlabel("Frequency bin")
        ax.set_ylabel("Time bin")
    return fig, axes


def plot_merged_debug(pipeline_result: dict[str, Any], figsize: tuple[int, int] = (24, 5)):
    corrected_sxx_db = np.asarray(pipeline_result["corrected_sxx_db"], dtype=np.float32)
    merged_coherence = np.asarray(pipeline_result["merged_coherence"], dtype=np.float32)
    merged_power = np.asarray(pipeline_result["merged_power"], dtype=np.float32)
    merged_score = np.asarray(pipeline_result["merged_score"], dtype=np.float32)
    merged_mask = np.asarray(pipeline_result["merged_mask"], dtype=bool)
    merged_support = np.asarray(pipeline_result.get("merged_support", np.zeros_like(corrected_sxx_db, dtype=bool)), dtype=bool)
    merged_boxes = list(pipeline_result.get("merged_boxes", []))
    display_transposed = bool(
        pipeline_result["input_record"].get(
            "display_transposed",
            input_kind_requires_display_transpose(pipeline_result["input_record"].get("input_kind")),
        )
    )
    vmin, vmax = _display_db_window(corrected_sxx_db)
    fig, axes = plt.subplots(1, 5, figsize=figsize, constrained_layout=True)
    _show_debug_panel(axes[0], corrected_sxx_db, "Input spectrogram", display_transposed, "viridis", vmin, vmax)
    _show_debug_panel(axes[1], merged_coherence, "Merged coherence", display_transposed, "plasma", 0.0, 1.0)
    _show_debug_panel(axes[2], merged_power, "Merged normalized power", display_transposed, "cividis", 0.0, 1.0)
    _show_debug_panel(axes[3], merged_score, "Merged coherence + power score", display_transposed, "magma", 0.0, 1.0)
    support_display = merged_support.T if display_transposed else merged_support
    axes[3].imshow(np.where(support_display, 1.0, np.nan), aspect="auto", origin="lower", cmap="winter", alpha=0.18, interpolation="nearest")
    _draw_signal_boxes(axes[3], merged_boxes, display_transposed)
    _show_debug_overlay(axes[4], corrected_sxx_db, merged_mask, "Grouped final overlay", display_transposed, vmin, vmax, boxes=merged_boxes)
    return fig, axes


def plot_subsection_debug(
    pipeline_result: dict[str, Any],
    subsection_index: int,
    figsize: tuple[int, int] = (26, 5),
):
    chunk = next(
        (candidate for candidate in pipeline_result["chunk_results"] if int(candidate["chunk_index"]) == int(subsection_index)),
        None,
    )
    if chunk is None:
        raise ValueError(f"No subsection found for index {subsection_index}")

    row_start = int(chunk["row_start"])
    row_stop = int(chunk["row_stop"])
    raw_chunk = np.asarray(pipeline_result["input_record"]["sxx_db"][row_start:row_stop, :], dtype=np.float32)
    display_transposed = bool(
        pipeline_result["input_record"].get(
            "display_transposed",
            input_kind_requires_display_transpose(pipeline_result["input_record"].get("input_kind")),
        )
    )
    raw_vmin, raw_vmax = _display_db_window(raw_chunk)
    fig, axes = plt.subplots(1, 5, figsize=figsize, constrained_layout=True)
    _show_debug_panel(axes[0], raw_chunk, f"Subsection {subsection_index} original spectrogram", display_transposed, "viridis", raw_vmin, raw_vmax)
    _show_debug_panel(axes[1], np.asarray(chunk["coherence_px"], dtype=np.float32), f"Subsection {subsection_index} coherence component", display_transposed, "plasma", 0.0, 1.0)
    _show_debug_panel(axes[2], np.asarray(chunk["power_px"], dtype=np.float32), f"Subsection {subsection_index} normalized power", display_transposed, "cividis", 0.0, 1.0)
    _show_debug_panel(axes[3], np.asarray(chunk["score_px"], dtype=np.float32), f"Subsection {subsection_index} coherence + power score", display_transposed, "magma", 0.0, 1.0)
    _show_debug_overlay(
        axes[4],
        raw_chunk,
        np.asarray(chunk["mask_px"], dtype=bool),
        f"Subsection {subsection_index} final mask overlay",
        display_transposed,
        raw_vmin,
        raw_vmax,
    )
    return fig, axes