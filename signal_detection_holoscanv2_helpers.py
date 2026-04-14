from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from matplotlib.patches import Rectangle
from PIL import Image
from scipy import ndimage, signal
from scipy import ndimage as ndi
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINO_IMAGE_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@dataclass
class WidebandChunkConfig:
    chunk_bandwidth_hz: float = 50e6
    chunk_overlap_hz: float = 10e6
    uncalibrated_chunk_fraction: float = 0.40
    uncalibrated_overlap_fraction: float = 0.20
    ignore_sideband_percent: float = 0.10
    ignore_sideband_hz: float | None = None
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
    grouping_seed_score_q: float = 0.72
    grouping_bridge_freq_px: int = 33
    grouping_bridge_time_px: int = 5
    grouping_min_component_size: int = 24
    grouping_min_freq_span_px: int = 18
    grouping_min_time_span_px: int = 2
    grouping_min_density: float = 0.06
    frontend_row_q: float = 25.0
    frontend_reference_q: float = 75.0
    frontend_smooth_sigma: float = 12.0
    frontend_max_boost_db: float = 12.0
    dino_max_height_px: int | None = None
    dino_max_width_px: int | None = None
    use_coherence_gate: bool = False
    use_texture_debug: bool = False
    coherence_max_height_px: int | None = None
    coherence_max_width_px: int | None = None
    parallel_backend: str = "serial"
    max_workers: int = 1


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
    with path.open("rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError(f"{path.name}: unsupported PGM magic {magic!r}")

        header_tokens: list[bytes] = []
        while len(header_tokens) < 3:
            line = f.readline()
            if not line:
                raise ValueError(f"{path.name}: truncated PGM header")
            line = line.strip()
            if not line or line.startswith(b"#"):
                continue
            header_tokens.extend(line.split())

        cols, rows, maxval = map(int, header_tokens[:3])
        if maxval > 255:
            raise ValueError(f"{path.name}: only 8-bit PGM supported (maxval={maxval})")

        data = f.read(rows * cols)
        if len(data) != rows * cols:
            raise ValueError(f"{path.name}: unexpected payload length {len(data)}")

    arr = np.frombuffer(data, dtype=np.uint8).reshape(rows, cols).astype(np.float32)
    return np.ascontiguousarray(arr)


def read_pgm(path: str | Path) -> np.ndarray:
    return read_pgm_raw(path)


def read_complex_tensor_npy(path: str | Path) -> np.ndarray:
    path = Path(path)
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2:
        raise ValueError(f"{path.name}: expected a 2D tensor snapshot, got shape {arr.shape}")
    if not np.iscomplexobj(arr):
        raise ValueError(f"{path.name}: expected complex64 tensor data")
    return np.ascontiguousarray(arr.astype(np.complex64, copy=False))


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
            "frequency_axis_calibrated": sample_rate_hz is not None and sample_rate_hz > 0,
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
            "frequency_axis_calibrated": sample_rate_hz is not None and sample_rate_hz > 0,
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
    cfg: WidebandChunkConfig,
    target_chunk_rows: int = 512,
    target_overlap_rows: int = 128,
) -> WidebandChunkConfig:
    freq_axis_hz = np.asarray(input_record.get("freq_axis_hz", []), dtype=np.float32).reshape(-1)
    if freq_axis_hz.size < 2:
        return cfg

    bin_hz = float(np.median(np.abs(np.diff(freq_axis_hz))))
    if not np.isfinite(bin_hz) or bin_hz <= 0.0:
        return cfg

    input_kind = input_record.get("input_kind")
    if input_kind != "tensor_npy":
        return cfg

    calibrated_axis = has_calibrated_frequency_axis(input_record)

    num_rows = int(freq_axis_hz.size)
    patch_size = 16
    target_chunk_rows = int(max(patch_size * 2, target_chunk_rows))
    target_overlap_rows = int(max(patch_size, min(target_overlap_rows, target_chunk_rows - patch_size)))
    target_chunk_rows = int(min(num_rows, max(target_chunk_rows, patch_size)))
    target_overlap_rows = int(min(target_overlap_rows, max(0, target_chunk_rows - patch_size)))

    chunk_bandwidth_hz = float(target_chunk_rows * bin_hz)
    chunk_overlap_hz = float(target_overlap_rows * bin_hz)

    return replace(
        cfg,
        chunk_bandwidth_hz=chunk_bandwidth_hz,
        chunk_overlap_hz=chunk_overlap_hz,
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
            return [
                {
                    "chunk_index": 0,
                    "row_start": int(valid_idx[0]),
                    "row_stop": int(valid_idx[-1]) + 1,
                    "freq_start_hz": float(freq_axis_hz[valid_idx[0]]),
                    "freq_stop_hz": float(freq_axis_hz[valid_idx[-1]]),
                }
            ]
        overlap_rows = int(np.clip(round(chunk_rows * overlap_fraction), 0, chunk_rows - 1))
        step_rows = max(1, chunk_rows - overlap_rows)
        chunks: list[dict[str, Any]] = []
        chunk_index = 0
        start_pos = 0
        while start_pos < valid_count:
            stop_pos = min(start_pos + chunk_rows, valid_count)
            in_chunk = valid_idx[start_pos:stop_pos]
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


def _resize_dino_image_if_needed(
    img_rgb: Image.Image,
    patch_size: int,
    max_height_px: int | None = None,
    max_width_px: int | None = None,
) -> Image.Image:
    width, height = img_rgb.size
    scale = 1.0
    if max_width_px is not None and width > int(max_width_px):
        scale = min(scale, float(max_width_px) / float(width))
    if max_height_px is not None and height > int(max_height_px):
        scale = min(scale, float(max_height_px) / float(height))
    if scale < 1.0:
        new_width = max(int(patch_size * 2), int(round(width * scale)))
        new_height = max(int(patch_size * 2), int(round(height * scale)))
        img_rgb = img_rgb.resize((new_width, new_height), Image.BILINEAR)
    return _prep_dino_image(img_rgb, patch_size)


def _extract_dino_features_from_rgb(
    img_rgb: Image.Image,
    model,
    patch_size: int,
    device: str,
    max_height_px: int | None = None,
    max_width_px: int | None = None,
):
    img_rgb = _resize_dino_image_if_needed(
        img_rgb,
        patch_size,
        max_height_px=max_height_px,
        max_width_px=max_width_px,
    )
    x = DINO_IMAGE_TRANSFORM(img_rgb).unsqueeze(0).to(device)
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


def dino_seed_patch_map(sxx_db_local: np.ndarray, patch_h: int, patch_w: int, valid_row_mask: np.ndarray | None = None) -> np.ndarray:
    local_support = _local_relative_power_support_map(sxx_db_local, valid_row_mask=valid_row_mask, floor_q=30.0)
    persistence_px = ndimage.uniform_filter(local_support, size=(1, 7), mode="nearest")
    local_contrast_px = np.clip(local_support - ndimage.uniform_filter(local_support, size=(5, 5), mode="nearest"), 0.0, None)
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

    structure = np.ones(
        (
            max(1, int(bridge_freq_px)),
            max(1, int(bridge_time_px)),
        ),
        dtype=bool,
    )
    bridged_mask = ndimage.binary_closing(working_mask, structure=structure)
    bridged_mask = ndimage.binary_fill_holes(bridged_mask)

    component_labels, n_components = ndimage.label(bridged_mask)
    grouped_mask = np.zeros_like(working_mask, dtype=bool)
    boxes: list[dict[str, int | float]] = []
    component_rows: list[dict[str, int | float]] = []

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
        freq_start = int(row_coords.min())
        freq_stop = int(row_coords.max()) + 1
        time_start = int(col_coords.min())
        time_stop = int(col_coords.max()) + 1
        freq_span = int(freq_stop - freq_start)
        time_span = int(time_stop - time_start)
        bbox_area = max(freq_span * time_span, 1)
        filled_area = int(np.count_nonzero(component_mask))
        density = float(filled_area / bbox_area)

        score_peak = float(np.max(score_map_arr[component_mask])) if score_map_arr is not None and np.any(component_mask) else 0.0
        score_mean = float(np.mean(score_map_arr[component_mask])) if score_map_arr is not None and np.any(component_mask) else 0.0
        keep_component = (
            filled_area >= int(min_component_size)
            and freq_span >= int(min_freq_span_px)
            and time_span >= int(min_time_span_px)
            and (density >= float(min_density) or score_peak >= peak_score_floor)
        )

        component_rows.append({
            "component_id": int(component_id),
            "freq_start": freq_start,
            "freq_stop": freq_stop,
            "time_start": time_start,
            "time_stop": time_stop,
            "freq_span": freq_span,
            "time_span": time_span,
            "filled_area": filled_area,
            "bbox_area": int(bbox_area),
            "density": density,
            "score_mean": score_mean,
            "score_peak": score_peak,
            "kept": bool(keep_component),
        })

        if not keep_component:
            continue

        grouped_mask[freq_start:freq_stop, time_start:time_stop] = True
        boxes.append({
            "freq_start": freq_start,
            "freq_stop": freq_stop,
            "time_start": time_start,
            "time_stop": time_stop,
            "freq_span": freq_span,
            "time_span": time_span,
            "filled_area": filled_area,
            "density": density,
            "score_mean": score_mean,
            "score_peak": score_peak,
        })

    if valid_row_mask is not None:
        grouped_mask[~valid_row_mask, :] = False

    return {
        "seed_mask": working_mask.astype(bool),
        "bridged_mask": bridged_mask.astype(bool),
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
    )
    region_groups["seed_threshold"] = float(seed_threshold)
    region_groups["seed_score_q"] = float(seed_score_q)
    region_groups["raw_mask_fraction"] = float(np.mean(merged_mask))
    region_groups["grouped_mask_fraction"] = float(np.mean(np.asarray(region_groups["grouped_mask"], dtype=bool)))
    return region_groups


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
                linestyle="-",
            )
        )


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
    dino_max_height_px: int | None = None,
    dino_max_width_px: int | None = None,
    valid_row_mask: np.ndarray | None = None,
):
    img_rgb, input_debug = build_signal_agnostic_dino_input(sxx_db_local, db_min=db_min, db_max=db_max)
    feat_local, grid_h, grid_w, img_used = _extract_dino_features_from_rgb(
        img_rgb,
        model,
        patch_size,
        device,
        max_height_px=dino_max_height_px,
        max_width_px=dino_max_width_px,
    )
    seed_patch = dino_seed_patch_map(sxx_db_local, grid_h, grid_w, valid_row_mask=valid_row_mask)
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
    score_map = _normalize_map01_local(0.90 * support_map + 0.10 * (support_map * seed_norm))
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
    background = ndimage.uniform_filter(
        x_db,
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


def multi_scale_structure_tensor_gate(
    sxx_db_local: np.ndarray,
    patch_h: int,
    patch_w: int,
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
    for grad_sigma in tuple(float(v) for v in scales):
        coherence, energy = _structure_tensor_components(residual_n, grad_sigma=grad_sigma, integ_sigma=max(1.0, 1.8 * grad_sigma))
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
        "coherence_patch": patch_mean_map(coherence_px, patch_h, patch_w),
        "energy_patch": patch_mean_map(energy_px, patch_h, patch_w),
        "gate_patch": patch_mean_map(gate_px, patch_h, patch_w),
        "residual_patch": patch_mean_map(residual_n, patch_h, patch_w),
        "work_shape_px": (int(work_sxx_db.shape[0]), int(work_sxx_db.shape[1])),
    }


def _soft_gate_dino_score(dino_score_map: np.ndarray, gate_patch: np.ndarray, floor: float = 0.25):
    dino_n = _normalize_map01_local(dino_score_map, 5.0, 95.0)
    gate_n = _normalize_map01_local(gate_patch, 5.0, 95.0)
    score = dino_n * (float(floor) + (1.0 - float(floor)) * gate_n)
    return _normalize_map01_local(score, 5.0, 95.0).astype(np.float32), dino_n, gate_n


def apply_coherence_gate_to_dino_result(
    dino_group: dict[str, Any],
    sxx_db_local: np.ndarray,
    gate_floor: float = 0.25,
    min_component_size: int = 3,
    coherence_max_height_px: int | None = None,
    coherence_max_width_px: int | None = None,
):
    patch_h, patch_w = tuple(dino_group["shape"])
    coherence = multi_scale_structure_tensor_gate(
        sxx_db_local,
        patch_h,
        patch_w,
        max_height_px=coherence_max_height_px,
        max_width_px=coherence_max_width_px,
    )
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
    out["coherence_work_shape_px"] = tuple(int(v) for v in coherence["work_shape_px"])
    return out


def apply_coherence_gate_to_dino_result_from_maps(
    dino_group: dict[str, Any],
    coherence: dict[str, Any],
    gate_floor: float = 0.25,
    min_component_size: int = 3,
):
    raw_score = np.asarray(dino_group["score"], dtype=np.float32)
    gated_score, _, gate_patch = _soft_gate_dino_score(raw_score, coherence["gate_patch"], floor=gate_floor)
    thr = float(np.quantile(gated_score, 0.60))
    mask = _smooth_binary_label_map((gated_score >= thr).astype(np.uint8), iters=1, min_component_size=min_component_size)
    out = dict(dino_group)
    out["score"] = gated_score.astype(np.float32)
    out["mask"] = mask.astype(np.uint8)
    out["threshold"] = thr
    out["coherence_gate_px"] = np.asarray(coherence["gate_px"], dtype=np.float32)
    out["coherence_energy_px"] = np.asarray(coherence["energy_px"], dtype=np.float32)
    out["coherence_residual_px"] = np.asarray(coherence["residual_n"], dtype=np.float32)
    out["coherence_gate_patch"] = gate_patch.astype(np.float32)
    out["coherence_energy_patch"] = np.asarray(coherence["energy_patch"], dtype=np.float32)
    out["coherence_residual_patch"] = np.asarray(coherence["residual_patch"], dtype=np.float32)
    out["coherence_work_shape_px"] = tuple(int(v) for v in coherence["work_shape_px"])
    return out


def bypass_coherence_gate_for_dino_result(dino_group: dict[str, Any], sxx_db_local: np.ndarray) -> dict[str, Any]:
    sxx_db_local = np.asarray(sxx_db_local, dtype=np.float32)
    patch_h, patch_w = tuple(dino_group["shape"])
    out = dict(dino_group)
    out["score"] = np.asarray(dino_group["score"], dtype=np.float32)
    out["mask"] = np.asarray(dino_group["mask"], dtype=np.uint8)
    out["coherence_gate_px"] = np.zeros_like(sxx_db_local, dtype=np.float32)
    out["coherence_energy_px"] = np.zeros_like(sxx_db_local, dtype=np.float32)
    out["coherence_residual_px"] = np.zeros_like(sxx_db_local, dtype=np.float32)
    out["coherence_gate_patch"] = np.zeros((patch_h, patch_w), dtype=np.float32)
    out["coherence_energy_patch"] = np.zeros((patch_h, patch_w), dtype=np.float32)
    out["coherence_residual_patch"] = np.zeros((patch_h, patch_w), dtype=np.float32)
    out["coherence_work_shape_px"] = tuple(int(v) for v in sxx_db_local.shape)
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


def power_prior_patch_map(
    sxx_db_local: np.ndarray,
    patch_h: int,
    patch_w: int,
    valid_row_mask: np.ndarray | None = None,
) -> np.ndarray:
    local_support = _local_relative_power_support_map(sxx_db_local, valid_row_mask=valid_row_mask, floor_q=30.0)
    return patch_mean_map(local_support, patch_h, patch_w)


def _resize_patch_map_to_pixels(map_patch: np.ndarray, rows: int, cols: int, resample: int) -> np.ndarray:
    img = Image.fromarray(np.asarray(map_patch, dtype=np.float32), mode="F")
    out = img.resize((int(cols), int(rows)), resample=resample)
    return np.asarray(out, dtype=np.float32)


def _resize_patch_map_to_pixels_time_preserving(
    map_patch: np.ndarray,
    rows: int,
    cols: int,
    row_resample: int = Image.BILINEAR,
    col_resample: int = Image.NEAREST,
) -> np.ndarray:
    img = Image.fromarray(np.asarray(map_patch, dtype=np.float32), mode="F")
    patch_rows, patch_cols = np.asarray(map_patch, dtype=np.float32).shape
    if int(cols) != int(patch_cols):
        img = img.resize((int(cols), int(patch_rows)), resample=col_resample)
    if int(rows) != int(patch_rows):
        img = img.resize((int(cols), int(rows)), resample=row_resample)
    return np.asarray(img, dtype=np.float32)


def _resize_patch_mask_to_pixels(mask_patch: np.ndarray, rows: int, cols: int) -> np.ndarray:
    resized = _resize_patch_map_to_pixels(mask_patch.astype(np.float32), rows, cols, Image.NEAREST)
    return resized >= 0.5


def _row_profile_to_patch_map(row_profile: np.ndarray, patch_h: int, patch_w: int, cols: int) -> np.ndarray:
    row_profile = np.asarray(row_profile, dtype=np.float32).reshape(-1)
    tiled = np.repeat(row_profile[:, None], int(cols), axis=1)
    return patch_mean_map(tiled, patch_h, patch_w)


def _row_profile_to_pixel_map(row_profile: np.ndarray, cols: int) -> np.ndarray:
    row_profile = np.asarray(row_profile, dtype=np.float32).reshape(-1)
    return np.repeat(row_profile[:, None], int(cols), axis=1).astype(np.float32)


def _row_bool_mask_to_pixel_map(row_mask: np.ndarray, cols: int) -> np.ndarray:
    row_mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    return np.repeat(row_mask[:, None], int(cols), axis=1)


def _calibrate_score_map_local(
    score_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
    low_q: float = 60.0,
    high_q: float = 99.0,
) -> tuple[np.ndarray, dict[str, float]]:
    score_map = np.asarray(score_map, dtype=np.float32)
    if valid_mask is None:
        vals = score_map.reshape(-1)
    else:
        vals = score_map[np.asarray(valid_mask, dtype=bool)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(score_map, dtype=np.float32), {
            "low_q": float(low_q),
            "high_q": float(high_q),
            "low_value": 0.0,
            "high_value": 1.0,
        }
    lo = float(np.percentile(vals, low_q))
    hi = float(np.percentile(vals, high_q))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = 1.0
    if hi <= lo + 1e-6:
        hi = lo + 1e-6
    calibrated = np.clip((score_map - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return calibrated, {
        "low_q": float(low_q),
        "high_q": float(high_q),
        "low_value": lo,
        "high_value": hi,
    }


def _chunk_sparse_evidence_weight(
    strict_raw_patch: np.ndarray,
    support_mask_patch: np.ndarray | None = None,
) -> dict[str, float]:
    strict_raw_patch = np.asarray(strict_raw_patch, dtype=np.float32)
    if support_mask_patch is None:
        active_mask = np.ones_like(strict_raw_patch, dtype=bool)
    else:
        active_mask = np.asarray(support_mask_patch, dtype=bool)
        if active_mask.shape != strict_raw_patch.shape:
            raise ValueError("support_mask_patch shape must match strict_raw_patch shape")
        if not np.any(active_mask):
            active_mask = np.ones_like(strict_raw_patch, dtype=bool)
    vals = strict_raw_patch[active_mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "weight": 0.0,
            "q50": 0.0,
            "q85": 0.0,
            "q97": 0.0,
            "top_mean": 0.0,
            "support_fraction": 0.0,
        }
    q50 = float(np.quantile(vals, 0.50))
    q85 = float(np.quantile(vals, 0.85))
    q97 = float(np.quantile(vals, 0.97))
    top_k = max(4, int(np.ceil(0.02 * vals.size)))
    top_mean = float(np.mean(np.partition(vals, vals.size - top_k)[-top_k:]))
    peak_excess = max(q97 - q85, 0.0)
    tail_excess = max(top_mean - q85, 0.0)
    support_fraction = float(np.mean(active_mask))
    support_term = float(np.clip(np.sqrt(support_fraction / 0.03), 0.0, 1.0))
    evidence = float(np.clip((0.60 * peak_excess + 0.40 * tail_excess) / 0.12, 0.0, 1.0))
    weight = float(np.clip(evidence * (0.25 + 0.75 * support_term), 0.05, 1.0))
    return {
        "weight": weight,
        "q50": q50,
        "q85": q85,
        "q97": q97,
        "top_mean": top_mean,
        "support_fraction": support_fraction,
    }


def run_chunk_detector(
    sxx_db_chunk: np.ndarray,
    model,
    patch_size: int,
    device: str,
    cfg: WidebandChunkConfig,
    valid_row_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    effective_dino_max_height_px = cfg.dino_max_height_px
    effective_dino_max_width_px = cfg.dino_max_width_px
    if str(device).lower() == "cpu":
        if effective_dino_max_height_px is None:
            effective_dino_max_height_px = 512
        if effective_dino_max_width_px is None:
            effective_dino_max_width_px = 320
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
        dino_max_height_px=effective_dino_max_height_px,
        dino_max_width_px=effective_dino_max_width_px,
        valid_row_mask=valid_row_mask,
    )
    t1 = time.perf_counter()
    patch_h, patch_w = tuple(dino_group["shape"])
    if cfg.use_coherence_gate:
        raw_coherence = multi_scale_structure_tensor_gate(
            sxx_db_chunk,
            patch_h,
            patch_w,
            max_height_px=cfg.coherence_max_height_px,
            max_width_px=cfg.coherence_max_width_px,
        )
        dino_gated = apply_coherence_gate_to_dino_result_from_maps(
            dino_group,
            raw_coherence,
            gate_floor=cfg.dino_coherence_gate_floor,
            min_component_size=max(3, cfg.min_component_size // 2),
        )
    else:
        dino_gated = bypass_coherence_gate_for_dino_result(dino_group, sxx_db_chunk)
    t2 = time.perf_counter()
    dino_score_px_raw = _resize_patch_map_to_pixels_time_preserving(
        dino_gated["score"],
        sxx_db_chunk.shape[0],
        sxx_db_chunk.shape[1],
        row_resample=Image.BILINEAR,
        col_resample=Image.NEAREST,
    )
    dino_score_px = np.clip(dino_score_px_raw, 0.0, 1.0).astype(np.float32)
    power_px = _normalize_map01_local(
        _local_relative_power_support_map(sxx_db_chunk, valid_row_mask=valid_row_mask, floor_q=30.0),
        5.0,
        95.0,
    )
    power_patch = _normalize_map01_local(
        patch_mean_map(power_px, patch_h, patch_w),
        5.0,
        95.0,
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
    fused_px = _normalize_map01_local(
        dino_score_px * ((1.0 - cfg.power_fusion_gain) + cfg.power_fusion_gain * power_px),
        5.0,
        95.0,
    )
    strict_px_raw = (
        fused_px * (float(cfg.merge_companion_floor) + (1.0 - float(cfg.merge_companion_floor)) * burst_gate["companion_px"])
    ).astype(np.float32)
    strict_px = _normalize_map01_local(
        strict_px_raw,
        5.0,
        95.0,
    )
    support_px_raw = _normalize_map01_local(0.70 * dino_score_px + 0.30 * burst_gate["companion_px"], 5.0, 95.0)
    support_patch = patch_mean_map(support_px_raw, patch_h, patch_w)
    support_thr = float(np.quantile(support_px_raw, float(np.clip(cfg.merge_support_q, 0.50, 0.95))))
    support_mask_px = _smooth_binary_label_map((support_px_raw >= support_thr).astype(np.uint8), iters=1, min_component_size=max(3, cfg.min_component_size // 2))
    support_mask = patch_mean_map(support_mask_px.astype(np.float32), patch_h, patch_w) >= 0.5
    fused_thr = float(np.quantile(strict_px, cfg.final_score_q))
    fused_mask = _smooth_binary_label_map((strict_px >= fused_thr).astype(np.uint8), iters=1, min_component_size=cfg.min_component_size)
    fused_patch = patch_mean_map(fused_px, patch_h, patch_w)
    strict_patch_raw = patch_mean_map(strict_px_raw, patch_h, patch_w)
    strict_patch = patch_mean_map(strict_px, patch_h, patch_w)
    sparse_evidence = _chunk_sparse_evidence_weight(strict_patch_raw, support_mask_patch=support_mask.astype(bool))
    if valid_row_mask is None:
        valid_row_mask = np.ones(sxx_db_chunk.shape[0], dtype=bool)
    else:
        valid_row_mask = np.asarray(valid_row_mask, dtype=bool).reshape(-1)
        if valid_row_mask.shape[0] != sxx_db_chunk.shape[0]:
            raise ValueError("valid_row_mask length must match the number of chunk rows")
    chunk_evidence_weight = float(sparse_evidence["weight"])
    strict_patch_weighted = np.clip(strict_patch * chunk_evidence_weight, 0.0, 1.0).astype(np.float32)
    support_patch_weighted = np.clip(support_patch * chunk_evidence_weight, 0.0, 1.0).astype(np.float32)
    companion_patch_weighted = np.clip(burst_gate["companion_patch"] * chunk_evidence_weight, 0.0, 1.0).astype(np.float32)
    strict_px_weighted = np.clip(strict_px * chunk_evidence_weight, 0.0, 1.0).astype(np.float32)
    support_px_weighted = np.clip(support_px_raw * chunk_evidence_weight, 0.0, 1.0).astype(np.float32)
    companion_px_weighted = np.clip(burst_gate["companion_px"] * chunk_evidence_weight, 0.0, 1.0).astype(np.float32)
    score_px_raw = strict_px
    score_px = np.clip(score_px_raw, 0.0, 1.0).astype(np.float32)
    support_px = support_px_weighted >= 0.50
    score_valid_mask = _row_bool_mask_to_pixel_map(valid_row_mask, sxx_db_chunk.shape[1])
    score_px_calibrated, score_calibration = _calibrate_score_map_local(score_px, valid_mask=score_valid_mask)
    score_px_contribution = np.clip(score_px_calibrated * chunk_evidence_weight, 0.0, 1.0).astype(np.float32)
    mask_px = fused_mask.astype(bool)
    companion_px = companion_px_weighted.astype(np.float32)
    t3 = time.perf_counter()
    return {
        "dino_group": dino_group,
        "dino_gated": dino_gated,
        "burst_gate": burst_gate,
        "dino_score_px": dino_score_px.astype(np.float32),
        "power_px": power_px.astype(np.float32),
        "fused_px": fused_px.astype(np.float32),
        "strict_px_raw": strict_px_raw.astype(np.float32),
        "strict_px": strict_px.astype(np.float32),
        "power_patch": power_patch.astype(np.float32),
        "fused_patch": fused_patch.astype(np.float32),
        "strict_patch_raw": strict_patch_raw.astype(np.float32),
        "strict_patch": strict_patch.astype(np.float32),
        "strict_patch_weighted": strict_patch_weighted.astype(np.float32),
        "support_patch": support_patch.astype(np.float32),
        "support_patch_weighted": support_patch_weighted.astype(np.float32),
        "companion_patch_weighted": companion_patch_weighted.astype(np.float32),
        "chunk_sparse_evidence": sparse_evidence,
        "support_threshold": support_thr,
        "support_mask_patch": support_mask.astype(np.uint8),
        "fused_mask_patch": fused_mask.astype(np.uint8),
        "fused_threshold": fused_thr,
        "score_px": score_px.astype(np.float32),
        "score_px_raw": score_px_raw.astype(np.float32),
        "score_px_calibrated": score_px_calibrated.astype(np.float32),
        "score_px_contribution": score_px_contribution.astype(np.float32),
        "score_calibration": score_calibration,
        "mask_px": mask_px.astype(bool),
        "support_px": support_px.astype(bool),
        "companion_px": companion_px.astype(np.float32),
        "timing_ms": {
            "dino_group_ms": (t1 - t0) * 1000.0,
            "dino_coherence_ms": (t2 - t1) * 1000.0,
            "fuse_ms": (t3 - t2) * 1000.0,
            "total_ms": (t3 - t0) * 1000.0,
        },
        "dino_input_size_px": tuple(int(v) for v in dino_group["input_img"].size[::-1]),
    }


def _chunk_blend_weights(length: int) -> np.ndarray:
    if length <= 2:
        return np.ones(length, dtype=np.float32)
    base = np.hanning(length).astype(np.float32)
    if float(np.max(base)) <= 0.0:
        return np.ones(length, dtype=np.float32)
    base = base / float(np.max(base))
    return (0.2 + 0.8 * base).astype(np.float32)


def merge_chunk_results(
    global_shape: tuple[int, int],
    chunk_results: list[dict[str, Any]],
    final_score_q: float = 0.90,
    min_component_size: int = 6,
    global_valid_row_mask: np.ndarray | None = None,
    grouping_seed_score_q: float = 0.72,
    grouping_bridge_freq_px: int = 33,
    grouping_bridge_time_px: int = 5,
    grouping_min_component_size: int = 24,
    grouping_min_freq_span_px: int = 18,
    grouping_min_time_span_px: int = 2,
    grouping_min_density: float = 0.06,
):
    merged_base_score = np.zeros(global_shape, dtype=np.float32)
    merged_support = np.zeros(global_shape, dtype=bool)
    merged_companion_sum = np.zeros(global_shape, dtype=np.float32)
    merged_overlap_weight = np.zeros(global_shape[0], dtype=np.float32)
    for chunk in chunk_results:
        row_start = int(chunk["row_start"])
        row_stop = int(chunk["row_stop"])
        chunk_weights = _chunk_blend_weights(row_stop - row_start)[:, None]
        weighted_score = np.asarray(chunk.get("score_px_contribution", chunk.get("score_px_calibrated", chunk["score_px"])), dtype=np.float32) * chunk_weights
        weighted_companion = chunk["companion_px"] * chunk_weights
        merged_base_score[row_start:row_stop, :] = np.maximum(merged_base_score[row_start:row_stop, :], weighted_score)
        merged_companion_sum[row_start:row_stop, :] += weighted_companion
        merged_overlap_weight[row_start:row_stop] += chunk_weights[:, 0]
        merged_support[row_start:row_stop, :] |= np.asarray(chunk["support_px"], dtype=bool)
    overlap_norm = np.maximum(merged_overlap_weight[:, None], 1e-6)
    merged_companion = np.divide(
        merged_companion_sum,
        overlap_norm,
        out=np.zeros_like(merged_companion_sum),
        where=overlap_norm > 0.0,
    )
    merged_score = _normalize_map01_local(
        merged_base_score
        * (0.30 + 0.70 * _normalize_map01_local(merged_companion, 5.0, 95.0)),
        5.0,
        95.0,
    )
    valid_row_mask = np.ones(global_shape[0], dtype=bool)
    if global_valid_row_mask is not None:
        valid_row_mask = np.asarray(global_valid_row_mask, dtype=bool).reshape(-1)
        if valid_row_mask.shape[0] != global_shape[0]:
            raise ValueError("global_valid_row_mask length must match global_shape rows")
    valid_scores = merged_score[np.logical_and(valid_row_mask[:, None], merged_support)]
    threshold = _robust_high_quantile_threshold(valid_scores, final_score_q) if valid_scores.size else 1.0
    raw_merged_mask = _smooth_binary_label_map(
        np.logical_and(merged_score >= threshold, merged_support).astype(np.uint8),
        iters=1,
        min_component_size=min_component_size,
    )
    raw_merged_mask[~valid_row_mask, :] = 0
    merged_score[~valid_row_mask, :] = 0.0
    merged_support[~valid_row_mask, :] = False
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
    )
    merged_mask = np.asarray(merged_region_groups["grouped_mask"], dtype=bool)
    return {
        "merged_score": merged_score.astype(np.float32),
        "merged_mask": merged_mask.astype(bool),
        "raw_merged_mask": np.asarray(raw_merged_mask, dtype=bool),
        "merged_threshold": threshold,
        "valid_row_mask": valid_row_mask.astype(bool),
        "merged_support": merged_support.astype(bool),
        "merged_base_score": merged_base_score.astype(np.float32),
        "merged_companion": merged_companion.astype(np.float32),
        "merged_boxes": list(merged_region_groups["boxes"]),
        "merged_region_groups": merged_region_groups,
    }


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return float("nan")
    a = a[finite]
    b = b[finite]
    if a.size < 2 or float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _temporal_roughness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 2:
        return 0.0
    diffs = np.diff(x, axis=1)
    scale = float(np.std(x))
    if scale < 1e-6:
        return 0.0
    return float(np.mean(np.abs(diffs)) / scale)


def _spatial_roughness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 2:
        return 0.0
    diffs = np.diff(x, axis=0)
    scale = float(np.std(x))
    if scale < 1e-6:
        return 0.0
    return float(np.mean(np.abs(diffs)) / scale)


def analyze_subsection_boundary_artifacts(
    pipeline_result: dict[str, Any],
    figsize: tuple[int, int] = (20, 10),
) -> dict[str, Any]:
    corrected_sxx_db = np.asarray(pipeline_result["corrected_sxx_db"], dtype=np.float32)
    rows, cols = corrected_sxx_db.shape
    frontend = pipeline_result["frontend"]
    chunk_results = pipeline_result["chunk_results"]
    merged_base_score = np.asarray(pipeline_result.get("merged_base_score", np.zeros_like(corrected_sxx_db)), dtype=np.float32)
    merged_companion = np.asarray(pipeline_result.get("merged_companion", np.zeros_like(corrected_sxx_db)), dtype=np.float32)
    row_chunk_count = np.zeros(rows, dtype=np.float32)
    row_blend_peak = np.zeros(rows, dtype=np.float32)
    row_blend_sum = np.zeros(rows, dtype=np.float32)
    boundary_mask = np.zeros(rows, dtype=bool)
    boundary_rows: list[int] = []
    for chunk in chunk_results:
        row_start = int(chunk["row_start"])
        row_stop = int(chunk["row_stop"])
        weights = _chunk_blend_weights(row_stop - row_start)
        row_chunk_count[row_start:row_stop] += 1.0
        row_blend_peak[row_start:row_stop] = np.maximum(row_blend_peak[row_start:row_stop], weights)
        row_blend_sum[row_start:row_stop] += weights
        boundary_rows.extend([row_start, max(row_start, row_stop - 1)])
        boundary_mask[row_start] = True
        boundary_mask[max(row_start, row_stop - 1)] = True

    row_merged_companion = np.mean(merged_companion, axis=1).astype(np.float32)
    row_merged_base = np.mean(merged_base_score, axis=1).astype(np.float32)
    row_boundary_energy = np.abs(np.diff(row_merged_base, prepend=row_merged_base[:1])).astype(np.float32)
    blend_boundary_energy = np.abs(np.diff(row_blend_sum, prepend=row_blend_sum[:1])).astype(np.float32)
    boundary_rows = sorted(set(boundary_rows))

    summary = {
        "corr(merged_base, blend_sum)": _safe_corrcoef(row_merged_base, row_blend_sum),
        "corr(merged_companion, blend_sum)": _safe_corrcoef(row_merged_companion, row_blend_sum),
        "corr(merged_base, boost_db)": _safe_corrcoef(row_merged_base, np.asarray(frontend["boost_db"], dtype=np.float32)),
        "boundary_energy_mean": float(np.mean(row_boundary_energy[boundary_mask])) if np.any(boundary_mask) else 0.0,
        "non_boundary_energy_mean": float(np.mean(row_boundary_energy[~boundary_mask])) if np.any(~boundary_mask) else 0.0,
        "mean_chunks_covering_row": float(np.mean(row_chunk_count)),
        "max_chunks_covering_row": int(np.max(row_chunk_count)) if row_chunk_count.size else 0,
    }

    profile_rows = []
    for row_idx in boundary_rows:
        if row_idx < 0 or row_idx >= rows:
            continue
        profile_rows.append({
            "row": int(row_idx),
            "boost_db": float(np.asarray(frontend["boost_db"], dtype=np.float32)[row_idx]),
            "merged_base": float(row_merged_base[row_idx]),
            "blend_sum": float(row_blend_sum[row_idx]),
            "blend_peak": float(row_blend_peak[row_idx]),
            "chunk_count": int(row_chunk_count[row_idx]),
            "boundary_energy": float(row_boundary_energy[row_idx]),
        })

    display_transposed = bool(
        pipeline_result["input_record"].get(
            "display_transposed",
            input_kind_requires_display_transpose(pipeline_result["input_record"].get("input_kind")),
        )
    )
    display_base = merged_base_score.T if display_transposed else merged_base_score

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes[0][0].imshow(display_base, aspect="auto", origin="lower", cmap="inferno", vmin=0.0, vmax=1.0)
    axes[0][0].set_title("Merged base striping")

    x = np.arange(rows, dtype=np.int32)
    axes[0][1].plot(x, np.asarray(frontend["boost_db"], dtype=np.float32), label="boost_db")
    axes[0][1].plot(x, row_merged_base, label="merged_base")
    axes[0][1].plot(x, _normalize_vector01(row_blend_sum), label="normalized_blend_sum")
    axes[0][1].plot(x, _normalize_vector01(row_chunk_count), label="normalized_chunk_count")
    for row_idx in boundary_rows:
        axes[0][1].axvline(row_idx, color="black", alpha=0.08, linewidth=0.8)
    axes[0][1].set_title("Row profiles vs subsection boundaries")
    axes[0][1].legend(loc="best")

    axes[1][0].plot(x, row_merged_base, label="merged_base_mean")
    axes[1][0].plot(x, row_merged_companion, label="merged_companion_mean")
    axes[1][0].plot(x, row_boundary_energy, label="merged_base_boundary_energy")
    axes[1][0].set_title("Boundary response in merged maps")
    axes[1][0].legend(loc="best")

    axes[1][1].scatter(row_blend_sum, row_merged_base, s=12, alpha=0.6, label="blend_sum")
    axes[1][1].scatter(np.asarray(frontend["boost_db"], dtype=np.float32), row_merged_base, s=12, alpha=0.6, label="boost_db")
    axes[1][1].set_title("What the merged base follows")
    axes[1][1].set_xlabel("Driver value")
    axes[1][1].set_ylabel("Merged base")
    axes[1][1].legend(loc="best")

    return {
        "summary": summary,
        "profile_rows": profile_rows,
        "boundary_rows": boundary_rows,
        "figure": fig,
        "axes": axes,
    }


def analyze_temporal_smearing(
    pipeline_result: dict[str, Any],
    chunk_indices: list[int] | None = None,
    top_k: int = 4,
    figsize: tuple[int, int] = (22, 10),
) -> dict[str, Any]:
    chunk_results = pipeline_result["chunk_results"]
    corrected_sxx_db = np.asarray(pipeline_result["corrected_sxx_db"], dtype=np.float32)
    if not chunk_results:
        raise ValueError("pipeline_result contains no chunk_results")

    if chunk_indices is None:
        ranked = sorted(
            chunk_results,
            key=lambda chunk: float(np.mean(np.asarray(chunk.get("score_px_raw", chunk["score_px"]), dtype=np.float32))),
            reverse=True,
        )
        selected_chunks = ranked[: max(1, int(top_k))]
    else:
        selected = set(int(idx) for idx in chunk_indices)
        selected_chunks = [chunk for chunk in chunk_results if int(chunk["chunk_index"]) in selected]
        if not selected_chunks:
            raise ValueError("No chunk indices matched the requested subset")

    chunk_rows = []
    for chunk in selected_chunks:
        row_start = int(chunk["row_start"])
        row_stop = int(chunk["row_stop"])
        chunk_corrected = corrected_sxx_db[row_start:row_stop, :]
        rows, cols = chunk_corrected.shape
        patch_h, patch_w = tuple(np.asarray(chunk["strict_patch"]).shape)
        companion_patch = np.asarray(chunk["burst_gate"]["companion_patch"], dtype=np.float32)
        companion_nearest = _resize_patch_map_to_pixels(companion_patch, rows, cols, Image.NEAREST)
        companion_bilinear = _resize_patch_map_to_pixels(companion_patch, rows, cols, Image.BILINEAR)
        strict_nearest = _resize_patch_map_to_pixels(np.asarray(chunk["strict_patch"], dtype=np.float32), rows, cols, Image.NEAREST)
        strict_bilinear = np.asarray(chunk.get("score_px_raw", chunk["score_px"]), dtype=np.float32)
        dino_input = np.asarray(chunk["dino_group"]["input_debug"]["input_gray01"], dtype=np.float32)
        temporal_window = max(5, int(2 * (max(1, cols // 32)) + 1))
        coherence_work_cols = int(chunk["dino_gated"].get("coherence_work_shape_px", (rows, cols))[1])
        chunk_rows.append({
            "subsection_index": int(chunk["chunk_index"]),
            "rows": (row_start, row_stop),
            "time_bins": int(cols),
            "patch_grid": (int(patch_h), int(patch_w)),
            "time_bins_per_patch": float(cols / max(patch_w, 1)),
            "companion_temporal_window_px": int(temporal_window),
            "coherence_resize_ratio": float(cols / max(coherence_work_cols, 1)),
            "temporal_roughness_corrected": _temporal_roughness(chunk_corrected),
            "temporal_roughness_dino_input": _temporal_roughness(dino_input),
            "temporal_roughness_companion_nearest": _temporal_roughness(companion_nearest),
            "temporal_roughness_companion_bilinear": _temporal_roughness(companion_bilinear),
            "temporal_roughness_strict_nearest": _temporal_roughness(strict_nearest),
            "temporal_roughness_strict_bilinear": _temporal_roughness(strict_bilinear),
            "spatial_roughness_corrected": _spatial_roughness(chunk_corrected),
            "spatial_roughness_strict_bilinear": _spatial_roughness(strict_bilinear),
        })

    focus_chunk = selected_chunks[0]
    row_start = int(focus_chunk["row_start"])
    row_stop = int(focus_chunk["row_stop"])
    focus_corrected = corrected_sxx_db[row_start:row_stop, :]
    focus_rows, focus_cols = focus_corrected.shape
    focus_companion_patch = np.asarray(focus_chunk["burst_gate"]["companion_patch"], dtype=np.float32)
    focus_companion_nearest = _resize_patch_map_to_pixels(focus_companion_patch, focus_rows, focus_cols, Image.NEAREST)
    focus_companion_bilinear = _resize_patch_map_to_pixels(focus_companion_patch, focus_rows, focus_cols, Image.BILINEAR)
    focus_strict_nearest = _resize_patch_map_to_pixels(np.asarray(focus_chunk["strict_patch"], dtype=np.float32), focus_rows, focus_cols, Image.NEAREST)
    focus_strict_bilinear = np.asarray(focus_chunk.get("score_px_raw", focus_chunk["score_px"]), dtype=np.float32)
    focus_display_transposed = bool(
        pipeline_result["input_record"].get(
            "display_transposed",
            input_kind_requires_display_transpose(pipeline_result["input_record"].get("input_kind")),
        )
    )

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    panels = [
        (focus_corrected, "Corrected chunk"),
        (focus_companion_nearest, "Companion resized nearest"),
        (focus_companion_bilinear, "Companion resized bilinear"),
        (focus_strict_nearest, "Strict score resized nearest"),
        (focus_strict_bilinear, "Strict score current bilinear"),
    ]
    for ax, (panel, title) in zip(axes.ravel()[:5], panels):
        ax.imshow(panel.T if focus_display_transposed else panel, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    stage_labels = [
        "corrected",
        "dino_input",
        "companion_nearest",
        "companion_bilinear",
        "strict_nearest",
        "strict_bilinear",
    ]
    stage_values = [
        chunk_rows[0]["temporal_roughness_corrected"],
        chunk_rows[0]["temporal_roughness_dino_input"],
        chunk_rows[0]["temporal_roughness_companion_nearest"],
        chunk_rows[0]["temporal_roughness_companion_bilinear"],
        chunk_rows[0]["temporal_roughness_strict_nearest"],
        chunk_rows[0]["temporal_roughness_strict_bilinear"],
    ]
    axes[1][2].bar(stage_labels, stage_values)
    axes[1][2].set_title(f"Temporal roughness by stage | subsection {focus_chunk['chunk_index']}")
    axes[1][2].tick_params(axis="x", rotation=30)

    summary = {
        "selected_subsections": [int(chunk["chunk_index"]) for chunk in selected_chunks],
        "median_time_bins_per_patch": float(np.median([row["time_bins_per_patch"] for row in chunk_rows])),
        "median_companion_temporal_window_px": float(np.median([row["companion_temporal_window_px"] for row in chunk_rows])),
        "median_strict_bilinear_temporal_roughness": float(np.median([row["temporal_roughness_strict_bilinear"] for row in chunk_rows])),
        "median_corrected_temporal_roughness": float(np.median([row["temporal_roughness_corrected"] for row in chunk_rows])),
    }
    return {
        "summary": summary,
        "chunk_rows": chunk_rows,
        "figure": fig,
        "axes": axes,
    }


def analyze_frontend_power_bias(
    pipeline_result: dict[str, Any],
    figsize: tuple[int, int] = (18, 10),
) -> dict[str, Any]:
    input_record = pipeline_result["input_record"]
    frontend = pipeline_result["frontend"]
    raw_sxx_db = np.asarray(input_record["sxx_db"], dtype=np.float32)
    corrected_sxx_db = np.asarray(pipeline_result["corrected_sxx_db"], dtype=np.float32)
    valid_row_mask = np.asarray(pipeline_result.get("valid_row_mask", np.ones(raw_sxx_db.shape[0], dtype=bool)), dtype=bool)
    raw_row_mean = np.mean(raw_sxx_db, axis=1).astype(np.float32)
    raw_row_p75 = np.percentile(raw_sxx_db, 75.0, axis=1).astype(np.float32)
    corrected_row_mean = np.mean(corrected_sxx_db, axis=1).astype(np.float32)
    row_floor = np.asarray(frontend["row_floor_db"], dtype=np.float32)
    response_db = np.asarray(frontend["response_db"], dtype=np.float32)
    boost_db = np.asarray(frontend["boost_db"], dtype=np.float32)

    summary = {
        "corr(boost_db, raw_row_mean_db)": _safe_corrcoef(boost_db[valid_row_mask], raw_row_mean[valid_row_mask]),
        "corr(boost_db, raw_row_p75_db)": _safe_corrcoef(boost_db[valid_row_mask], raw_row_p75[valid_row_mask]),
        "corr(boost_db, row_floor_db)": _safe_corrcoef(boost_db[valid_row_mask], row_floor[valid_row_mask]),
        "corr(boost_db, response_db)": _safe_corrcoef(boost_db[valid_row_mask], response_db[valid_row_mask]),
        "corr(boost_db, corrected_row_mean_db)": _safe_corrcoef(boost_db[valid_row_mask], corrected_row_mean[valid_row_mask]),
        "reference_db": float(frontend["reference_db"]),
    }

    subsection_rows = []
    for chunk in pipeline_result["chunk_results"]:
        row_start = int(chunk["row_start"])
        row_stop = int(chunk["row_stop"])
        subsection_rows.append({
            "subsection_index": int(chunk["chunk_index"]),
            "rows": (row_start, row_stop),
            "raw_row_mean_db": float(np.mean(raw_row_mean[row_start:row_stop])),
            "response_db": float(np.mean(response_db[row_start:row_stop])),
            "boost_db": float(np.mean(boost_db[row_start:row_stop])),
            "corrected_row_mean_db": float(np.mean(corrected_row_mean[row_start:row_stop])),
        })

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    row_idx = np.arange(raw_sxx_db.shape[0], dtype=np.int32)
    axes[0][0].plot(row_idx, raw_row_mean, label="raw_row_mean_db")
    axes[0][0].plot(row_idx, row_floor, label="row_floor_db")
    axes[0][0].plot(row_idx, response_db, label="response_db")
    axes[0][0].set_title("Global row-power drivers")
    axes[0][0].legend(loc="best")

    axes[0][1].plot(row_idx, boost_db, label="boost_db")
    axes[0][1].set_title("Boost is row-global")
    axes[0][1].legend(loc="best")

    axes[1][0].scatter(raw_row_mean[valid_row_mask], boost_db[valid_row_mask], s=12, alpha=0.6)
    axes[1][0].set_title("Boost vs raw row mean")
    axes[1][0].set_xlabel("Raw row mean (dB)")
    axes[1][0].set_ylabel("Boost (dB)")

    axes[1][1].scatter(response_db[valid_row_mask], boost_db[valid_row_mask], s=12, alpha=0.6)
    axes[1][1].set_title("Boost vs smoothed response")
    axes[1][1].set_xlabel("Smoothed response (dB)")
    axes[1][1].set_ylabel("Boost (dB)")

    return {
        "summary": summary,
        "subsection_rows": subsection_rows,
        "figure": fig,
        "axes": axes,
    }


def nonlocal_texture_recurrence_patch_map(input_gray01: np.ndarray, patch_h: int, patch_w: int, k: int = 6) -> np.ndarray:
    gray = np.asarray(input_gray01, dtype=np.float32)
    if gray.ndim != 2:
        raise ValueError("input_gray01 must be a 2D array")
    block_h = max(1, gray.shape[0] // int(patch_h))
    block_w = max(1, gray.shape[1] // int(patch_w))
    h_use = int(patch_h) * block_h
    w_use = int(patch_w) * block_w
    gray_crop = gray[:h_use, :w_use]
    patches = gray_crop.reshape(int(patch_h), block_h, int(patch_w), block_w).transpose(0, 2, 1, 3)
    patches = patches.reshape(int(patch_h) * int(patch_w), block_h * block_w).astype(np.float32)
    patches = patches - np.mean(patches, axis=1, keepdims=True)
    patches = patches / np.maximum(np.std(patches, axis=1, keepdims=True), 1e-4)
    if patches.shape[0] <= 1:
        return np.ones((int(patch_h), int(patch_w)), dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=min(int(k) + 1, patches.shape[0]), metric="cosine")
    nn.fit(patches)
    distances = nn.kneighbors(patches, return_distance=True)[0]
    recurrence = np.mean(distances[:, 1:], axis=1).astype(np.float32)
    score = _normalize_map01_local(recurrence.reshape(int(patch_h), int(patch_w)), 5.0, 95.0)
    return score.astype(np.float32)


def nonlocal_texture_recurrence_mask_from_gray(
    input_gray01: np.ndarray,
    patch_h: int,
    patch_w: int,
    k: int = 6,
    q: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, float]:
    texture_score = nonlocal_texture_recurrence_patch_map(input_gray01, patch_h, patch_w, k=k)
    texture_thr = float(np.quantile(texture_score, float(np.clip(q, 0.50, 0.99))))
    texture_mask = texture_score >= texture_thr
    return texture_mask.astype(np.uint8), texture_score.astype(np.float32), texture_thr


def _component_speckle_score(mask: np.ndarray, min_size: int = 6) -> float:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    active = float(np.count_nonzero(mask_u8))
    if active <= 0.0:
        return 0.0
    comp, n_comp = ndimage.label(mask_u8)
    if n_comp <= 0:
        return 0.0
    sizes = ndimage.sum(mask_u8, comp, index=np.arange(1, n_comp + 1))
    sizes = np.asarray(sizes, dtype=np.float32)
    kept = float(np.sum(sizes[sizes >= float(max(1, min_size))]))
    return float(np.clip(kept / max(active, 1.0), 0.0, 1.0))


def texture_reliability_score(texture_map: np.ndarray, agreement_map: np.ndarray, min_size: int = 6) -> dict[str, float]:
    texture_map = _normalize_map01_local(np.asarray(texture_map, dtype=np.float32), 5.0, 95.0)
    agreement_map = _normalize_map01_local(np.asarray(agreement_map, dtype=np.float32), 5.0, 95.0)
    texture_thr = float(np.quantile(texture_map, 0.90))
    agreement_thr = float(np.quantile(agreement_map, 0.85))
    texture_top_mask = texture_map >= texture_thr
    agreement_top_mask = agreement_map >= agreement_thr
    if np.any(texture_top_mask):
        peakiness = float(np.mean(texture_map[texture_top_mask]) - np.median(texture_map))
        agreement_support = float(np.mean(agreement_map[texture_top_mask]))
    else:
        peakiness = 0.0
        agreement_support = 0.0
    peakiness = float(np.clip(peakiness / 0.5, 0.0, 1.0))
    speckle_score = _component_speckle_score(texture_top_mask.astype(np.uint8), min_size=min_size)
    union = np.logical_or(texture_top_mask, agreement_top_mask)
    overlap = 0.0
    if np.any(union):
        overlap = float(np.count_nonzero(np.logical_and(texture_top_mask, agreement_top_mask)) / np.count_nonzero(union))
    reliability = float(np.clip(
        0.35 * peakiness + 0.25 * speckle_score + 0.20 * agreement_support + 0.20 * overlap,
        0.0,
        1.0,
    ))
    return {
        "reliability": reliability,
        "peakiness": peakiness,
        "speckle_score": speckle_score,
        "agreement_support": float(np.clip(agreement_support, 0.0, 1.0)),
        "overlap": overlap,
    }


def _texture_passthrough_from_structure(structure_map: np.ndarray, min_size: int = 6) -> dict[str, float]:
    structure_map = _normalize_map01_local(np.asarray(structure_map, dtype=np.float32), 5.0, 95.0)
    structure_thr = float(np.quantile(structure_map, 0.85))
    structure_top_mask = structure_map >= structure_thr
    structure_peak = 0.0
    if np.any(structure_top_mask):
        structure_peak = float(np.mean(structure_map[structure_top_mask]))
    structure_speckle = _component_speckle_score(structure_top_mask.astype(np.uint8), min_size=min_size)
    structure_clean_score = float(np.clip(0.70 * structure_speckle + 0.30 * structure_peak, 0.0, 1.0))
    speckle_clean_thr = 0.96
    speckle_strong_thr = 0.50
    if structure_speckle >= speckle_clean_thr:
        clean_ramp = float(np.clip((structure_speckle - speckle_clean_thr) / max(1.0 - speckle_clean_thr, 1e-6), 0.0, 1.0))
        texture_passthrough = 0.25 + 0.75 * clean_ramp
    elif structure_speckle >= speckle_strong_thr:
        speckle_ramp = float(np.clip((structure_speckle - speckle_strong_thr) / max(speckle_clean_thr - speckle_strong_thr, 1e-6), 0.0, 1.0))
        texture_passthrough = 0.05 + 0.20 * speckle_ramp
    else:
        low_ramp = float(np.clip(structure_speckle / max(speckle_strong_thr, 1e-6), 0.0, 1.0))
        texture_passthrough = 0.02 + 0.03 * low_ramp
    return {
        "structure_speckle_score": structure_speckle,
        "structure_peak": structure_peak,
        "structure_clean_score": structure_clean_score,
        "texture_passthrough": float(np.clip(texture_passthrough, 0.0, 1.0)),
    }


def build_texture_augmented_chunk_experiment(
    chunk: dict[str, Any],
    final_score_q: float = 0.88,
    texture_knn: int = 6,
    min_component_size: int = 6,
) -> dict[str, Any]:
    dino_group = chunk["dino_group"]
    input_debug = dino_group.get("input_debug", {})
    input_gray01 = np.asarray(input_debug.get("input_gray01"), dtype=np.float32)
    if input_gray01.ndim != 2:
        raise ValueError("chunk does not contain a usable DINO gray input for texture analysis")

    patch_h, patch_w = tuple(chunk["dino_gated"]["shape"])
    texture_patch = nonlocal_texture_recurrence_patch_map(input_gray01, patch_h, patch_w, k=texture_knn)
    dino_patch = _normalize_map01_local(np.asarray(chunk["dino_gated"]["score"], dtype=np.float32), 5.0, 95.0)
    power_patch = _normalize_map01_local(np.asarray(chunk["power_patch"], dtype=np.float32), 5.0, 95.0)
    agreement_patch = np.sqrt(np.clip(dino_patch * power_patch, 0.0, None)).astype(np.float32)
    texture_policy = texture_reliability_score(texture_patch, agreement_patch, min_size=max(3, min_component_size // 2))
    structure_policy = _texture_passthrough_from_structure(
        np.asarray(chunk["dino_gated"]["coherence_gate_patch"], dtype=np.float32),
        min_size=max(3, min_component_size // 2),
    )
    support_patch = _normalize_map01_local(np.asarray(chunk["support_patch"], dtype=np.float32), 5.0, 95.0)
    strict_patch = _normalize_map01_local(np.asarray(chunk["strict_patch"], dtype=np.float32), 5.0, 95.0)
    companion_patch = _normalize_map01_local(np.asarray(chunk["burst_gate"]["companion_patch"], dtype=np.float32), 5.0, 95.0)
    texture_gain = float(np.clip(texture_policy["reliability"] * structure_policy["texture_passthrough"], 0.0, 1.0))
    texture_bonus = texture_gain * texture_patch * np.clip(0.35 + 0.65 * support_patch, 0.0, 1.0)
    experiment_patch = _normalize_map01_local(strict_patch + texture_bonus, 5.0, 95.0)

    patch_weight = np.ones_like(experiment_patch, dtype=np.float32)
    pixel_weight = np.ones(tuple(np.asarray(chunk["score_px"], dtype=np.float32).shape), dtype=np.float32)
    experiment_patch_weighted = np.clip(experiment_patch * patch_weight, 0.0, 1.0).astype(np.float32)
    rows, cols = tuple(np.asarray(chunk["score_px"], dtype=np.float32).shape)
    experiment_score_px_raw = _resize_patch_map_to_pixels(experiment_patch_weighted, rows, cols, Image.BILINEAR)
    experiment_score_px = np.clip(experiment_score_px_raw * pixel_weight, 0.0, 1.0).astype(np.float32)
    score_valid_mask = np.ones_like(experiment_score_px, dtype=bool)
    experiment_score_px_calibrated, experiment_score_calibration = _calibrate_score_map_local(experiment_score_px, valid_mask=score_valid_mask)
    experiment_threshold = float(np.quantile(experiment_patch, float(np.clip(final_score_q, 0.50, 0.99))))
    experiment_mask_patch = _smooth_binary_label_map(
        (experiment_patch >= experiment_threshold).astype(np.uint8),
        iters=1,
        min_component_size=min_component_size,
    )
    experiment_mask_px = _resize_patch_mask_to_pixels(experiment_mask_patch, rows, cols)

    return {
        "texture_patch": texture_patch.astype(np.float32),
        "agreement_patch": agreement_patch.astype(np.float32),
        "texture_bonus_patch": texture_bonus.astype(np.float32),
        "texture_gain": texture_gain,
        "texture_policy": texture_policy,
        "structure_policy": structure_policy,
        "score_patch": experiment_patch.astype(np.float32),
        "score_patch_weighted": experiment_patch_weighted.astype(np.float32),
        "score_px_raw": experiment_score_px_raw.astype(np.float32),
        "score_px": experiment_score_px.astype(np.float32),
        "score_px_calibrated": experiment_score_px_calibrated.astype(np.float32),
        "score_calibration": experiment_score_calibration,
        "threshold": experiment_threshold,
        "mask_patch": experiment_mask_patch.astype(np.uint8),
        "mask_px": experiment_mask_px.astype(bool),
        "support_px": np.asarray(chunk["support_px"], dtype=bool),
        "companion_px": _resize_patch_map_to_pixels(companion_patch, rows, cols, Image.BILINEAR).astype(np.float32),
    }


def run_texture_augmented_experiment(
    pipeline_result: dict[str, Any],
    final_score_q: float = 0.88,
    texture_knn: int = 6,
    min_component_size: int | None = None,
) -> dict[str, Any]:
    chunk_results = pipeline_result["chunk_results"]
    if min_component_size is None:
        min_component_size = int(getattr(pipeline_result.get("config"), "min_component_size", 6))
    experiment_chunks: list[dict[str, Any]] = []
    for chunk in chunk_results:
        experiment = build_texture_augmented_chunk_experiment(
            chunk,
            final_score_q=final_score_q,
            texture_knn=texture_knn,
            min_component_size=min_component_size,
        )
        experiment_chunks.append({
            **chunk,
            "score_px": experiment["score_px"],
            "score_px_raw": experiment["score_px_raw"],
            "score_px_calibrated": experiment["score_px_calibrated"],
            "score_calibration": experiment["score_calibration"],
            "mask_px": experiment["mask_px"],
            "support_px": experiment["support_px"],
            "companion_px": experiment["companion_px"],
            "texture_experiment": experiment,
        })
    merged = merge_chunk_results(
        pipeline_result["corrected_sxx_db"].shape,
        experiment_chunks,
        final_score_q=final_score_q,
        min_component_size=min_component_size,
    )
    return {
        "chunk_results": experiment_chunks,
        **merged,
        "texture_knn": int(texture_knn),
        "final_score_q": float(final_score_q),
    }


def run_chunked_pipeline(
    input_record: dict[str, Any],
    model,
    patch_size: int,
    device: str,
    cfg: WidebandChunkConfig,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    calibrated_axis = has_calibrated_frequency_axis(input_record)
    effective_ignore_sideband_hz = cfg.ignore_sideband_hz if calibrated_axis else None
    ignore_info = compute_ignore_sideband_rows(
        input_record["freq_axis_hz"],
        ignore_sideband_percent=cfg.ignore_sideband_percent,
        min_keep_rows=max(int(patch_size), 16),
        ignore_sideband_hz=effective_ignore_sideband_hz,
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
        uncalibrated_chunk_fraction=cfg.uncalibrated_chunk_fraction,
        uncalibrated_overlap_fraction=cfg.uncalibrated_overlap_fraction,
    )
    chunk_results: list[dict[str, Any]] = []
    for chunk in chunk_plan:
        row_slice = slice(chunk["row_start"], chunk["row_stop"])
        chunk_valid_row_mask = valid_row_mask[row_slice]
        detection = run_chunk_detector(
            corrected_sxx_db[row_slice, :],
            model,
            patch_size,
            device,
            cfg,
            valid_row_mask=chunk_valid_row_mask,
        )
        chunk_results.append({**chunk, **detection, "valid_row_mask": chunk_valid_row_mask.astype(bool)})
    merged = merge_chunk_results(
        corrected_sxx_db.shape,
        chunk_results,
        final_score_q=cfg.final_score_q,
        min_component_size=cfg.min_component_size,
        global_valid_row_mask=valid_row_mask,
        grouping_seed_score_q=cfg.grouping_seed_score_q,
        grouping_bridge_freq_px=cfg.grouping_bridge_freq_px,
        grouping_bridge_time_px=cfg.grouping_bridge_time_px,
        grouping_min_component_size=cfg.grouping_min_component_size,
        grouping_min_freq_span_px=cfg.grouping_min_freq_span_px,
        grouping_min_time_span_px=cfg.grouping_min_time_span_px,
        grouping_min_density=cfg.grouping_min_density,
    )
    t1 = time.perf_counter()
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
    display_raw = np.asarray(record.get("display_sxx_db", record["sxx_db"]), dtype=np.float32)
    display_transposed = bool(record.get("display_transposed", input_kind_requires_display_transpose(record.get("input_kind"))))
    display_corrected = corrected_sxx_db.T if display_transposed else corrected_sxx_db
    raw_vmin, raw_vmax = _display_db_window(record["sxx_db"])
    corrected_vmin, corrected_vmax = _display_db_window(corrected_sxx_db)
    fig, axes = plt.subplots(4, 1, figsize=figsize, constrained_layout=True)
    axes[0].imshow(display_raw, aspect="auto", origin="lower", cmap="viridis", vmin=raw_vmin, vmax=raw_vmax)
    axes[0].set_title("Wideband input spectrogram")
    axes[0].set_ylabel("Time bin")
    axes[1].imshow(display_corrected, aspect="auto", origin="lower", cmap="viridis", vmin=corrected_vmin, vmax=corrected_vmax)
    axes[1].set_title("Globally corrected spectrogram")
    axes[1].set_ylabel("Time bin")
    axes[2].plot(frontend["row_floor_db"], freq_axis_hz, label="Row floor")
    axes[2].plot(frontend["response_db"], freq_axis_hz, label="Smoothed response")
    axes[2].axvline(frontend["reference_db"], color="tab:green", linestyle="--", label="Reference")
    axes[3].plot(frontend["boost_db"], freq_axis_hz, label="Boost (dB)")
    for chunk in pipeline_result["chunk_plan"]:
        if display_transposed:
            axes[1].axvline(chunk["row_start"], color="white", alpha=0.15, linewidth=0.8)
            axes[1].axvline(chunk["row_stop"] - 1, color="white", alpha=0.15, linewidth=0.8)
        else:
            axes[1].axhline(chunk["row_start"], color="white", alpha=0.15, linewidth=0.8)
            axes[1].axhline(chunk["row_stop"] - 1, color="white", alpha=0.15, linewidth=0.8)
    ignored_rows = np.flatnonzero(~valid_row_mask)
    if ignored_rows.size > 0:
        low_block = ignored_rows[ignored_rows < (corrected_sxx_db.shape[0] // 2)]
        high_block = ignored_rows[ignored_rows >= (corrected_sxx_db.shape[0] // 2)]
        if low_block.size > 0:
            for ax in axes[:2]:
                if display_transposed:
                    ax.axvspan(low_block[0], low_block[-1], color="black", alpha=0.18)
                else:
                    ax.axhspan(low_block[0], low_block[-1], color="black", alpha=0.18)
        if high_block.size > 0:
            for ax in axes[:2]:
                if display_transposed:
                    ax.axvspan(high_block[0], high_block[-1], color="black", alpha=0.18)
                else:
                    ax.axhspan(high_block[0], high_block[-1], color="black", alpha=0.18)
    axes[2].set_title("Global frontend correction profile")
    axes[2].set_xlabel("Level (dB)")
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].legend(loc="best")
    axes[3].set_title("Frontend boost profile")
    axes[3].set_xlabel("Boost (dB)")
    axes[3].set_ylabel("Frequency (Hz)")
    axes[3].legend(loc="best")
    return fig, axes


def plot_chunk_examples(pipeline_result: dict[str, Any], max_chunks: int = 4, figsize: tuple[int, int] = (22, 5)):
    chunk_results = pipeline_result["chunk_results"][: max(1, int(max_chunks))]
    input_kind = pipeline_result["input_record"].get("input_kind")
    display_transposed = bool(pipeline_result["input_record"].get("display_transposed", input_kind_requires_display_transpose(input_kind)))
    fig, axes = plt.subplots(len(chunk_results), 4, figsize=(figsize[0], figsize[1] * len(chunk_results)), constrained_layout=True)
    if len(chunk_results) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_idx, chunk in enumerate(chunk_results):
        tile = pipeline_result["corrected_sxx_db"][chunk["row_start"]:chunk["row_stop"], :]
        display_tile = tile.T if display_transposed else tile
        display_mask = chunk["mask_px"].T if display_transposed else chunk["mask_px"]
        dino = chunk["dino_gated"]
        axes[row_idx][0].imshow(display_tile, aspect="auto", origin="lower", cmap="viridis")
        axes[row_idx][0].set_title(f"Spectrogram subsection {chunk['chunk_index']} corrected tile")
        axes[row_idx][1].imshow(dino["input_img"])
        axes[row_idx][1].set_title("DINO input")
        axes[row_idx][2].imshow(chunk["strict_patch"], cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[row_idx][2].set_title("Subsection strict patch score")
        axes[row_idx][3].imshow(display_mask, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_idx][3].set_title("Subsection pixel mask")
        for ax in axes[row_idx]:
            ax.set_xticks([])
            ax.set_yticks([])
    return fig, axes


def plot_merged_detection(pipeline_result: dict[str, Any], figsize: tuple[int, int] = (20, 6)):
    corrected_sxx_db = pipeline_result["corrected_sxx_db"]
    merged_score = pipeline_result["merged_score"]
    merged_mask = pipeline_result["merged_mask"]
    merged_support = np.asarray(pipeline_result.get("merged_support", np.ones_like(merged_mask, dtype=bool)), dtype=bool)
    merged_boxes = list(pipeline_result.get("merged_boxes", []))
    valid_row_mask = np.asarray(pipeline_result.get("valid_row_mask", np.ones(corrected_sxx_db.shape[0], dtype=bool)), dtype=bool)
    display_transposed = bool(pipeline_result["input_record"].get("display_transposed", input_kind_requires_display_transpose(pipeline_result["input_record"].get("input_kind"))))
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
    axes[1].set_title("Merged strict score + support")
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


def _debug_display_orientation(pipeline_result: dict[str, Any]) -> tuple[bool, np.ndarray, float, float]:
    corrected_sxx_db = np.asarray(pipeline_result["corrected_sxx_db"], dtype=np.float32)
    display_transposed = bool(
        pipeline_result["input_record"].get(
            "display_transposed",
            input_kind_requires_display_transpose(pipeline_result["input_record"].get("input_kind")),
        )
    )
    vmin, vmax = _display_db_window(corrected_sxx_db)
    return display_transposed, corrected_sxx_db, vmin, vmax


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
):
    display_base = base.T if display_transposed else base
    display_overlay = overlay.T if display_transposed else overlay
    ax.imshow(display_base, aspect="auto", origin="lower", cmap="gray", vmin=base_vmin, vmax=base_vmax, interpolation="nearest")
    ax.imshow(np.where(display_overlay, 1.0, np.nan), aspect="auto", origin="lower", cmap=overlay_cmap, alpha=overlay_alpha, interpolation="nearest")
    _draw_signal_boxes(ax, boxes, display_transposed)
    ax.set_title(title)
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Time bin")


def _build_merged_debug_views(pipeline_result: dict[str, Any]) -> dict[str, np.ndarray]:
    corrected_sxx_db = np.asarray(pipeline_result["corrected_sxx_db"], dtype=np.float32)
    merged_raw_dino_mask = np.zeros_like(corrected_sxx_db, dtype=bool)
    merged_dino_score = np.zeros_like(corrected_sxx_db, dtype=np.float32)
    merged_support = np.zeros_like(corrected_sxx_db, dtype=bool)
    merged_power = np.zeros_like(corrected_sxx_db, dtype=np.float32)
    for chunk in pipeline_result["chunk_results"]:
        row_start = int(chunk["row_start"])
        row_stop = int(chunk["row_stop"])
        rows = row_stop - row_start
        cols = corrected_sxx_db.shape[1]
        raw_dino_mask_px = _resize_patch_mask_to_pixels(np.asarray(chunk["dino_group"]["mask"], dtype=np.float32), rows, cols)
        merged_raw_dino_mask[row_start:row_stop, :] |= raw_dino_mask_px
        merged_dino_score[row_start:row_stop, :] = np.maximum(
            merged_dino_score[row_start:row_stop, :],
            np.asarray(chunk.get("dino_score_px", chunk["score_px"]), dtype=np.float32),
        )
        merged_support[row_start:row_stop, :] |= np.asarray(chunk["support_px"], dtype=bool)
        merged_power[row_start:row_stop, :] = np.maximum(
            merged_power[row_start:row_stop, :],
            np.asarray(chunk.get("power_px", np.zeros((rows, cols), dtype=np.float32)), dtype=np.float32),
        )
    return {
        "raw_dino_mask": merged_raw_dino_mask,
        "dino_score": merged_dino_score,
        "support": merged_support,
        "power": merged_power,
    }


def plot_merged_debug(pipeline_result: dict[str, Any], figsize: tuple[int, int] = (24, 5)):
    corrected_sxx_db = pipeline_result["corrected_sxx_db"]
    merged_score = np.asarray(pipeline_result["merged_score"], dtype=np.float32)
    merged_mask = np.asarray(pipeline_result["merged_mask"], dtype=bool)
    merged_support = np.asarray(pipeline_result.get("merged_support", np.zeros_like(corrected_sxx_db, dtype=bool)), dtype=bool)
    merged_boxes = list(pipeline_result.get("merged_boxes", []))
    display_transposed, corrected_sxx_db, vmin, vmax = _debug_display_orientation(pipeline_result)
    debug_views = _build_merged_debug_views(pipeline_result)
    fig, axes = plt.subplots(1, 5, figsize=figsize, constrained_layout=True)
    _show_debug_panel(axes[0], corrected_sxx_db, "Input spectrogram", display_transposed, "viridis", vmin, vmax)
    _show_debug_panel(axes[1], debug_views["raw_dino_mask"].astype(np.float32), "Raw DINO mask", display_transposed, "gray", 0.0, 1.0)
    _show_debug_panel(axes[2], debug_views["dino_score"], "DINO score + support", display_transposed, "magma", 0.0, 1.0)
    support_display = debug_views["support"].T if display_transposed else debug_views["support"]
    axes[2].imshow(np.where(support_display, 1.0, np.nan), aspect="auto", origin="lower", cmap="winter", alpha=0.18, interpolation="nearest")
    _draw_signal_boxes(axes[2], merged_boxes, display_transposed)
    _show_debug_panel(axes[3], debug_views["power"], "Normalized power", display_transposed, "cividis", 0.0, 1.0)
    _show_debug_overlay(axes[4], corrected_sxx_db, merged_mask, "Grouped final overlay", display_transposed, vmin, vmax, boxes=merged_boxes)
    return fig, axes


def _compute_subsection_debug_maps(
    pipeline_result: dict[str, Any],
    chunk: dict[str, Any],
    corrected_chunk: np.ndarray,
) -> dict[str, np.ndarray | float]:
    cache_version = 2
    texture_source = "coherence"
    texture_display = "score"
    cached = chunk.get("_subsection_debug_cache")
    if (
        isinstance(cached, dict)
        and int(cached.get("cache_version", 0)) == cache_version
        and cached.get("texture_source") == texture_source
        and cached.get("texture_display") == texture_display
    ):
        return cached
    cfg = pipeline_result.get("config")
    patch_h, patch_w = tuple(chunk["dino_group"]["shape"])
    coherence = multi_scale_structure_tensor_gate(
        corrected_chunk,
        patch_h,
        patch_w,
        max_height_px=None if cfg is None else cfg.coherence_max_height_px,
        max_width_px=None if cfg is None else cfg.coherence_max_width_px,
    )
    coherence_gray01 = np.asarray(np.clip(coherence["coherence_px"], 0.0, 1.0), dtype=np.float32)
    if coherence_gray01.ndim == 2:
        texture_mask_patch, texture_score_patch, texture_threshold = nonlocal_texture_recurrence_mask_from_gray(
            coherence_gray01,
            patch_h,
            patch_w,
            k=6,
            q=0.90,
        )
    else:
        texture_mask_patch = np.zeros((patch_h, patch_w), dtype=np.uint8)
        texture_score_patch = np.zeros((patch_h, patch_w), dtype=np.float32)
        texture_threshold = 1.0
    texture_mask_px = _resize_patch_mask_to_pixels(texture_mask_patch.astype(np.float32), corrected_chunk.shape[0], corrected_chunk.shape[1])
    texture_score_px = np.clip(
        _resize_patch_map_to_pixels(texture_score_patch.astype(np.float32), corrected_chunk.shape[0], corrected_chunk.shape[1], Image.NEAREST),
        0.0,
        1.0,
    ).astype(np.float32)
    debug_maps = {
        "cache_version": cache_version,
        "coherence_px": np.asarray(coherence["coherence_px"], dtype=np.float32),
        "texture_mask_px": texture_mask_px.astype(bool),
        "texture_score_patch": texture_score_patch.astype(np.float32),
        "texture_score_px": np.asarray(texture_score_px, dtype=np.float32),
        "texture_threshold": float(texture_threshold),
        "texture_source": texture_source,
        "texture_display": texture_display,
    }
    chunk["_subsection_debug_cache"] = debug_maps
    return debug_maps


def plot_subsection_debug(
    pipeline_result: dict[str, Any],
    subsection_index: int,
    figsize: tuple[int, int] = (34, 5),
):
    chunk = next(
        (candidate for candidate in pipeline_result["chunk_results"] if int(candidate["chunk_index"]) == int(subsection_index)),
        None,
    )
    if chunk is None:
        raise ValueError(f"No subsection found for index {subsection_index}")
    row_start = int(chunk["row_start"])
    row_stop = int(chunk["row_stop"])
    corrected_chunk = np.asarray(pipeline_result["corrected_sxx_db"][row_start:row_stop, :], dtype=np.float32)
    rows, cols = corrected_chunk.shape
    display_transposed, _, vmin, vmax = _debug_display_orientation(pipeline_result)
    raw_dino_mask = _resize_patch_mask_to_pixels(np.asarray(chunk["dino_group"]["mask"], dtype=np.float32), rows, cols)
    dino_score = np.asarray(chunk.get("dino_score_px", chunk["score_px"]), dtype=np.float32)
    support = np.asarray(chunk["support_px"], dtype=bool)
    power = np.asarray(chunk.get("power_px", np.zeros_like(corrected_chunk)), dtype=np.float32)
    on_demand_debug = _compute_subsection_debug_maps(pipeline_result, chunk, corrected_chunk)
    coherence = np.asarray(on_demand_debug["coherence_px"], dtype=np.float32)
    texture_score = np.asarray(on_demand_debug["texture_score_px"], dtype=np.float32)
    final_mask = np.asarray(chunk["mask_px"], dtype=bool)
    fig, axes = plt.subplots(1, 7, figsize=figsize, constrained_layout=True)
    _show_debug_panel(axes[0], corrected_chunk, f"Subsection {subsection_index} input", display_transposed, "viridis", vmin, vmax)
    _show_debug_panel(axes[1], raw_dino_mask.astype(np.float32), f"Subsection {subsection_index} raw DINO mask", display_transposed, "gray", 0.0, 1.0)
    _show_debug_panel(axes[2], dino_score, f"Subsection {subsection_index} DINO score + support", display_transposed, "magma", 0.0, 1.0)
    support_display = support.T if display_transposed else support
    axes[2].imshow(np.where(support_display, 1.0, np.nan), aspect="auto", origin="lower", cmap="winter", alpha=0.18, interpolation="nearest")
    _show_debug_panel(axes[3], power, f"Subsection {subsection_index} normalized power", display_transposed, "cividis", 0.0, 1.0)
    _show_debug_panel(axes[4], coherence, f"Subsection {subsection_index} coherence", display_transposed, "plasma", 0.0, 1.0)
    _show_debug_panel(axes[5], texture_score, f"Subsection {subsection_index} texture score from coherence", display_transposed, "magma", 0.0, 1.0)
    _show_debug_overlay(axes[6], corrected_chunk, final_mask, f"Subsection {subsection_index} final overlay", display_transposed, vmin, vmax)
    return fig, axes