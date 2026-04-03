# Speed Optimization Todo

## Goal

Speed up the DINO grouping and DINO coherence portions of the RF signal detection pipeline, with a path toward an NVIDIA Holoscan GPU processing pipeline.

## Current Hotspots

- DINO feature extraction runs on GPU, but grouping moves features back to CPU/NumPy immediately after inference.
- DINO grouping uses CPU-side PCA, sklearn k-NN, affinity construction, and several Python loops.
- Coherence gating uses repeated SciPy Gaussian and uniform filters on CPU.
- Full-resolution coherence is computed before patch reduction, which adds avoidable cost.

## Phase 1: Fastest Notebook Wins

1. Keep DINO features on GPU after inference instead of converting to NumPy immediately.
2. Replace sklearn nearest-neighbor search with a GPU alternative.
   Options: FAISS GPU, PyTorch cosine similarity plus `topk`.
3. Replace CPU PCA with a GPU implementation or remove PCA if accuracy impact is acceptable.
4. Reduce `DINO_GROUP_K` if possible to lower graph construction cost.
5. Reduce coherence scales from `(0.8, 1.6, 3.2)` to one or two scales if quality remains acceptable.
6. Profile whether full-resolution coherence is necessary, or whether a patch-grid approximation is sufficient.
7. Batch multiple slices together for DINO inference when memory allows.

## Phase 2: GPU Port of Postprocessing

1. Reimplement DINO grouping with GPU tensors.
   Scope:
   - feature normalization
   - PCA or projection
   - affinity matrix / k-NN graph
   - support map computation
   - thresholding and mask generation
2. Remove Python patch-grid loops and replace them with tensorized operations.
3. Port structure-tensor coherence to GPU using separable convolutions.
   Options:
   - PyTorch conv kernels
   - CUDA custom kernels
   - cuCIM / OpenCV CUDA if integration is simpler
4. Keep final fusion and thresholding on GPU.
5. Only copy compact outputs to CPU for plotting, logging, or downstream serialization.

## Phase 3: Holoscan Pipeline Design

1. Frontend correction / spectrogram preprocessing operator on GPU.
2. DINO inference operator on GPU.
3. DINO grouping operator on GPU.
4. Coherence gate operator on GPU.
5. Fusion and threshold operator on GPU.
6. Visualization or export operator on CPU only when needed.

## Recommended Order of Work

1. Eliminate GPU-to-CPU conversion after DINO inference.
2. Move neighbor search to FAISS GPU or PyTorch.
3. Port the coherence stage to GPU.
4. Replace remaining Python loops in grouping with vectorized tensor code.
5. Integrate the GPU-native path into Holoscan operators.

## Algorithm Tradeoffs To Evaluate

- Smaller k in grouping versus detection quality.
- Fewer coherence scales versus robustness.
- No PCA versus PCA versus fixed learned projection.
- Patch-grid coherence versus full-resolution coherence.
- Larger DINO batches versus memory pressure and latency.

## Holoscan-Specific Notes

- Holoscan is most useful if tensors stay resident on GPU between operators.
- Avoid a design where DINO inference is on GPU but grouping/coherence bounce back to CPU.
- Prefer a staged migration: prototype operators in Python first, then move the slowest stable pieces to C++/CUDA only if needed.

## Deployment Risk Note

- Repository notes indicate Python TorchScript load/CUDA/eval already succeeds, but strict C++ TorchScript initialization still needs live runtime verification.
- Do not assume a pure C++ TorchScript path is the safest first production route.
- Safer first options:
  - Python Holoscan operator wrapping the existing model path
  - TensorRT-backed inference with separate GPU postprocessing operators

## Suggested First Benchmark Pass

1. Measure runtime of:
   - DINO inference
   - grouping
   - coherence
   - final fusion
2. Run the same input with:
   - current pipeline
   - reduced coherence scales
   - reduced grouping k
   - GPU k-NN prototype
3. Compare both latency and output stability.

## Definition of Done

- End-to-end GPU-resident path for DINO inference, grouping, coherence, and fusion.
- Minimal CPU transfers.
- Verified latency improvement on representative RF spectrogram inputs.
- No unacceptable regression in detection quality.