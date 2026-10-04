# Performance Recommendations

Tuning guide for the v2 pipeline plus the next optimizations worth doing.
Measure first: `python bench/bench_match.py --minutes 2 --height 2160`.

## Picking settings

| Situation | Do |
|---|---|
| First run on a new match / smoke test | `profile: fast` and a trim window (`trim_start`/`trim_end`) |
| Normal processing on an RTX 30/40/50, Spark | `balanced` (default) |
| Small players far from the camera, ball lost often | `quality` (imgsz 1536, `yolov8m`, ball tiles) or `balanced` + `inference_imgsz: 1536` |
| Apple Silicon | `balanced` on Ultra-class; `fast` on Max and smaller |
| CPU only | `fast` with a trim window; full matches take hours |
| Want newer weights | `VH_MODEL_FAMILY=yolo26` (same sizes), or a fine-tuned `best.pt` as `yolo_model` |
| Repeated runs on one CUDA box | `VH_TENSORRT=1` with a GPU image built `INSTALL_TENSORRT=1`; keep `/models` on a volume so engines persist |

Detection + tracking is 50-80% of runtime on GPUs. Cost scales with
`(imgsz)^2 / vid_stride` and model size (n 1.7x, s 1x, m 0.5x the
throughput of s). Proxy and render are fixed per match and bound by
NVDEC/NVENC (or VideoToolbox) throughput.

## Host settings

1. Put `VH_OUTPUT_ROOT` (`/data`) and sources on NVMe; the proxy pass reads
   the whole source and the render reads it again.
2. Docker Desktop (Windows): raise WSL2 `memory`/`processors` in
   `.wslconfig`; keep sources on a local NTFS drive (bind mounts from
   network shares are slow).
3. GPU worker: `ipc: host` and a large `shm_size` are set in compose; keep
   them.
4. One worker per GPU. Do not run `worker` and `worker-gpu` together (queue
   claim is not atomic, `docs/DEPLOYMENT.md`).
5. Stop other GPU users (games, local LLMs) during processing; Ollama models
   unload after each answer by default (`VH_LLM_KEEP_ALIVE=0`).

## Next optimizations

1. **Atomic queue claim** (`UPDATE ... WHERE status='queued'` /
   `SKIP LOCKED`) so several workers can share a queue, then per-GPU workers.
2. **GPU-resident proxy**: `-hwaccel cuda -hwaccel_output_format cuda` with
   `scale_cuda` and NVENC keeps frames on the GPU; the distro ffmpeg supports
   it. Same for the final render (crop on the CPU is the remaining copy).
3. **Decode straight to tensors** (NVDEC -> torch via PyNvVideoCodec or
   torchaudio's StreamReader) to skip the proxy re-decode on CUDA hosts.
4. **Chunked, resumable jobs**: split long matches into segments with
   overlap so a crash resumes instead of restarting, and segments can run on
   several GPUs.
5. **INT8 TensorRT** for the detector after a calibration set exists from
   reviewed matches.
6. **Persist per-stage timings** from `progress.json` into the job record so
   the estimator can be calibrated from real runs per hardware class.

## Benchmark matrix

| Dimension | Values |
|---|---|
| Source | 1080p30, 4K30 (Falcon), 4K60 |
| Length | bench 2 min, projection 90 min; one real full match per release |
| Hardware | each `hardware_class` in `perf_profiles.HARDWARE_CLASSES` |
| Profile | fast, balanced, quality; TensorRT on/off on CUDA |

Track: per-stage fps and seconds (bench JSON), end-to-end wall time, GPU
utilization and peak memory, camera quality metrics (`camera_quality.json`)
and event precision/recall on labeled matches.
