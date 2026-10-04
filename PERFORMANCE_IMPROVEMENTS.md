# Performance Improvements (v2)

Why a 90-minute 4K match took ~40 hours, and what v2 does instead. Source:
`PLAN.md` ("Where time went"). Status per item reflects the v2 rebuild in
progress; the integration wave wires the new modules into
`VideoHighlights.py` / `job_runner.py`.

| # | v1 cost | v2 change | Module |
|---|---|---|---|
| 1 | 3-4 full 4K decode passes per run (tracking, debug video, full movie, trim re-encode) plus seek-heavy team/card scans | One ffmpeg pass writes `proxy_1080p.mp4` + `audio_analysis.wav` + thumbnails; every analysis stage reads the proxy. The source is decoded once more, by ffmpeg, for the final render. | `frame_source.build_proxy` |
| 2 | Per-frame Python render: CPU 4K decode, crop, upscale back to 4K, 25 MB raw pipe write per frame | ffmpeg-native render: crop driven by `camera_crops.txt` (sendcmd), scale to `output_height` (1080), NVENC/VideoToolbox/x264, audio muxed in the same command | `camera_render` |
| 3 | Inference batch 1 at imgsz 960 on CPU-decoded 4K frames; `vid_stride` still decoded skipped frames | Threaded proxy decode, batched inference (auto batch by VRAM), fp16 on CUDA, optional cached TensorRT engine, MPS on Apple | `detectors`, `device`, `tracking_engine` |
| 4 | moviepy in reel, wide clips, montage, overlays | ffmpeg concat/xfade; clips cut from the finished movie with stream copy | `broadcast`, event clips |
| 5 | On-demand clips put `-ss` after `-i` (decoded from 0:00) | `-ss` before `-i` | `event_clip_renderer` |
| 6 | Debug video, full movie, reel, cards, auto colours and LLM all on by default | Profiles `fast` / `balanced` / `quality`; debug video off | `perf_profiles` |

Kept from v1: NVENC -> libx264 -> mpeg4 encoder fallback for clip export,
FP16 on CUDA, parallel clip rendering, audio-less retry on audio codec errors.

## Hardware detection

`GET /v1/health/gpu` (`backend/services/gpu_status.py`) reports CUDA, MPS,
`nvidia-smi`, ffmpeg `hwaccels` and `encoders`, and derives
`recommended_device`, `recommended_encoder`, `recommended_hwaccel` and
`hardware_class`. `backend/services/device.py` verifies that a listed
encoder/hwaccel actually works with a tiny real encode before using it
(ffmpeg lists NVENC on machines with no NVIDIA GPU).

## Containers

* CPU image installs torch from the CPU wheel index (~200 MB instead of
  the multi-GB CUDA wheel).
* GPU image uses the CUDA *base* image: torch wheels bundle CUDA/cuDNN, so
  the cudnn-runtime base duplicated ~3 GB.
* Ubuntu 24.04 ffmpeg in the GPU image has `-hwaccel cuda`, `h264_nvenc`,
  `hevc_nvenc`, `scale_cuda`; `NVIDIA_DRIVER_CAPABILITIES` includes `video`.
* Weights are baked into `/models` at build time; no runtime download.

## Expected runtime (90-minute 4K30 match, minutes)

From `perf_profiles.estimate_runtime` (planning model; measure with
`bench/bench_match.py`):

| hardware class | fast | balanced | quality |
|---|---:|---:|---:|
| `rtx_4090` | 18 | 28 | 54 |
| `rtx_4080` | 20 | 32 | 73 |
| `rtx_3080` | 24 | 42 | 110 |
| `dgx_spark` | 26 | 41 | 88 |
| `apple_m2_ultra` | 30 | 57 | 157 |
| `apple_m1_max` | 46 | 103 | 334 |
| `cpu_8core` | 249 | 764 | 3110 |

Balanced, RTX 4080: proxy 6.0, detect+track 16.1, analysis 1.8, render 7.7,
clips+reel 0.5 = 32 min (PLAN.md budget: 30-40 min).

## Validate on a machine

```bash
curl -s localhost:8000/v1/health/gpu
python bench/bench_match.py --minutes 2 --height 2160 --profile balanced
```

The bench measures proxy fps, detection fps at the profile imgsz/batch on
the selected device, and ffmpeg crop+scale+encode fps, then projects a
90-minute match next to the class reference numbers.
