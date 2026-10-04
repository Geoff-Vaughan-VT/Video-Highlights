# bench

`bench_match.py` times the real pipeline stages on this machine and projects a
90-minute match. See the module docstring for flags.

```bash
python bench/bench_match.py --minutes 2 --height 2160            # 4K30 synthetic, balanced profile
python bench/bench_match.py --minutes 2 --height 2160 --profile fast
python bench/bench_match.py --source /media/match.mp4 --minutes 3 # real footage (first 3 min)
docker compose --profile gpu run --rm worker-gpu python bench/bench_match.py --minutes 2 --height 2160
```

Results land in `bench/results/<UTC timestamp>.json|.md` (git-ignored). The
projection uses `backend/services/perf_profiles.estimate_runtime` with the
measured proxy / detect / render fps; stages that could not run fall back to
the hardware-class reference numbers, shown side by side.
