from __future__ import annotations

from pathlib import Path

import pytest

from backend.services import perf_profiles as pp


def test_profiles_match_artifacts_table() -> None:
    assert set(pp.PROFILES) == {"fast", "balanced", "quality"}
    fast, balanced, quality = pp.PROFILES["fast"], pp.PROFILES["balanced"], pp.PROFILES["quality"]
    assert (fast["proxy_height"], balanced["proxy_height"], quality["proxy_height"]) == (720, 1080, 1080)
    assert (fast["inference_imgsz"], balanced["inference_imgsz"], quality["inference_imgsz"]) == (960, 1280, 1536)
    assert (fast["vid_stride"], balanced["vid_stride"], quality["vid_stride"]) == (2, 1, 1)
    assert (fast["output_height"], balanced["output_height"], quality["output_height"]) == (1080, 1080, 1440)
    assert (fast["yolo_model"], balanced["yolo_model"], quality["yolo_model"]) == ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt")
    assert all(p["batch_size"] == "auto" for p in pp.PROFILES.values())
    assert all(p["debug_video"] is False for p in pp.PROFILES.values())
    assert (fast["ball_tiles"], balanced["ball_tiles"], quality["ball_tiles"]) == (False, False, True)
    assert all(set(p) == set(pp.PROFILE_KEYS) for p in pp.PROFILES.values())


def test_resolve_job_config_explicit_keys_win(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VH_MODEL_FAMILY", raising=False)
    cfg = pp.resolve_job_config({"profile": "fast", "inference_imgsz": 1280, "camera_mode": "follow_ball", "vid_stride": None})
    assert cfg["profile"] == "fast"
    assert cfg["inference_imgsz"] == 1280  # explicit wins
    assert cfg["vid_stride"] == 2  # None means "unset"
    assert cfg["proxy_height"] == 720
    assert cfg["camera_mode"] == "follow_ball"  # non-profile keys pass through
    assert cfg["profile_overrides"] == ["inference_imgsz"]


def test_resolve_job_config_default_and_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VH_MODEL_FAMILY", raising=False)
    cfg = pp.resolve_job_config({})
    assert cfg["profile"] == "balanced" and cfg["yolo_model"] == "yolov8s.pt"
    assert pp.resolve_job_config(None)["profile"] == "balanced"
    lenient = pp.resolve_job_config({"profile": "turbo"})
    assert lenient["profile"] == "balanced" and "profile_warning" in lenient
    with pytest.raises(ValueError):
        pp.resolve_job_config({"profile": "turbo"}, strict=True)
    # Does not mutate the input.
    original = {"profile": "quality"}
    pp.resolve_job_config(original)
    assert original == {"profile": "quality"}
    # PROFILES itself is not mutated by callers.
    cfg["proxy_height"] = 1
    assert pp.PROFILES["balanced"]["proxy_height"] == 1080


def test_model_family_swap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VH_MODEL_FAMILY", "yolo26")
    assert pp.resolve_job_config({"profile": "quality"})["yolo_model"] == "yolo26m.pt"
    # Explicit model is never swapped.
    assert pp.resolve_job_config({"profile": "quality", "yolo_model": "yolov8m.pt"})["yolo_model"] == "yolov8m.pt"
    assert pp.apply_model_family("/weights/best.pt", "yolo26") == "/weights/best.pt"
    assert pp.apply_model_family("yolov8n.pt", "bogus") == "yolov8n.pt"
    assert pp.model_size_letter("yolo26m.pt") == "m"
    assert pp.model_size_letter("custom.pt") == "s"


def test_resolve_model_path(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VH_MODEL_DIR", str(tmp_path))
    # Not present anywhere, dir writable -> download target inside VH_MODEL_DIR.
    assert pp.resolve_model_path("yolov8x.pt") == str(tmp_path / "yolov8x.pt")
    (tmp_path / "yolov8s.pt").write_bytes(b"x")
    assert pp.resolve_model_path("yolov8s.pt") == str(tmp_path / "yolov8s.pt")
    # Bundled repo weights are found when VH_MODEL_DIR lacks them.
    if (pp.REPO_ROOT / "yolov8n.pt").exists():
        found = pp.resolve_model_path("yolov8n.pt")
        assert Path(found).resolve() == (pp.REPO_ROOT / "yolov8n.pt").resolve()
    explicit = tmp_path / "best.pt"
    explicit.write_bytes(b"x")
    assert pp.resolve_model_path(str(explicit)) == str(explicit)
    monkeypatch.setenv("VH_MODEL_DIR", str(tmp_path / "missing"))
    assert pp.resolve_model_path("yolov8x.pt") == "yolov8x.pt"


def test_estimate_runtime_shape_and_ordering() -> None:
    est = pp.estimate_runtime(5400, "balanced", "rtx_4080")
    names = [stage["name"] for stage in est["stages"]]
    assert names == ["proxy", "detect_track", "analysis", "render", "clips_reel"]
    assert est["total_s"] == pytest.approx(sum(est["stages_s"].values()), abs=0.5)
    assert set(est["stages_s"]) == set(names)
    # PLAN.md budget: 90 min 4K30 on a 4080 lands at roughly 30-40 minutes.
    assert 25 <= est["total_min"] <= 45
    # Faster hardware is faster; quality costs more than fast.
    assert pp.estimate_runtime(5400, "balanced", "rtx_4090")["total_s"] < est["total_s"]
    assert pp.estimate_runtime(5400, "balanced", "cpu_8core")["total_s"] > est["total_s"]
    assert pp.estimate_runtime(5400, "fast", "rtx_4080")["total_s"] < est["total_s"] < pp.estimate_runtime(5400, "quality", "rtx_4080")["total_s"]
    # Apple Studio is 2-3x slower on detection than a 4080.
    apple = pp.estimate_runtime(5400, "balanced", "apple_m2_ultra")
    ratio = apple["stages_s"]["detect_track"] / est["stages_s"]["detect_track"]
    assert 1.7 <= ratio <= 3.2
    # Lower source resolution is cheaper to decode.
    assert pp.estimate_runtime(5400, "balanced", "rtx_4080", source_height=1080)["stages_s"]["proxy"] < est["stages_s"]["proxy"]


def test_estimate_runtime_every_class_and_profile() -> None:
    for hw in pp.hardware_classes():
        for profile in pp.profile_names():
            est = pp.estimate_runtime(600, profile, hw)
            assert est["total_s"] > 0
            assert all(stage["seconds"] >= 0 for stage in est["stages"])
    table = pp.runtime_table()
    assert set(table) == set(pp.HARDWARE_CLASSES)
    with pytest.raises(ValueError):
        pp.estimate_runtime(60, "balanced", "tpu_9000")
    with pytest.raises(ValueError):
        pp.estimate_runtime(60, "turbo", "rtx_4080")


def test_estimate_runtime_measured_and_config_overrides() -> None:
    measured = pp.estimate_runtime(600, "balanced", "cpu_8core", measured={"proxy_fps": 300, "detect_fps": 100, "render_fps": 200})
    stages = {stage["name"]: stage for stage in measured["stages"]}
    assert stages["proxy"]["fps"] == 300 and stages["render"]["fps"] == 200
    assert stages["proxy"]["basis"] == "measured"
    assert stages["proxy"]["seconds"] == pytest.approx(600 * 30 / 300, rel=1e-3)
    debug = pp.estimate_runtime(600, "balanced", "rtx_4080", config={"debug_video": True})
    assert "debug_video" in debug["stages_s"]
    strided = pp.estimate_runtime(600, "balanced", "rtx_4080", config={"vid_stride": 2})
    base = pp.estimate_runtime(600, "balanced", "rtx_4080")
    assert strided["stages_s"]["detect_track"] < base["stages_s"]["detect_track"]
    trt = pp.estimate_runtime(600, "balanced", "rtx_4080", tensorrt=True)
    assert trt["stages_s"]["detect_track"] < base["stages_s"]["detect_track"]


def test_suggest_batch_size() -> None:
    assert pp.suggest_batch_size("rtx_4080", 1280) == 16
    assert pp.suggest_batch_size("rtx_4080", 640) == 64
    assert pp.suggest_batch_size("cpu_8core", 1536) == 2
    assert 1 <= pp.suggest_batch_size("unknown", 1280) <= 64


@pytest.mark.parametrize(
    "status, expected",
    [
        ({"nvidia_smi": {"available": True, "gpus": [{"name": "NVIDIA GeForce RTX 4090", "memory_total_mb": 24564}]},
          "torch": {"cuda_available": True}}, "rtx_4090"),
        ({"nvidia_smi": {"available": True, "gpus": [{"name": "NVIDIA GeForce RTX 5080", "memory_total_mb": 16303}]},
          "torch": {"cuda_available": True}}, "rtx_4080"),
        ({"nvidia_smi": {"available": True, "gpus": [{"name": "NVIDIA GeForce RTX 3080", "memory_total_mb": 10240}]},
          "torch": {"cuda_available": True}}, "rtx_3080"),
        ({"nvidia_smi": {"available": True, "gpus": [{"name": "NVIDIA GB10", "memory_total_mb": None}]},
          "torch": {"cuda_available": True}, "platform": {"machine": "aarch64"}}, "dgx_spark"),
        ({"nvidia_smi": {"available": False, "gpus": []},
          "torch": {"cuda_available": True, "devices": []}, "platform": {"machine": "aarch64"}}, "dgx_spark"),
        ({"nvidia_smi": {"available": True, "gpus": [{"name": "Some Future GPU", "memory_total_mb": 32000}]},
          "torch": {"cuda_available": True}}, "rtx_4090"),
        ({"mps_available": True, "platform": {"cpu_brand": "Apple M2 Ultra"}}, "apple_m2_ultra"),
        ({"mps_available": True, "platform": {"cpu_brand": "Apple M4 Max"}}, "apple_m2_ultra"),
        ({"torch": {"mps_available": True}, "platform": {"cpu_brand": "Apple M1 Max"}}, "apple_m1_max"),
        ({"mps_available": True, "platform": {"cpu_brand": "Apple M2"}}, "apple_m1_max"),
        ({"torch": {"cuda_available": False}, "nvidia_smi": {"available": False, "gpus": []}}, "cpu_8core"),
        ({}, "cpu_8core"),
        (None, "cpu_8core"),
    ],
)
def test_classify_hardware(status, expected) -> None:
    assert pp.classify_hardware(status) == expected
