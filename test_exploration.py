"""Smoke test of the exploration report (python -m exploration) on a tiny sound + vibration dataset."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from exploration.explore import main
from exploration.signal_stats import separation_auc, signal_stats

SR = 8000


def test_signal_stats_flags_and_separation():
    t = np.arange(SR) / SR
    sine = np.sin(2 * np.pi * 100 * t)[None, :]
    stats, (f, p) = signal_stats(sine, SR)
    assert abs(stats["dominant_freq_hz"] - 100) < 5 and abs(stats["crest_factor"] - np.sqrt(2)) < 0.05
    assert stats["clipped_fraction"] < 1e-3 and not stats["is_silent"] and not stats["has_nan_or_inf"]
    clipped, _ = signal_stats(np.clip(3 * sine, -1, 1), SR)
    assert clipped["clipped_fraction"] > 0.1
    assert signal_stats(np.zeros((1, SR)), SR)[0]["is_silent"]
    assert separation_auc([3, 4, 1, 2], [1, 1, 0, 0]) == 1.0 and separation_auc([1, 2, 3, 4], [1, 1, 0, 0]) == 1.0


def test_exploration_report(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "rig"
    t = np.arange(SR) / SR
    for folder, n, fault in (("normal", 12, None), ("anomaly/imbalance", 4, "imbalance"), ("anomaly/bearing", 4, "bearing")):
        (root / "m" / folder).mkdir(parents=True)
        for i in range(n):
            base = np.sin(2 * np.pi * (30 + i) * t)
            extra = {"imbalance": 3 * base, "bearing": 0.5 * np.sin(2 * np.pi * 2000 * t)}.get(fault, 0)
            accel = [(g * (base + extra) + rng.normal(0, 0.1, SR)) for g in (1.0, 0.8, 0.6)]
            mic = 0.2 * base + rng.normal(0, 0.2, SR)
            np.savetxt(root / "m" / folder / f"{i}.csv", np.stack([*accel, mic], axis=1), delimiter=",", fmt="%.4f")
    shutil.copy(root / "m" / "normal" / "0.csv", root / "m" / "anomaly" / "bearing" / "copy.csv")  # a duplicate
    out = tmp_path / "exploration"

    main(["--config", "configs/folder.yaml", "--out", str(out), "--clips-per-group", "10", "--set",
          f"data_root={root}", f"cache_dir={tmp_path / 'cache'}", "folder.sample_rate=8000",
          "folder.vibration_channels=[0, 1, 2]", "folder.sound_channels=[3]", "folder.segment_seconds=0.5"])

    report = (out / "folder" / "report.md").read_text(encoding="utf-8")
    for heading in ("### Inventory", "### Sound", "### Vibration", "#### Separability", "#### 2-D map",
                    "#### Channel correlation", "byte-identical"):
        assert heading in report, heading
    figures = {p.name for p in (out / "folder" / "figures" / "m").iterdir()}
    for kind in ("waveforms", "spectrograms", "spectra", "stats", "embedding"):
        assert {f"{kind}_sound.png", f"{kind}_vibration.png"} <= figures
    assert "channels_vibration.png" in figures and "counts.png" in figures

    summary = json.loads((out / "folder" / "summary.json").read_text())
    vib = next(s for s in summary["machines"] if s["modality"] == "vibration")
    assert vib["fault_types"] == 2 and vib["channels"] == 3 and vib["best_stat_auc"] > 0.8
    stats = pd.read_csv(out / "folder" / "clip_stats.csv")
    assert set(stats["modality"]) == {"sound", "vibration"} and set(stats["class"]) == {"normal", "imbalance", "bearing"}
    overview = pd.read_csv(out / "overview.csv")
    assert len(overview) == 2 and set(overview["dataset"]) == {"folder"}
