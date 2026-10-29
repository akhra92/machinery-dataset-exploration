"""Explore datasets before training: what is in them, what the signals look like, and how hard they look.

Each dataset is read through the same loaders, train/test split and log-mel settings as training (its config in
configs/), so the report shows exactly what the models will get. For every machine type and sensor:
  - inventory: clips and files per split / class / domain, fault types, sample rates, channels, durations
  - quality: unreadable files, NaN/Inf, silent clips, clipping, DC offset, duplicate files, train/test leakage
  - examples: waveforms and log-mel spectrograms (the model input) of normal clips and each fault type
  - spectra: average power spectrum and average log-mel band level per class
  - statistics: RMS, crest factor, kurtosis, spectral centroid, ... per class, and how well each one separates
    normal from faulty clips on its own (a rough difficulty estimate before any training)
  - distribution shift: normal train vs normal test clips, source vs target domain
  - 2-D maps (PCA, t-SNE) of clip features; channel correlation for multi-channel sensors
Statistics and figures use a random sample of clips per (split, class, domain) group (--clips-per-group);
counts, file hashes and fault lists cover every clip.

Output: <out>/<dataset>/report.md (+ figures/, clip_stats.csv, summary.json) and <out>/overview.md comparing every
dataset explored so far.

    python -m exploration --all                                      # every config whose data is downloaded
    python -m exploration --config configs/mimii.yaml configs/paderborn.yaml
    python -m exploration --config configs/mafaulda.yaml --clips-per-group 50 --set data_root=data/other
"""
import argparse
import hashlib
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from msad.data import get_dataset
from msad.features import spectrogram
from msad.run import load_config

from . import plots
from .signal_stats import STATS, separation_auc, signal_stats

NAMES = {"default": "dcase2024"}   # config file stem -> dataset name in the report
TEMPLATES = {"folder"}             # configs that are templates, not datasets (skipped by --all)
BOX_STATS = ["rms", "crest_factor", "kurtosis", "skewness", "zero_crossings_per_s", "spectral_centroid_hz",
             "spectral_bandwidth_hz", "spectral_rolloff_hz", "dominant_freq_hz", "spectral_flatness"]


# ---------------------------------------------------------------- helpers

def clip_stats(spec: np.ndarray) -> np.ndarray:
    """One vector per clip: mean and std of each log-mel band over time (for the 2-D map)."""
    return np.concatenate([spec.mean(axis=1), spec.std(axis=1)]).astype(np.float32)


def dataset_name(config: Path) -> str:
    return NAMES.get(config.stem, config.stem)


def clip_class(clip) -> str:
    if clip.label is None:
        return "unlabelled"
    return "normal" if clip.label == 0 else (clip.fault or "anomaly")


def label_name(label) -> str:
    return "unlabelled" if label is None else ("normal", "anomaly")[label]


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fmt(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "-"
        return f"{v:.3g}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def md_table(headers: list[str], rows: list[list]) -> str:
    def cell(v):
        return fmt(v).replace("|", "\\|")
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def class_order(classes) -> list[str]:
    return sorted(set(classes), key=lambda c: (c != "normal", c == "unlabelled", c))


# ---------------------------------------------------------------- one machine type

class MachineExplorer:
    def __init__(self, ds, cfg: dict, machine: str, out: Path, args, rng: np.random.Generator):
        self.ds, self.cfg, self.machine, self.args, self.rng = ds, cfg, machine, args, rng
        self.fig_dir = out / "figures" / machine
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.out = out
        self.issues: list[str] = []
        self.notes: list[str] = []

    def fig(self, name: str) -> str:
        return (self.fig_dir / name).relative_to(self.out).as_posix()

    def run(self) -> tuple[list[str], list[dict], pd.DataFrame]:
        md = [f"## {self.machine}", ""]
        clips = {}
        for split in ("train", "test"):
            try:
                clips[split] = self.ds.clips(self.machine, split)
            except Exception as e:
                self.issues.append(f"cannot list {split} clips: {e}")
                clips[split] = []
        if not any(clips.values()):
            md += ["**No clips found.** " + "; ".join(self.issues), ""]
            return md, [], pd.DataFrame()

        groups: dict[tuple[str, str, str], list] = defaultdict(list)
        for split, items in clips.items():
            for c in items:
                groups[(split, clip_class(c), c.domain or "")].append(c)
        count_rows = [{"split": s, "class": k, "domain": d, "clips": len(v), "files": len({c.path for c in v}),
                       "label": label_name(v[0].label)} for (s, k, d), v in sorted(groups.items())]
        md += self.inventory(clips, count_rows)
        md += self.file_checks(clips)

        sampled = []
        for key in sorted(groups):
            items = groups[key]
            take = sorted(self.rng.choice(len(items), min(self.args.clips_per_group, len(items)), replace=False))
            sampled += [(key[0], items[i]) for i in take]

        summaries, frames = [], []
        for modality in self.ds.modalities:
            section, summary, df = self.explore_modality(modality, sampled, clips, count_rows)
            md += section
            if summary:
                summaries.append(summary)
            frames.append(df)
        for summary in summaries:
            summary["issues"] = len(self.issues)
        md[2:2] = self.issue_block()
        return md, summaries, pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # -- inventory and file-level checks ----------------------------------------------------------------

    def inventory(self, clips: dict, count_rows: list[dict]) -> list[str]:
        plots.counts(count_rows, self.fig_dir / "counts.png", f"{self.machine}: clips per split / class / domain")
        train_labels = {c.label for c in clips["train"]}
        if 1 in train_labels:
            self.notes.append("training split lists anomalous clips; training uses normal clips only")
        test = clips["test"]
        n_normal_test = sum(c.label == 0 for c in test)
        n_anomaly = sum(c.label == 1 for c in test)
        if test and n_normal_test < 20:
            self.issues.append(f"only {n_normal_test} normal test clips: AUC will be noisy, compare several seeds")
        if test and n_anomaly == 0:
            self.issues.append("no labelled anomalies in the test split: AUC cannot be computed")
        if not test:
            self.issues.append("test split is empty")
        return ["### Inventory", "",
                md_table(["split", "class", "domain", "clips", "files"],
                         [[r["split"], r["class"], r["domain"] or "-", r["clips"], r["files"]] for r in count_rows]),
                "", f"![clip counts]({self.fig('counts.png')})", ""]

    def file_checks(self, clips: dict) -> list[str]:
        files = {split: sorted({c.path for c in items}) for split, items in clips.items()}
        missing = [p for ps in files.values() for p in ps if not Path(p).exists()]
        empty = [p for ps in files.values() for p in ps if Path(p).exists() and Path(p).stat().st_size == 0]
        for name, bad in (("missing", missing), ("empty (0 bytes)", empty)):
            if bad:
                self.issues.append(f"{len(bad)} {name} file(s), e.g. {bad[0]}")
        lines = ["### File checks", ""]
        if self.args.no_hash:
            return lines + ["Duplicate / leakage check skipped (--no-hash).", ""]
        hashes: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
        all_files = [(s, p) for s, ps in files.items() for p in ps if p not in missing]
        for split, p in tqdm(all_files, desc=f"{self.machine} hashing", leave=False):
            hashes[file_md5(Path(p))][split].append(p)
        dup = {h: v for h, v in hashes.items() if sum(len(x) for x in v.values()) > 1}
        leak = {h: v for h, v in dup.items() if len(v) > 1}
        n_dup_files = sum(sum(len(x) for x in v.values()) for v in dup.values())
        if leak:
            example = " = ".join(Path(p).name for ps in next(iter(leak.values())).values() for p in ps[:1])
            self.issues.append(f"train/test leakage: {len(leak)} recording(s) appear in both splits "
                               f"(identical bytes), e.g. {example}")
        if dup:
            self.issues.append(f"{n_dup_files} files are byte-identical copies of another file ({len(dup)} groups)")
        return lines + [f"{len(all_files):,} files hashed (MD5): {len(dup)} duplicate group(s), "
                        f"{len(leak)} shared between train and test.", ""]

    def issue_block(self) -> list[str]:
        lines = ["**Issues:** none found." if not self.issues else "**Issues:**"]
        lines += [f"- ⚠ {i}" for i in self.issues] + [f"- note: {n}" for n in self.notes] + [""]
        return lines

    # -- one sensor ------------------------------------------------------------------------------------

    def explore_modality(self, modality: str, sampled, clips: dict, count_rows: list[dict]):
        fcfg = self.cfg["features"][modality]
        rows, vectors, band_means, psds, examples, corrs, errors = [], [], [], [], defaultdict(list), [], []
        freq = None
        for split, clip in tqdm(sampled, desc=f"{self.machine} {modality}", leave=False):
            try:
                sig, sr = self.ds.load(clip, modality)
                stats, (f, p) = signal_stats(sig, sr)
                spec = spectrogram(sig, sr, fcfg)
            except Exception as e:
                errors.append(f"{Path(clip.path).name}: {type(e).__name__}: {e}")
                continue
            if freq is None:
                freq = f
            psds.append(p if f.shape == freq.shape and np.allclose(f, freq) else np.interp(freq, f, p))
            vectors.append(clip_stats(spec))
            band_means.append(spec.mean(axis=1))
            cls = clip_class(clip)
            rows.append({"machine": self.machine, "modality": modality, "split": split, "class": cls,
                         "label": label_name(clip.label), "domain": clip.domain or "", "fault": clip.fault,
                         "file": str(clip.path), "segment": clip.segment, **stats})
            # examples: normal clips from training, faults from test
            wanted = (split == "train") if cls == "normal" else True
            if wanted and len(examples[cls]) < self.args.examples:
                name = Path(clip.path).name + (f" [{clip.segment}]" if clip.segment is not None else "")
                examples[cls].append({"signal": np.atleast_2d(sig), "sr": sr, "spec": spec, "name": name,
                                      "n_channels": np.atleast_2d(sig).shape[0]})
            if clip.label == 0 and np.atleast_2d(sig).shape[0] > 1 and len(corrs) < 50:
                with np.errstate(invalid="ignore", divide="ignore"):
                    corrs.append(np.nan_to_num(np.corrcoef(np.atleast_2d(sig))))
        title = f"{self.machine} · {modality}"
        md = [f"### {modality.capitalize()}", ""]
        if errors:
            self.issues.append(f"{modality}: {len(errors)} of {len(sampled)} sampled clips could not be read, "
                               f"e.g. {errors[0]}")
        if not rows:
            return md + ["No readable clips.", ""], {}, pd.DataFrame()
        df = pd.DataFrame(rows)
        md += self.signal_properties(df, modality, count_rows)

        tag = modality
        ordered = [(c, examples[c]) for c in class_order(examples) if examples[c]][: self.args.max_example_rows]
        plots.examples(ordered, self.fig_dir / f"waveforms_{tag}.png", f"{title}: examples", "wave")
        plots.examples(ordered, self.fig_dir / f"spectrograms_{tag}.png", f"{title}: examples", "spec")
        md += ["#### Examples", "", f"![waveforms]({self.fig(f'waveforms_{tag}.png')})", "",
               f"![spectrograms]({self.fig(f'spectrograms_{tag}.png')})", ""]

        classes = df["class"].to_numpy()
        psd_by = {c: np.mean([psds[i] for i in np.flatnonzero(classes == c)], axis=0) for c in class_order(classes)}
        mel_by = {c: np.mean([band_means[i] for i in np.flatnonzero(classes == c)], axis=0)
                  for c in class_order(classes)}
        plots.spectra(freq, psd_by, mel_by, self.fig_dir / f"spectra_{tag}.png", f"{title}: average spectra per class")
        md += ["#### Average spectra", "",
               "Where the fault curves leave the normal (blue) curve is where a model can see the fault.", "",
               f"![spectra]({self.fig(f'spectra_{tag}.png')})", ""]

        plots.stat_boxes(df, BOX_STATS, self.fig_dir / f"stats_{tag}.png", f"{title}: statistics per class")
        md += self.statistics_tables(df)
        md += [f"![statistics]({self.fig(f'stats_{tag}.png')})", ""]

        summary = self.summary(df, modality, count_rows)
        md += self.separability(df, summary)
        md += self.shift(df, summary)
        md += self.embedding(np.array(vectors), df, tag, title)
        if corrs:
            corr = np.mean(corrs, axis=0)
            plots.channel_correlation(corr, self.fig_dir / f"channels_{tag}.png", f"{title}: channel correlation")
            off = corr[~np.eye(len(corr), dtype=bool)]
            md += ["#### Channel correlation (normal clips)", "",
                   f"Mean |correlation| between channels: {np.abs(off).mean():.2f} "
                   "(near 1 = the channels carry the same information).", "",
                   f"![channels]({self.fig(f'channels_{tag}.png')})", ""]
        return md, summary, df

    def signal_properties(self, df: pd.DataFrame, modality: str, count_rows: list[dict]) -> list[str]:
        rates = sorted(df["sample_rate"].unique())
        chans = sorted(df["n_channels"].unique())
        dur = df["duration_s"]
        if len(rates) > 1:
            self.issues.append(f"{modality}: mixed sample rates {rates}")
        if dur.max() > 1.05 * dur.min():
            self.notes.append(f"{modality}: clip durations vary from {dur.min():.2f} to {dur.max():.2f} s")
        flags = {"NaN / Inf values": df["has_nan_or_inf"].sum(), "silent (all zero)": df["is_silent"].sum(),
                 "clipped (flat tops > 0.1% of samples)": (df["clipped_fraction"] > 1e-3).sum()}
        for what, n in flags.items():
            if n:
                self.issues.append(f"{modality}: {n} of {len(df)} sampled clips {what}")
        high_dc = (df["dc_offset_ratio"] > 1).mean()
        if high_dc > 0.5:
            self.notes.append(f"{modality}: DC offset larger than the signal's RMS in {high_dc:.0%} of clips "
                              "(sensor bias or gravity; the spectrogram's lowest band absorbs it)")
        fcfg = self.cfg["features"][modality]
        model_sr = fcfg.get("sample_rate") or rates[0]
        n_clips = sum(r["clips"] for r in count_rows)
        hours = n_clips * dur.median() / 3600
        rows = [["sample rate (Hz)", ", ".join(fmt(r) for r in rates)],
                ["channels", ", ".join(str(c) for c in chans)],
                ["clip duration (s)", f"{dur.median():.3g} (min {dur.min():.3g}, max {dur.max():.3g})"],
                ["total audio / signal", f"{hours:.2f} h over {n_clips:,} clips"],
                ["model input", f"{fcfg['n_mels']} mel bands x {chans[0]} channel(s), {fmt(model_sr)} Hz, "
                                f"up to {fmt(min(fcfg.get('fmax') or model_sr / 2, model_sr / 2))} Hz"],
                ["clips analysed", f"{len(df):,} (up to {self.args.clips_per_group} per split/class/domain)"]]
        return ["#### Signal properties", "", md_table(["", ""], rows), ""]

    def statistics_tables(self, df: pd.DataFrame) -> list[str]:
        classes = class_order(df["class"])
        med = df.groupby("class")[list(STATS)].median().reindex(classes)
        rows = [[stat] + [med.loc[c, stat] for c in classes] for stat in STATS]
        return ["#### Statistics per class (median over analysed clips)", "",
                md_table(["statistic"] + classes, rows), ""]

    def separability(self, df: pd.DataFrame, summary: dict) -> list[str]:
        labelled = df[df["label"] != "unlabelled"]
        y = (labelled["label"] == "anomaly").astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            return ["#### Separability", "", "Needs both normal and anomalous clips.", ""]
        aucs = {s: separation_auc(labelled[s], y) for s in STATS}
        ranked = sorted(aucs.items(), key=lambda kv: -np.nan_to_num(kv[1]))
        summary["best_stat"], summary["best_stat_auc"] = ranked[0]
        rows = [[s, a, STATS[s]] for s, a in ranked[:6]]
        faults = [c for c in class_order(labelled["class"]) if c != "normal"]
        fault_rows = []
        normal = labelled[labelled["label"] == "normal"]
        for fault in faults:
            part = pd.concat([normal, labelled[labelled["class"] == fault]])
            yy = (part["class"] == fault).astype(int).to_numpy()
            fa = {s: separation_auc(part[s], yy) for s in STATS}
            best = max(fa, key=lambda s: np.nan_to_num(fa[s]))
            fault_rows.append([fault, best, fa[best], (labelled["class"] == fault).sum()])
        summary["hardest_fault"] = min(fault_rows, key=lambda r: r[2])[0] if fault_rows else ""
        return ["#### Separability: can one number tell normal from faulty?", "",
                "AUC of a single statistic as the anomaly score (0.5 = no better than chance, 1.0 = perfect "
                "split). High values mean an easy dataset; the models should beat the best one.", "",
                md_table(["statistic", "AUC", "meaning"], rows), "",
                "Best single statistic per fault type (normal vs that fault):", "",
                md_table(["fault", "best statistic", "AUC", "clips analysed"], fault_rows), ""]

    def shift(self, df: pd.DataFrame, summary: dict) -> list[str]:
        normal = df[df["label"] == "normal"]
        lines = ["#### Distribution shift among normal clips", "",
                 "AUC of a statistic for telling two groups of normal clips apart. Near 0.5 = same distribution. "
                 "High train-vs-test values mean test normals look different from training (more false alarms); "
                 "high source-vs-target values are expected in domain-shift datasets.", ""]
        rows = []
        for name, column, a, b in (("train vs test", "split", "train", "test"),
                                   ("source vs target", "domain", "source", "target")):
            part = normal[normal[column].isin([a, b])]
            if part[column].nunique() < 2:
                continue
            yy = (part[column] == b).astype(int).to_numpy()
            aucs = {s: separation_auc(part[s], yy) for s in STATS}
            worst = max(aucs, key=lambda s: np.nan_to_num(aucs[s]))
            rows.append([name, worst, aucs[worst], (part[column] == a).sum(), (part[column] == b).sum()])
            summary[f"shift_{column}_auc"] = aucs[worst]
            if name == "train vs test" and aucs[worst] > 0.8:
                self.issues.append(f"normal test clips differ from training ({worst}: AUC {aucs[worst]:.2f}); "
                                   "check the split (e.g. different recording sessions or operating conditions)")
        if not rows:
            return lines + ["Not enough groups to compare.", ""]
        return lines + [md_table(["comparison", "most different statistic", "AUC", "n (first)", "n (second)"],
                                 rows), ""]

    def embedding(self, vectors: np.ndarray, df: pd.DataFrame, tag: str, title: str) -> list[str]:
        if len(vectors) < 5:
            return []
        x = StandardScaler().fit_transform(np.nan_to_num(vectors))
        points = {"PCA": PCA(n_components=2, random_state=0).fit_transform(x)}
        if len(x) >= 15:
            perplexity = float(min(30, (len(x) - 1) / 3))
            points["t-SNE"] = TSNE(n_components=2, perplexity=perplexity, init="pca",
                                   random_state=self.args.seed).fit_transform(x)
        labels = np.array([None if l == "unlabelled" else int(l == "anomaly") for l in df["label"]], dtype=object)
        use_domain = df.loc[df["label"] == "normal", "domain"].nunique() > 1
        group = df["domain"].to_numpy() if use_domain else df["split"].to_numpy()
        plots.embedding(points, labels, group, "domain" if use_domain else "split",
                        self.fig_dir / f"embedding_{tag}.png", f"{title}: 2-D map of clips")
        return ["#### 2-D map of clips", "",
                "If anomalies (orange) form their own region, the autoencoder should do well; if they "
                "sit inside the normal cloud, the dataset is hard.", "",
                f"![embedding]({self.fig(f'embedding_{tag}.png')})", ""]

    def summary(self, df: pd.DataFrame, modality: str, count_rows: list[dict]) -> dict:
        def count(**kw):
            return sum(r["clips"] for r in count_rows if all(r[k] == v for k, v in kw.items()))
        faults = sorted({r["class"] for r in count_rows if r["label"] == "anomaly"})
        return {
            "machine": self.machine, "modality": modality,
            "sample_rate": ", ".join(fmt(r) for r in sorted(df["sample_rate"].unique())),
            "channels": int(df["n_channels"].max()),
            "clip_seconds": float(df["duration_s"].median()),
            "train_clips": count(split="train", label="normal"),
            "test_normal": count(split="test", label="normal"),
            "test_anomaly": count(split="test", label="anomaly"),
            "unlabelled": count(label="unlabelled"),
            "fault_types": len(faults), "faults": faults,
            "domains": ", ".join(sorted({r["domain"] for r in count_rows if r["domain"]})) or "-",
            "hours": sum(r["clips"] for r in count_rows) * float(df["duration_s"].median()) / 3600,
            "best_stat": "", "best_stat_auc": float("nan"),
        }


# ---------------------------------------------------------------- one dataset

def explore_dataset(config: Path, overrides: list[str], args) -> Path | None:
    cfg = load_config(config, overrides)
    name = dataset_name(config)
    if not Path(cfg["data_root"]).exists():
        print(f"skipping {name}: no data under {cfg['data_root']}")
        return None
    try:
        ds = get_dataset(cfg)
        machines = ds.machines()
    except Exception as e:
        print(f"skipping {name}: {e}")
        return None
    if args.machine:
        machines = [m for m in machines if m in args.machine]
    if not machines:
        print(f"skipping {name}: no data under {cfg['data_root']}")
        return None

    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    print(f"{name}: {len(machines)} machine type(s), sensors: {', '.join(ds.modalities)}")
    sections, summaries, frames = [], [], []
    for machine in machines:
        explorer = MachineExplorer(ds, cfg, machine, out, args, rng)
        try:
            md, summ, df = explorer.run()
        except Exception:
            md, summ, df = [f"## {machine}", "", "**Exploration failed:**", "", "```",
                            traceback.format_exc(), "```", ""], [], pd.DataFrame()
        sections += md
        summaries += summ
        frames.append(df)
        print(f"  {machine}: {len(explorer.issues)} issue(s)")

    stats = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    stats.to_csv(out / "clip_stats.csv", index=False)
    summary = {"dataset": name, "config": config.as_posix(), "data_root": str(cfg["data_root"]),
               "sensors": list(ds.modalities), "seconds": round(time.time() - t0, 1), "machines": summaries}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    head = [f"# Dataset exploration: {name}", "",
            f"Config `{config.as_posix()}`, data `{cfg['data_root']}`, sensors: {', '.join(ds.modalities)}. "
            f"Generated {time.strftime('%Y-%m-%d %H:%M')} in {summary['seconds']} s; statistics from up to "
            f"{args.clips_per_group} random clips per split / class / domain (seed {args.seed}).", "",
            "## Summary", "", summary_table(summaries, with_dataset=False), "",
            "How to read it: *best single-stat AUC* is how well one simple number (RMS, kurtosis, ...) already "
            "separates normal from faulty clips; near 1.0 means an easy dataset, near 0.5 a hard one. "
            "*Shift* is the same measure between groups of normal clips (train vs test); near 0.5 is good.", ""]
    (out / "report.md").write_text("\n".join(head + sections), encoding="utf-8")
    print(f"  -> {out / 'report.md'}")
    return out


def summary_table(summaries: list[dict], with_dataset: bool) -> str:
    headers = (["dataset"] if with_dataset else []) + [
        "machine", "sensor", "Hz", "ch", "clip s", "train", "test normal", "test anomaly", "fault types",
        "domains", "hours", "best single-stat AUC", "shift train/test", "issues"]
    rows = []
    for s in summaries:
        rows.append(([s.get("dataset", "")] if with_dataset else []) + [
            s["machine"], s["modality"], s["sample_rate"], s["channels"], s["clip_seconds"], s["train_clips"],
            s["test_normal"], s["test_anomaly"], s["fault_types"], s["domains"], s["hours"],
            f"{fmt(s['best_stat_auc'])} ({s['best_stat']})" if s["best_stat"] else "-",
            s.get("shift_split_auc", float("nan")), s.get("issues", 0)])
    return md_table(headers, rows)


def write_overview(out_root: Path) -> Path:
    """Combine the summary.json of every dataset explored into <out>/overview.md and overview.csv."""
    summaries = []
    for path in sorted(out_root.glob("*/summary.json")):
        data = json.loads(path.read_text())
        for s in data["machines"]:
            summaries.append({"dataset": data["dataset"], **s})
    flat = pd.DataFrame([{k: (", ".join(v) if isinstance(v, list) else v) for k, v in s.items()} for s in summaries])
    flat.to_csv(out_root / "overview.csv", index=False)
    lines = ["# Dataset overview", "",
             "One row per dataset, machine type and sensor. Details and figures: `<dataset>/report.md`.", "",
             summary_table(summaries, with_dataset=True), "",
             "Things to compare when choosing datasets:", "",
             "- **Size**: train clips and test anomalies (fewer than ~20 normal test clips gives noisy AUCs).",
             "- **Fault coverage**: number of fault types, and whether they match faults you expect in your factory.",
             "- **Difficulty**: best single-stat AUC near 1.0 is easy; you learn more from datasets where it is lower.",
             "- **Sensors**: sound and vibration recorded together (fusion) is closest to the final system.",
             "- **Shift**: domains, or high train/test shift, show how the model copes with changed conditions.",
             "- **Issues**: leakage, duplicates, clipping or unreadable files must be fixed or excluded first.",
             "- **License**: non-commercial (e.g. DCASE) limits later product use.", ""]
    (out_root / "overview.md").write_text("\n".join(lines), encoding="utf-8")
    return out_root / "overview.md"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m exploration", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", default=[], help="config file(s) of the dataset(s) to explore")
    parser.add_argument("--all", action="store_true", help="every config in configs/ whose data is present")
    parser.add_argument("--machine", nargs="+", help="only these machine types (default: all)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    parser.add_argument("--out", default="outputs/exploration")
    parser.add_argument("--clips-per-group", type=int, default=30,
                        help="clips analysed per split / class / domain group (default 30)")
    parser.add_argument("--examples", type=int, default=3, help="example clips per class in the figures")
    parser.add_argument("--max-example-rows", type=int, default=10, help="classes shown in the example figures")
    parser.add_argument("--no-hash", action="store_true", help="skip the duplicate / leakage check (faster)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    configs = [Path(c) for c in args.config]
    if args.all:
        configs += [p for p in sorted(Path("configs").glob("*.yaml")) if p.stem not in TEMPLATES]
    if not configs:
        parser.error("give --config FILE ... or --all")
    if args.set and len(configs) > 1:
        parser.error("--set applies to one config at a time")

    explored = [explore_dataset(c, args.set, args) for c in dict.fromkeys(configs)]
    if not any(explored):
        print("nothing explored: download / convert a dataset first (see README, Data)")
        return
    print(f"overview -> {write_overview(Path(args.out))}")
