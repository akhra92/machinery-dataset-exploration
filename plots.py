"""Figures for the exploration report (matplotlib, saved as PNG).

Colors: normal is always blue and faults take the next categorical slots in a fixed order, so a fault keeps its
color across figures. More than 7 fault types fold into "other faults". Scatter plots use at most three colors
(normal / anomaly / unlabelled), the most that stay distinguishable for colour-blind readers.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NORMAL, ANOMALY = SERIES[0], SERIES[1]
MUTED, INK, INK2, GRID, AXIS, SURFACE = "#898781", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
OTHER = "other faults"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.fontsize": 8, "legend.frameon": False, "lines.linewidth": 1.5,
    "figure.titlesize": 11, "font.size": 9,
})


def class_colors(classes: list[str]) -> dict[str, str]:
    """normal -> blue, faults -> following slots in sorted order; the 8th fault onward -> OTHER (muted)."""
    faults = [c for c in classes if c not in ("normal", "unlabelled")]
    colors = {"normal": NORMAL, "unlabelled": MUTED}
    for i, c in enumerate(sorted(faults)):
        colors[c] = SERIES[1 + i] if i < len(SERIES) - 2 or len(faults) == len(SERIES) - 1 else MUTED
    return colors


def fold_classes(classes: list[str]) -> dict[str, str]:
    """Map each class to the series it is drawn as (itself, or OTHER past the palette's 8 slots)."""
    colors = class_colors(classes)
    return {c: (OTHER if colors[c] == MUTED and c not in ("unlabelled",) else c) for c in classes}


def _save(fig, path):
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def counts(rows: list[dict], path, title: str) -> None:
    """Horizontal bars: clips per split / class / domain."""
    rows = sorted(rows, key=lambda r: (r["split"] != "train", r["class"] != "normal", r["class"], r["domain"]))
    labels = [f"{r['split']} · {r['class']}" + (f" · {r['domain']}" if r["domain"] else "") for r in rows]
    values = [r["clips"] for r in rows]
    colors = [NORMAL if r["class"] == "normal" else MUTED if r["class"] == "unlabelled" else ANOMALY for r in rows]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(rows) + 1.0))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, values, color=colors, height=0.7, edgecolor=SURFACE, linewidth=1)
    for yi, v in zip(y, values):
        ax.text(v, yi, f" {v}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("clips")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_title(title, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (NORMAL, ANOMALY)]
    ax.legend(handles, ["normal", "anomaly"], loc="lower right")
    _save(fig, path)


def examples(groups: list[tuple[str, list[dict]]], path, title: str, kind: str) -> None:
    """Grid: one row per class, one column per example. kind = 'wave' (raw signal) or 'spec' (model input)."""
    n_rows, n_cols = len(groups), max(len(ex) for _, ex in groups)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 1.7 * n_rows + 0.6), squeeze=False)
    for r, (cls, items) in enumerate(groups):
        color = NORMAL if cls == "normal" else MUTED if cls == "unlabelled" else ANOMALY
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(items):
                ax.axis("off")
                continue
            ex = items[c]
            if kind == "wave":
                sig, sr = ex["signal"], ex["sr"]
                t = np.arange(sig.shape[1]) / sr
                step = max(1, sig.shape[1] // 20000)  # plotting every sample of long clips is slow and identical
                for ch in range(min(3, sig.shape[0])):
                    ax.plot(t[::step], sig[ch, ::step], color=color, lw=0.6, alpha=1.0 if ch == 0 else 0.45)
                ax.set_xlim(0, t[-1] if t.size else 1)
                ax.grid(False)
                if r == n_rows - 1:
                    ax.set_xlabel("time (s)")
            else:
                spec = ex["spec"]
                lo, hi = np.percentile(spec, [2, 98])  # a few extreme bins would otherwise wash out the rest
                ax.imshow(spec, origin="lower", aspect="auto", cmap="Blues", interpolation="nearest", vmin=lo, vmax=hi)
                for b in range(1, ex["n_channels"]):  # stacked channels (e.g. 3 accelerometer axes)
                    ax.axhline(b * spec.shape[0] / ex["n_channels"] - 0.5, color=SURFACE, lw=1)
                ax.grid(False)
                ax.set_xticks([])
            ax.set_yticks([]) if kind == "spec" else None
            ax.set_title(ex["name"], fontsize=7, color=MUTED, loc="left")
            if c == 0:
                ax.set_ylabel(cls, color=INK, fontsize=8, rotation=0, ha="right", va="center")
    note = ("raw signal, first channel (other channels lighter)" if kind == "wave"
            else "log-mel spectrogram as the models see it (bands bottom to top, time left to right; "
                 "channels stacked, separated by white lines; color scale per image, compare levels in the spectra)")
    fig.suptitle(f"{title}\n{note}", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def spectra(freq: np.ndarray, psd: dict[str, np.ndarray], logmel: dict[str, np.ndarray], path, title: str) -> None:
    """Left: average power spectrum (Hz, log scale) per class. Right: average log-mel band energy per class."""
    colors = class_colors(list(psd))
    fold = fold_classes(list(psd))
    order = sorted(psd, key=lambda c: (c != "normal", fold[c] == OTHER, c))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    drawn = set()
    for cls in order:
        label = fold[cls] if fold[cls] not in drawn else None
        drawn.add(fold[cls])
        lw = 2.0 if cls == "normal" else 1.2
        a1.plot(freq[1:], 10 * np.log10(psd[cls][1:] + 1e-30), color=colors[cls], lw=lw, label=label)
        a2.plot(logmel[cls], color=colors[cls], lw=lw, label=label)
    a1.set_xscale("log")
    a1.set_xlabel("frequency (Hz)")
    a1.set_ylabel("power (dB)")
    a1.set_title("Average power spectrum", loc="left")
    a2.set_xlabel("log-mel band (channels stacked)")
    a2.set_ylabel("mean level (dB)")
    a2.set_title("Average model input per band", loc="left")
    a2.legend(loc="best", ncols=2 if len(drawn) > 4 else 1)
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, path)


def stat_boxes(df, stats: list[str], path, title: str) -> None:
    """Box plot of each statistic per class (normal first)."""
    classes = sorted(df["class"].unique(), key=lambda c: (c != "normal", c == "unlabelled", c))
    n_cols = 4
    n_rows = int(np.ceil(len(stats) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 2.6 * n_rows), squeeze=False)
    for ax, stat in zip(axes.flat, stats):
        data = [df.loc[df["class"] == c, stat].dropna().to_numpy() for c in classes]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=True,
                        flierprops={"markersize": 2, "markeredgecolor": MUTED},
                        medianprops={"color": INK, "lw": 1.2})
        for patch, c in zip(bp["boxes"], classes):
            patch.set_facecolor(NORMAL if c == "normal" else MUTED if c == "unlabelled" else ANOMALY)
            patch.set_alpha(0.75)
            patch.set_edgecolor(SURFACE)
        ax.set_xticks(range(1, len(classes) + 1), classes, rotation=35, ha="right", fontsize=7)
        ax.set_title(stat, loc="left", fontsize=9)
        if stat in ("rms", "peak", "dominant_freq_hz", "spectral_centroid_hz") and (df[stat] > 0).all():
            ax.set_yscale("log")
    for ax in list(axes.flat)[len(stats):]:
        ax.axis("off")
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, path)


def embedding(points: dict[str, np.ndarray], label: np.ndarray, group: np.ndarray, group_name: str,
              path, title: str) -> None:
    """2-D maps (PCA, t-SNE) of the clip-level features. Row 1: normal vs anomaly. Row 2: train vs test or
    source vs target (normal clips only), to spot distribution shift."""
    fig, axes = plt.subplots(2, len(points), figsize=(5.2 * len(points), 8.4), squeeze=False)
    for col, (name, xy) in enumerate(points.items()):
        ax = axes[0, col]
        for value, color, text in ((None, MUTED, "unlabelled"), (0, NORMAL, "normal"), (1, ANOMALY, "anomaly")):
            mask = np.array([l is value if value is None else l == value for l in label])
            if mask.any():
                ax.scatter(xy[mask, 0], xy[mask, 1], s=16, color=color, label=text, alpha=0.8,
                           edgecolors=SURFACE, linewidths=0.4)
        ax.set_title(f"{name}: normal vs anomaly", loc="left")
        ax.legend(loc="best")
        ax = axes[1, col]
        normal = np.array([l == 0 for l in label])
        ax.scatter(xy[~normal, 0], xy[~normal, 1], s=10, color=GRID, label="anomaly / unlabelled")
        # train / source first; orange stays reserved for anomalies (row 1)
        values = sorted(set(group[normal]), key=lambda v: (v not in ("train", "source"), v))
        for value, color in zip(values, (NORMAL, SERIES[2], SERIES[3])):
            mask = normal & (group == value)
            ax.scatter(xy[mask, 0], xy[mask, 1], s=16, color=color, label=f"normal · {value}", alpha=0.8,
                       edgecolors=SURFACE, linewidths=0.4)
        ax.set_title(f"{name}: normal clips by {group_name}", loc="left")
        ax.legend(loc="best")
        for a in axes[:, col]:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle(f"{title}\neach dot is one clip (mean + std of every log-mel band); "
                 "close dots = similar clips", x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, path)


def channel_correlation(corr: np.ndarray, path, title: str) -> None:
    """Correlation between channels over normal clips (e.g. accelerometer axes): high = redundant channels."""
    n = corr.shape[0]
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * n, 0.8 + 0.5 * n))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if abs(corr[i, j]) > 0.6 else INK)
    ax.set_xticks(range(n), [f"ch{i}" for i in range(n)])
    ax.set_yticks(range(n), [f"ch{i}" for i in range(n)])
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, loc="left")
    _save(fig, path)
