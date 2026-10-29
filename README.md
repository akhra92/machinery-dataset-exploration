# Dataset exploration

Separate tool, not part of the Machinery-Defect-Prediction repository (`exploration/` is in its `.gitignore`). It
uses that repository's loaders and features, so clone it into the root of your clone, next to `configs/` and `src/`:
```bash
cd Machinery-Defect-Prediction
git clone https://github.com/akhra92/machinery-dataset-exploration exploration
```
Run the commands below from the clone root after `uv sync` (it installs matplotlib). Its test:
`python -m pytest exploration`.

Run this on every dataset **before** training and evaluation, to know what is in each one, spot data problems
early, and get a first idea of how hard each dataset is. It reads the data through the same loaders, train/test
split and log-mel settings as training (the dataset's config in `configs/`), so what you see is what the models get.

## Run

```bash
python -m exploration --all                                        # every dataset whose data is downloaded
python -m exploration --config configs/mimii.yaml configs/paderborn.yaml
python -m exploration --config configs/mimii.yaml --set data_root=data/mimii/-6dB/pump
python -m exploration --config configs/default.yaml --machine fan valve --clips-per-group 50
```

| Option | Default | |
|---|---|---|
| `--clips-per-group` | 30 | clips analysed per (split, class, domain) group; raise for more reliable statistics |
| `--examples` | 3 | example clips per class in the waveform / spectrogram figures |
| `--no-hash` | off | skip the duplicate / leakage check (it reads every file once) |
| `--out` | `outputs/exploration` | where reports go (not committed to Git) |

`--all` uses every config in `configs/` except the `folder.yaml` template and skips datasets that are not
downloaded. Run time is a few minutes per dataset; hashing dominates for large ones (MaFaulDa, MIMII).

## Output

```
outputs/exploration/
  overview.md / overview.csv        one row per dataset x machine type x sensor: compare datasets here
  <dataset>/report.md               full report with figures (open in VS Code / GitHub preview)
  <dataset>/figures/<machine>/      PNG figures
  <dataset>/clip_stats.csv          every statistic of every analysed clip (for your own analysis)
  <dataset>/summary.json
```

## What each report contains (per machine type and sensor)

| Section | What it shows | What to look for |
|---|---|---|
| Issues | problems found by the checks below | fix or exclude before training |
| Inventory | clips and files per split / class / domain, bar chart | enough normal training clips? ≥ 20 normal test clips? all fault types present? |
| File checks | MD5 of every file: duplicates, files in both train and test | any **leakage** inflates test scores |
| Signal properties | sample rate, channels, duration, total hours, model input size | mixed sample rates, varying durations |
| Examples | waveforms and log-mel spectrograms (the model input) of normal clips and each fault | can you see / hear the fault? |
| Average spectra | power spectrum (Hz) and average log-mel level per class | the frequency range where faults differ from normal; check `fmax` covers it |
| Statistics per class | RMS, crest factor, kurtosis, skewness, zero crossings, spectral centroid / bandwidth / roll-off / flatness, dominant frequency (table + box plots) | which properties change with each fault |
| Separability | AUC of each single statistic as an anomaly score, overall and per fault | near 1.0 = easy dataset (the models must beat this); near 0.5 = hard |
| Distribution shift | same AUC between normal train and normal test clips, and between source and target domain | train vs test well above 0.5 = test normals look different (false alarms) |
| 2-D map | PCA and t-SNE of clip features, coloured by normal / anomaly and by split or domain | anomalies in their own region = easy; mixed in = hard |
| Channel correlation | correlation between channels (multi-axis accelerometers, microphone arrays) | near 1 = redundant channels |

Quality checks on the analysed clips: unreadable files, NaN / Inf values, silent clips, clipping (flat tops),
large DC offset, mixed sample rates. Statistics come from a random sample (seeded); counts, hashes and fault lists
cover every clip.

## Choosing datasets (end of November)

Use `overview.md` and the reports to fill the Notion *Datasets* table: size, fault types, sensors, domains, issues,
and the best single-statistic AUC as a difficulty estimate. Prefer datasets that are clean, large enough, cover the
faults you expect in the factory, and are not trivially easy. Sound + vibration recorded together (MaFaulDa,
IMAD-DS) is closest to the final system. Mind licenses (DCASE is non-commercial).
