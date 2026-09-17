# surf-etds-qa

QA evaluation pipeline for a custom-built mechatronic phantom that tests the tracking
performance of surface-guided radiotherapy (SGRT) systems — here: Brainlab ExacTrac Dynamic
("ETD") — at C-arm linear accelerators of Tirol Kliniken (LKI) / Medical University of
Innsbruck. The phantom drives controlled, known translations and rotations; an independent
SURF terminal logs the actual axis positions. This repository compares the two time series
(SURF "ground truth" vs. ETD tracking) to quantify accuracy and robustness of the SGRT system
and to produce the annual QA report.

## Principle of operation

The phantom has two active axes (`Pos_H` horizontal, `Pos_R` rotation); a third (`Pos_V`,
vertical/pitch) is mechanically coupled to `Pos_H`. `kinematics.py` provides the forward
kinematics that turns the three motor positions into clinical 6-DoF coordinates
(`True_Lateral/Longitudinal/Vertical/Pitch/Roll/Yaw`); it is also used internally as a fast,
error-free reference curve during time-alignment refinement. `kinematics_werror_v2.py`
computes the same transformation via exact rotation composition including uncertainty
propagation (`uncertainties` package) and is what `DataConverter.apply_kinematics` actually
uses for the exported coordinates and their error tubes.

Every measurement produces two independent logs:

- **SURF terminal CSV** (`surf-etds-data/campaigns/<YYYY-MM-DD>_L<n>/phantom/*.csv`): raw
  motor positions and temperatures, 10 Hz.
- **ETD tracking JSON** (`.../etd/TrackingResult_*.json`): shift values
  (lateral/longitudinal/vertical/pitch/roll/yaw) and RMSE figures reported by the SGRT system,
  on its own time base.

Because the two systems start independently and run at slightly different rates, each run
contains a short **sync pulse** near the beginning and the end (a brief, defined axis
excursion that returns to baseline). `DataConverter.align_and_crop_signals` locates these
pulses in both time series from the axes' stand-still plateaus, derives a scale factor and
time offset between the two clocks from them, and crops both datasets to the actual
measurement run.

## Measurement window and alignment

Only the interval between the two sync pulses — not the full exported range — counts for the
numeric evaluation:

```
measurement window = [ mid(first sync pulse) + 3 s , mid(last sync pulse) − 3 s ]
```

The 3 s margin (`qa_metrics.DEFAULT_WINDOW_MARGIN_SEC`) keeps the sync motion itself, including
its settling, out of the error metrics. The sync timestamps are written into each
measurement's `metadata.json` (raw timestamps of both systems, common time base, scale factor)
so that `numqa` and the sphere-detection evaluation can reuse them without re-running the
alignment.

Sign convention: the ETD vertical axis points opposite to the phantom kinematics (and to the
sphere-detection Z axis). `qa_metrics.DOF_SPEC` is the single place where this mapping
(phantom column, ETD column, sign, unit) is defined; plots, the overview RMSE export and
`numqa` all read from it, so the convention cannot drift out of sync between them.

## Setup

```bash
git clone --recurse-submodules https://github.com/AG-Kollotzek/surf-etds-qa
# or, after a plain clone: git submodule update --init --recursive
pip install -r requirements.txt
```

Raw data live entirely in the `surf-etds-data` submodule; there is no local `data/raw/`
directory. `kaleido` is required for the Plotly PNG export used by the interactive paper
plots, and a working XeLaTeX installation (`fontspec`, `latexmk` or `xelatex`) is required to
compile the QA report.

## Workflow

Raw files are organised per campaign in the `surf-etds-data` submodule:

```
surf-etds-data/campaigns/<YYYY-MM-DD>_L<n>/
├── phantom/   SURF terminal CSV, *_QA.csv (sphere-detection reference), *_log.json/.txt
└── etd/       TrackingResult_*.json (tracking data) + TrackingResult_*.png (screenshot)
```

`data_paths.py` searches across all campaign folders ending in `_L<n>` for a given linac and
requires **exactly one** match per timestamp; ambiguous or missing files raise an error
instead of silently picking a file.

Each ETD scan is registered in `etds_qa_2026_config.json` with an ID, linac, `deflection`
(1 = small, 2 = large excursion), couch type/angle, heating-pad state and the timestamps of
its raw files. The main entry point is:

```bash
python etds_qa_evaluation.py <mode> <linac_id> [deflection: 1/2/all] [heatingpads: RT/32/all]
```

| Mode | Effect |
|---|---|
| `process` | alignment + kinematics, writes the 01/02/03 CSVs and `metadata.json` |
| `plotpaper` | like `process`, plus the interactive publication plot per measurement series |
| `plotqa` | like `process`, plus the two-page A4 QA overview per temperature setting |
| `numqa` | numeric evaluation inside the measurement window (requires a prior `process` run) |

(`plot`/`deviation` remain aliases for `plotpaper`/`process`.) A typical run for the annual QA:

```bash
python etds_qa_evaluation.py plotqa 1     # alignment, kinematics, QA overview pages
python etds_qa_evaluation.py numqa 1      # error metrics + pass-rate tables inside the window
python create_report.py 1                 # PDF report for Linac 1
```

Full details — config field reference, exported file layout, the sphere-detection comparison,
and known pitfalls (stale processing dates, multi-angle handling, silent failure paths) — are
documented in [`docs/workflow.md`](docs/workflow.md).

## Repository layout

```
DataConverter.py            Core class ETDQAProcessor: load CSV/JSON, sync-pulse detection,
                             time alignment, kinematics, export
data_paths.py                Locates raw files for a linac across all surf-etds-data campaigns
etds_qa_evaluation.py        CLI + orchestration (process/plotpaper/plotqa/numqa) + paper plot
qa_metrics.py                DOF_SPEC (mapping/sign), measurement window, MAE/RMSE/MaxAE,
                             tolerances, table aggregation
sphere_detection.py          Evaluates the *_QA.csv sphere-detection files + comparison table
qa_plots.py                  Two-page A4 QA overview (plotqa)
create_report.py             Builds the annual QA report (LaTeX -> PDF) from numqa/plotqa results
kinematics.py                Forward kinematics, motor axes -> clinical 6-DoF (no error propagation)
kinematics_werror_v2.py      Same kinematics incl. uncertainty propagation (exact rotation
                             composition); used by DataConverter for the exported coordinates
radius_calibration.py        Calibrates the kinematics lever arm against the sphere detection
                             (SD_CALIB_DEFAULT in kinematics_werror_v2.py)
geometry_diagnostic.py       Diagnostic helper, not part of the workflow
etds_qa_2026_config.json     Measurement metadata (linac, deflection, couch type/angle,
                             heating pads, raw-file timestamps)
zoom_box_config.json         Persisted zoom-box layouts of the interactive paper plots
rebuild_pdf.sh               Recompiles an existing report/build/L<n> without recomputing data
docs/workflow.md             Full step-by-step workflow documentation
report/
├── template/                LaTeX template (main.tex + Report/*.tex), placeholders <<...>>
├── report_config.json       Public report metadata: role codes only (authors, approvers,
│                            contact, issue-tracker URL) — see "Report generation"
└── history/                 Previous years' error metrics per linac, for the year-on-year table
surf-etds-data/               Git submodule with the raw campaign data (see "Workflow")
data/process/L<n>/
├── <single_angle|multi_angle>/L<n>_<RT|32>_Deflection<N>_Couch<angle>_<date>/
│                            Exported intermediate results per measurement (01/02/03 CSVs +
│                            metadata.json)
└── ETDS_L<n>_numqa_<date>.csv, ..._passrate_<date>.csv, ..._passrate_summary_<date>.csv
                             Aggregated QA tables per linac
```

`report/report_config.local.json` (git-ignored; copy `report/report_config.local.example.json`)
supplies real names and e-mail addresses (via a `people_file`), the institute's lab URL and the
logo used in internal report renders; it is never part of the public repository.

## Report generation

`report/report_config.json`, checked into this repository, contains **role codes only**:
`authors` and `approvers` (e.g. `QMP2`, `QMP3`), a `contact` (`Lead1`), and a
`public_contact_url` pointing at this repository's issue tracker
(`github.com/AG-Kollotzek/surf-etds-qa/issues`). Report approval and the signature field on
the report's overview page are reserved for `QMP<n>` (Qualified Medical Physicist) role codes.

```bash
python create_report.py <linac_id> [--date YYYY-MM-DD] [--out PATH] [--keep-build]
python create_report.py <linac_id> --public
```

Without `--public`, `create_report.py` builds the internal report: it fills the LaTeX template
in `report/template/`, pulls names, e-mail addresses and the logo from the git-ignored
`report_config.local.json`, and compiles with XeLaTeX (`latexmk -xelatex` preferred, else two
`xelatex` passes) into `report/output/ETDS_L<n>_QA_Report_<date>.pdf`.

`--public` produces a redacted variant with role codes and the issue-tracker URL only, no
logo, written as `..._public.pdf`. Before writing the file it checks the rendered PDF text and
metadata, as well as the intermediate `.tex` sources, against the e-mail addresses and names
from the local people list, and refuses to produce the file if any are still present. Without
the local people list it stops, unless `--no-people-list` is given (e-mail check only).

Both variants read the same data sources from `data/process/L<n>/`:
`ETDS_L<n>_passrate_summary_<date>.csv` and `ETDS_L<n>_numqa_<date>.csv` (Overview page),
`ETDS_L<n>_passrate_<date>.csv` (per-measurement passrates), the
`single_angle/L<n>_<RT|32>_Couch*_<date>_QA.pdf` plot pages from `plotqa`, and
`report/history/L<n>_history.csv` for the year-on-year comparison. Rows are classified as
**Action** (an action point present or pass rate < 95 %), **Watch** (95–99 % pass with
1–5 % watch) or **Pass** (otherwise); the pass/watch/action thresholds are the single source
of truth in `qa_metrics.TOLERANCE_ACCEPT` / `TOLERANCE_WATCH` and are injected into the
tolerance table automatically. `rebuild_pdf.sh <linac_id>` recompiles an existing
`report/build/L<n>` (from a run with `--keep-build`) without recomputing the underlying data.

## Related repositories

- [`surf-etds-phantom`](https://github.com/AG-Kollotzek/surf-etds-phantom) — phantom hardware,
  firmware and the SURF measurement terminal.
- [`surf-etds-analysis`](https://github.com/AG-Kollotzek/surf-etds-analysis) — the
  paper-oriented pipeline that characterises and validates tracking accuracy.
- [`surf-etds-data`](https://github.com/AG-Kollotzek/surf-etds-data) — the raw campaign data
  consumed by this repository (submodule).

## Citing

Citation metadata for this repository is provided in `CITATION.cff`. When citing results
produced with this pipeline, please also cite the `surf-etds-data` campaign(s) evaluated and,
where hardware or firmware details are relevant, `surf-etds-phantom`.

## Licence

Code (`*.py`, `*.sh`) is licensed under the MIT License. Documentation (this README,
`docs/`) and the result tables produced under `data/process/` and `report/` are licensed
under CC BY 4.0.

## Disclaimer

"ExacTrac" and "ExacTrac Dynamic" are trademarks of Brainlab AG; they are used here only
nominatively to identify the tracked SGRT system and imply no affiliation with or endorsement
by Brainlab. This repository is a research and quality-assurance tool, **not** a certified
medical device. Its output must not be used for clinical decisions without independent local
validation and sign-off by a Qualified Medical Physicist (QMP).
