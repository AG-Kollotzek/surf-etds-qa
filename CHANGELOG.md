# Changelog

## 1.0.0 — prepared for publication (September 2026)

- Raw data moved to the [`surf-etds-data`](https://github.com/AG-Kollotzek/surf-etds-data) repository, included as a
  submodule. `data_paths.py` finds each measurement across the campaigns of a linac and requires exactly one file
  per time stamp; `data/raw/` is gone. The evaluation results are unchanged.
- Report generation: `report/report_config.json` holds role codes only. Names, e-mail addresses, the lab URL and
  the logo come from a git-ignored local configuration for the internal, signable report. Only QMPs may approve and
  sign. `--public` produces a report with codes only and checks it for e-mail addresses and names. Configuration
  values are escaped for LaTeX; placeholders are replaced in one pass and a missing value is an error; the report
  falls back to TeX Gyre Heros where Arial is not installed.
- The finished sections of the SOP draft are published in English as [`docs/workflow.md`](docs/workflow.md).
- Removed: generated reports and build folders, the example report, the logo, the one-pager, `old_scripts/`,
  `kinematics_werror.py`, `quick_plotter.py`, `data/measurement.json` and a zip archive of old plots.
- `radius_calibration.py` runs with NumPy 2 (`np.ptp`).
- Licence files, citation metadata, contributors, unit tests and continuous integration added.

## 2026-08-10

- Yearly QA of linacs 0, 1, 3 and 4 evaluated; reports generated.
