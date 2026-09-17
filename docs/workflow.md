# ETDS Annual QA Workflow — From Raw Data to the Report

State: release 1.0.0 (September 2026), first written for the scripts of 2026-08-04 (commit `df0815c`).
This document describes only the **currently
implemented workflow** — not the target state. Known pitfalls are listed in Section 7.

Related repositories: the phantom hardware and the SURF measurement terminal live in
[`surf-etds-phantom`](https://github.com/AG-Kollotzek/surf-etds-phantom); the paper-analysis
pipeline lives in [`surf-etds-analysis`](https://github.com/AG-Kollotzek/surf-etds-analysis);
raw measurement data live in the [`surf-etds-data`](https://github.com/AG-Kollotzek/surf-etds-data)
submodule used by this repository (see Section 1).

---

## 0. Prerequisites

```bash
pip install -r requirements.txt
```

Additionally needed:

| Component | For | Check with |
|---|---|---|
| `kaleido` | PNG export of the Plotly plots | `python -c "import kaleido"` |
| `latexmk` **or** `xelatex` | report compilation (`fontspec` requires XeLaTeX) | `which latexmk xelatex` |
| `pdftotext` / `pdfinfo` (poppler) | only for `create_report.py --public`: checks the PDF text and metadata for names/e-mail addresses | `which pdftotext pdfinfo` |

Raw measurement data live in the `surf-etds-data` git submodule. Clone this repository with
`git clone --recurse-submodules ...`, or run `git submodule update --init` afterwards if you
already have a plain clone (the same hint is printed if the submodule is missing,
`data_paths.py:22`).

All commands are run **from the repository root**: the config file
(`etds_qa_2026_config.json`) is opened as a path relative to the current working directory
(`etds_qa_evaluation.py:23`, `:784`), and so is the processing output directory
`data/process/L<n>/` (`etds_qa_evaluation.py:795`). The raw-data location, by contrast, is not
affected by the working directory — it is resolved from the script's own file location
(`data_paths.py:11`).

---

## 1. Step 1 — Add a new campaign in surf-etds-data and update the submodule

### 1.1 Target folder structure

Raw data live in the `surf-etds-data` submodule, one folder per measurement day and linac:

```
surf-etds-data/campaigns/<YYYY-MM-DD>_L<n>/
├── phantom/     SURF terminal logs (motor positions, temperatures, 10 Hz)
│   ├── <name>_<HHMMSS>.csv            ← measurement run
│   ├── <name>_<HHMMSS>_QA.csv         ← sphere-detection export (optional)
│   └── <name>_<HHMMSS>_log.json/.txt  ← terminal session log (not read by this pipeline)
└── etd/         ExacTrac Dynamic tracking exports
    ├── TrackingResult_YYYY-MM-DD_HH-MM-SS.json    ← measurement data
    └── TrackingResult_YYYY-MM-DD_HH-MM-SS.png     ← screenshot (optional)
```

`<n>` is the linac number. A linac can have several campaign folders, one per measurement day;
`data_paths.py` searches **all** folders whose name ends in `_L<n>` and requires **exactly
one** match per time stamp (`campaign_dirs`/`_find_one`, `data_paths.py:15–27`).

The full procedure for adding a campaign (naming the folder, copying files unchanged, writing
`runs.csv`, running `tools/prepare_campaign.py` and `tools/validate.py`, opening a pull
request) is documented in `surf-etds-data`'s own `CONTRIBUTING.md`. Once the campaign is merged
there, update the submodule pointer in this repository (`cd surf-etds-data && git pull`,
back in the repository root `git add surf-etds-data && git commit`) so the new data becomes
visible to this pipeline.

### 1.2 File naming conventions (binding)

The scripts do not look up raw files by their full name but by a suffix glob on the time
stamp, implemented in `data_paths.py`:

| File | Search pattern | Consequence |
|---|---|---|
| SURF CSV | `phantom/*<surf_timestamp>.csv`, searched across all campaign folders of the linac (`data_paths.surf_csv`, `data_paths.py:36–38`) | the name can be anything, but it **must end in `HHMMSS.csv`** |
| ETD JSON | `etd/TrackingResult_*_<HH-MM-SS>.json` (`data_paths.etd_json`, `data_paths.py:46–48`) | the name must end in `TrackingResult_..._HH-MM-SS.json` |
| Sphere-detection | `<SURF-CSV name without .csv>_QA.csv` | derived from the CSV path (`sphere_detection.qa_csv_path_for`, called at `etds_qa_evaluation.py:628`) |

Important:
- `*_QA.csv` is actively **excluded** when collecting the measurement file
  (`data_paths.py:38`) — the sphere-detection file may safely sit next to it, but must be
  named exactly like that.
- The ETD time stamp is converted automatically from `161959` to `16-19-59`
  (`data_paths.etd_stamp_dashed`, `data_paths.py:30–33`). In the config it is written
  **without** dashes.
- Exactly one match is required per time stamp, across **all** campaign folders of the linac.
  Zero or more than one hit raises `FileNotFoundError`, listing every file found
  (`data_paths.py:20–27`); see 7.3/7.4.
- If the raw data for a measurement cannot be resolved this way, the measurement is skipped
  with `[!] FEHLT:` (`etds_qa_evaluation.py:454–461`) — the run does **not** abort.

### 1.3 Placement of the sphere-detection file

The `*_QA.csv` is optional. If it is missing, the QA plot shows the note "Keine
Sphere-Detection-Daten verfuegbar" instead of the comparison table on page 1
(`qa_plots.py:389`; the spider-plot page has the same placeholder at `qa_plots.py:158`); the
run continues normally.

---

## 2. Step 2 — Extend the config

File: `etds_qa_2026_config.json` (fixed path, `etds_qa_evaluation.py:23`).

Structure: a flat object, **key = measurement ID as a string**, one entry per ETD scan.

```json
"9": {
  "Linac": "1",
  "deflection": 1,
  "meas_couch_type": "single angle",
  "heatingpads": "OFF",
  "etds_timestamp": "161959",
  "surf_timestamp": "161730",
  "couch_angle": 0
}
```

### 2.1 Field reference

| Field | Type / allowed values | Effect |
|---|---|---|
| `Linac` | **String**, e.g. `"1"` | filter; compared with `!=` against the CLI argument (`etds_qa_evaluation.py:413`) — `1` as a number does **not** match |
| `deflection` | `1` (small displacement) or `2` (large) | folder name, plot grouping, CLI filter |
| `meas_couch_type` | `"single angle"` or `"multi angle"` | folder level (`single_angle`/`multi_angle`); **`multi angle` is skipped by `numqa`** (`etds_qa_evaluation.py:668–672`) |
| `heatingpads` | `"OFF"` or `"32"` | folder/plot label `RT` or `32`. CLI filter `RT` is internally mapped to `OFF` (`etds_qa_evaluation.py:408–409`) |
| `etds_timestamp` | string `"HHMMSS"` (6 characters) | locates `TrackingResult_*.json` |
| `surf_timestamp` | string `"HHMMSS"` | locates the SURF CSV; **several measurements can share the same value** |
| `couch_angle` | number, e.g. `0` or `-90` | feeds into kinematics **and** the folder name |

### 2.2 The critical rule: ID order = driving order

Several ETD scans usually share **one** SURF CSV (e.g. deflection 1 + 2 in one run, up to 4
scans for the couch series). The CSV then contains a matching number of sync-pulse pairs.
Which pair belongs to which measurement is determined **solely by the numerically sorted
config ID** (`compute_pair_index`, `etds_qa_evaluation.py:424–441`):

> All entries with the same `surf_timestamp` are sorted by `int(id)`; the position in that
> list is the index of the sync-pulse pair in the CSV.

**Consequences for the workflow:**
1. Always **append** new measurements with a continuing ID.
2. IDs within a `surf_timestamp` group must reflect the **actual chronological driving
   order**.
3. Renumbering IDs afterwards, or inserting entries in between, causes wrong time offsets for
   all subsequent measurements of the same CSV, **without any error message**.

Deliberate design decision: matching by wall-clock (`etds_timestamp` vs. `surf_timestamp`)
would be more fragile, because a misdetected sync pulse shifts all following offsets
(comment `etds_qa_evaluation.py:428–435`).

---

## 3. Step 3 — Processing (alignment + kinematics)

```bash
python etds_qa_evaluation.py <function> <linac_id> [deflection: 1/2/all] [heatingpads: RT/32/all]
```

Both filters are optional; missing or `all` means no filter.

| Function | Effect |
|---|---|
| `process` | alignment + kinematics, writes the 01/02/03 CSVs + `metadata.json` |
| `plotpaper` | like `process` + interactive publication plot per measurement series |
| `plotqa` | like `process` + A4 QA overview per temperature setting |
| `numqa` | numerical evaluation in the measurement window (**requires a previous `process` run**) |

Aliases: `plot` → `plotpaper`, `deviation` → `process`.

For the annual report, **`plotqa`** is sufficient — it includes `process` completely.

### 3.1 Output structure

```
data/process/L<n>/<single_angle|multi_angle>/L<n>_<RT|32>_Deflection<N>_Couch<angle>_<DATE>/
├── 01_after_align_etd.csv           raw signals after time alignment, before kinematics
├── 01_after_align_phantom.csv
├── 02_aligned_kin_applied_etd.csv   after kinematics (True_* columns)  ← basis for numqa
├── 02_aligned_kin_applied_phantom.csv
├── 03_alldof_rmse.csv               RMSE over the entire export (overview, NOT the QA metric)
├── 03_alldof_ae_time.csv            plotqa/numqa only
└── metadata.json                    config + sync time stamps + measurement window
```

`plotqa` additionally writes a **two-page** PDF per temperature setting:
`data/process/L<n>/single_angle/L<n>_<RT|32>_Couch<angle>_<DATE>_QA.pdf`

| Page | Content |
|---|---|
| 1 | time-series overview: 6 DoF fields + RMSE field + sphere-detection comparison table |
| 2 | spider plot of the sphere-detection deviations (3 translation axes), tolerance rings, value table |

On page 2, two magnitudes are plotted per deflection — `|SD − Phantom|` (X-ray against the
axis target) and `|SD − Surface Tracking|` (X-ray against ETD tracking), evaluated at the
displaced position at the end of the measurement window. Deflection 1 in red tones, deflection
2 in blue tones; the darker shade is always the comparison against the phantom target.

> **`<DATE>` is the processing date (today), not the measurement date**
> (`date_str = datetime.now()`, `etds_qa_evaluation.py:796`). Every run on a new day creates a
> **new folder** next to the old one. See the pitfalls in Section 7.

### 3.2 Measurement window

Not the entire export counts, only:

```
Measurement window = [ center(first sync pulse) + 3 s , center(last sync pulse) − 3 s ]
```

The margin is set in `qa_metrics.DEFAULT_WINDOW_MARGIN_SEC = 3.0` and is written into
`metadata.json` for every measurement (`measurement_window_margin_sec`,
`measurement_window_sec`), so that `numqa` can use it without redoing the alignment.

---

## 4. Step 4 — Numerical evaluation

```bash
python etds_qa_evaluation.py numqa <linac_id>
```

Reads the `02_*` CSVs of the **newest** processing folder per measurement series
(`find_latest_measurement_dir` — alphabetically last folder with a matching prefix,
`etds_qa_evaluation.py:653–659`).

Produces per measurement: `03_alldof_parameters.csv` (MAE / RMSE / max. absolute error per
DoF, measurement window only) and overwrites `03_alldof_ae_time.csv`.

Aggregated at the linac level, into `data/process/L<n>/`:

| File | Content |
|---|---|
| `ETDS_L<n>_numqa_<DATE>.csv` | DoF, Unit, Deflection, Pad_Temperature, Meas_Couch_Type, Couch_Angle, MAE, RMSE, MaxAE, N_Samples |
| `ETDS_L<n>_passrate_<DATE>.csv` | pass rates per individual measurement |
| `ETDS_L<n>_passrate_summary_<DATE>.csv` | pass rate per DoF, pooled over all deflection/pad combinations |

The summary pools **raw points**, not the percentage values of the individual measurements
(`qa_metrics.pool_ae_time_series`).

### 4.1 Tolerances (single source: `qa_metrics.py:30–31`)

```python
TOLERANCE_ACCEPT = 1.0   # |AE| <= 1.0  -> pass
TOLERANCE_WATCH  = 2.0   # 1.0 < |AE| <= 2.0 -> watch ;  > 2.0 -> act
```

The unit is that of the respective DoF: **mm** for translation, **degrees** for rotation.
These values are injected into the LaTeX template as `<<TOL_ACCEPT>>` / `<<TOL_WATCH>>`
(`create_report.py:661–662`) — the tolerance table in the report is therefore automatically
consistent with the evaluation.

### 4.2 Sign convention

`qa_metrics.DOF_SPEC` is the **single** place where the phantom column, ETD column, sign and
unit are defined per DoF. Relevant: `vertical` has `etd_sign = -1.0`, because the ETD vertical
axis points opposite to the phantom kinematics. Without this correction, an apparent error on
the order of the displacement itself appears instead of ~0.03 mm.

---

## 5. Step 5 — Generate the report

```bash
# internal build (signable PDF with names, from the local people list)
python create_report.py <linac_id> [--date YYYY-MM-DD] [--author QMP2,QMP3] \
                        [--approver QMP2] [--local-config PATH] [--out PATH] [--keep-build]

# public build (role codes only, issue tracker instead of names/e-mail, no logo)
python create_report.py <linac_id> --public [--date YYYY-MM-DD] [--out PATH]
```

Result: `report/output/ETDS_L<n>_QA_Report_<DATE>.pdf`, or `..._public.pdf` with `--public`
(`create_report.py:692–693`).

Internal flow: the template `report/template/` is copied to `report/build/L<n>` (or
`report/build/L<n>_public` with `--public`, `create_report.py:673`), every `<<PLACEHOLDER>>`
token is substituted in a single pass (`render_templates`, `create_report.py:524–541`; a
missing value for a placeholder is fatal), then compiled with `latexmk -xelatex` (preferred) or
two `xelatex` passes (`compile_pdf`, `create_report.py:544–563`). The build folder is cleaned
up unless `--keep-build` is given (`create_report.py:697–700`).

### 5.1 People and role codes

Operators and reviewers appear in the templates only as role codes (`QMP<n>`, `Student<n>`,
`RTT<n>`, matched against `ROLE_CODE`/`QMP_CODE`, `create_report.py:52–53`). Settings are
resolved in ascending precedence (`create_report.py:63–76,357–384`):

1. `DEFAULTS` in the script (`create_report.py:65–75`) — institution `"tirol kliniken"`,
   phantom `"SURF"`.
2. the committed `report/report_config.json` — role codes only: `authors`, `approvers`,
   `contact`, plus `institution`, `phantom`, `public_contact_url` (the issue-tracker URL).
3. the git-ignored `report/report_config.local.json` (only for the internal build, **not**
   applied with `--public`) — `people_file` (path to a local `people.json`-format list with
   names and e-mail addresses), `lab_url`, `logo`; template
   `report/report_config.local.example.json`.
4. `--author` / `--approver` on the command line (role codes only).

Only role codes matching `QMP<n>` may approve and sign a report
(`create_report.py:380–383`), and in the internal build their entry in the local people list
must have the role `QMP` (`create_report.py:447–450`); anything else makes the run fail (exit
status 2). The "Tested
by" line accepts `QMP<n>`, `Student<n>` or `RTT<n>`.

Names are resolved only for the internal build, via `people_file` from the local config
(`load_people`, `create_report.py:387–396`); a code that is not in the list stops the internal
build. With `--public` the report shows the role codes themselves (`person_label`,
`create_report.py:428–439`).

### 5.2 Public report (`--public`)

`--public` produces a report that contains **only** role codes, the issue-tracker URL from
`public_contact_url` as the contact, and no logo (`build_person_fields`,
`create_report.py:442–494`). After compiling, `check_public_report` rejects the PDF if
(`create_report.py:497–518`):

- the `.tex` sources contain `@`, `mailto:`, or an e-mail-address pattern (LaTeX column
  specifiers such as `@{}` are excluded from this check), or
- the PDF text or PDF metadata (via `pdftotext`/`pdfinfo`) contain `@`, or
- the `.tex` sources, PDF text, or PDF metadata contain a name, alias, or e-mail address from
  the local people list. Without `report/report_config.local.json` and its `people_file`
  the run stops before building (`local_name_tokens`, `create_report.py:399–425`);
  `--no-people-list` skips the name check and prints a warning.

If any of these checks trigger, report generation fails and no `_public.pdf` is written.

### 5.3 Data sources

| Source | fills |
|---|---|
| `ETDS_L<n>_passrate_summary_<date>.csv` | pass rates per DoF + traffic-light box (page 1) |
| `ETDS_L<n>_numqa_<date>.csv` | max. MAE/RMSE/absolute error (page 1), error table (page 3) |
| `ETDS_L<n>_passrate_<date>.csv` | pass rates per measurement (page 2) |
| `single_angle/L<n>_<RT\|32>_Couch*_<date>_QA.pdf` | plot pages in the appendix |
| `report/report_config.json` + `report/report_config.local.json` | role codes/names, `institution`, `phantom`, contact, logo |
| `report/history/L<n>_history.csv` | previous years' values (`;`-separated, `#` lines are comments) |
| `surf-etds-data/campaigns/*_L<n>/etd/` | measurement date/time in the header, earliest single-angle scan (`measurement_datetime`, `create_report.py:182–212`) |

Without `--date`, **each source is chosen independently** as the newest available file
(`find_latest_csv`, `find_latest_plot`, `create_report.py:150–179`). The plot lookup prefers
Couch 0.

### 5.4 Verdict logic (`classify_rates`)

| Result | Condition |
|---|---|
| **Action** | at least one action point present **or** pass < 95 % |
| **Watch** | pass 95–99 % with 1–5 % watch |
| **Pass** | otherwise |

(`create_report.py:122–133`.) The traffic-light box on page 1 shows the **worst** result of
all six DoF, the largest absolute error, and the pooled overall pass rate.

### 5.5 LaTeX iteration without recomputation

```bash
./rebuild_pdf.sh <linac_id>
```

Recompiles only `report/build/L<n>` (requires `--keep-build` on the previous run). It targets
only the internal build directory; for a `--public` build (`report/build/L<n>_public`), run
`latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex` in that directory manually.

---

## 6. Minimal end-to-end run (short form)

```bash
# 1. Add the new campaign to surf-etds-data/campaigns/<date>_L1/{phantom,etd}/ and update the
#    submodule pointer (see 1.1, naming rules in 1.2)
# 2. Extend etds_qa_2026_config.json by one entry per ETD scan (IDs in driving order!)

python etds_qa_evaluation.py plotqa 1     # alignment, kinematics, 02 CSVs, QA plot pages
python etds_qa_evaluation.py numqa 1      # metrics in the measurement window + pass-rate tables
python create_report.py 1                 # internal PDF report
# python create_report.py 1 --public      # public PDF report (role codes only)
```

Run all three evaluation steps **on the same day** (see 7.1).

---

## 7. Known pitfalls

### 7.1 Processing date ≠ measurement date → stale-data risk

Folder and file names carry the **date the script was run**, not the measurement date.
`create_report.py` looks up the metrics CSV and the plot PDF **independently of each other**,
each choosing the newest available version (`find_latest_csv`, `find_latest_plot`,
`create_report.py:150–179`). If `plotqa` is run on one day and `numqa` is not run along with
it, the report ends up combining new plots with old numbers.

**Recommendation:** always run `plotqa` → `numqa` → `create_report.py` as one block, and
generate the report with an explicit `--date`.

### 7.2 "multi angle" does not feed into the report/metrics

`numqa` filters on `meas_couch_type == "single angle"` (`etds_qa_evaluation.py:668–669`);
skipped measurements are only reported as `[i] numqa: N multi-angle-Messung(en)
übersprungen`. The `Meas_Couch_Type`/`Couch_Angle` columns stay in the schema so that the
couch series can be added later without a format change. Per the measurement protocol it is
not required for the annual QA.

### 7.3 Silent error paths

These cases do **not** abort the run and must be checked manually in the console output:

| Message | Cause |
|---|---|
| `[!] FEHLT: Rohdaten für ID <x>` | the time stamp matches no file (or more than one, see 7.4) |
| `[X] ID <x>: kein Verarbeitungsordner gefunden` | `numqa` without a previous `process` run |
| `[X] ID <x>: metadata.json enthaelt keine Sync-Zeitstempel` | folder from an old script version → repeat `process` |
| `Keine passenden Messungen für Linac <n>` | `Linac` in the config not written as a string, or the filter too narrow |

**Consequence:** check the console output of every run; compare the row count of
`ETDS_L<n>_numqa_<date>.csv` against the expected number of measurements (6 DoF × number of
single-angle measurements).

### 7.4 Ambiguous time stamps

`data_paths.py` requires **exactly one** hit per time stamp, searched across all campaign
folders of a linac; zero or more than one match raises `FileNotFoundError`, listing every file
found (`data_paths.py:20–27`). A duplicate `HHMMSS` value (e.g. data from different days mixed
under the same linac) therefore no longer picks the wrong file silently — the measurement is
skipped with `[!] FEHLT:` instead, same as a missing file (`etds_qa_evaluation.py:454–461`, see
7.3). The duplicate must still be resolved manually (rename one of the raw files) before that
measurement can be processed.

### 7.5 Year-over-year comparison currently empty

`report/history/L1_history.csv` contains only comment lines (and the column header) ⇒ the
"Maximum error metrics compared with previous years" table stays empty. It must be maintained
**manually** after every annual run; there is currently no automation that fills it in.

### 7.6 Un-versioned legacy data

`data/process/L1/32/` and `data/process/L1/RT/` are left over from the old folder logic
(before `single_angle`/`multi_angle`) and are no longer read by any current script. Do not use
them as a data source.

---

## 8. Script reference

| File | Role in the workflow |
|---|---|
| `etds_qa_evaluation.py` | **Main entry point**: CLI, orchestration (`process`/`plotpaper`/`plotqa`/`numqa`), paper plot |
| `data_paths.py` | Locates raw files in the `surf-etds-data` submodule; exactly one match per time stamp and linac |
| `DataConverter.py` | Core class `ETDQAProcessor`: load CSV/JSON, sync-pulse detection, time alignment, kinematics, export |
| `qa_metrics.py` | `DOF_SPEC` (mapping/sign), measurement window, MAE/RMSE/MaxAE, tolerances, table aggregation |
| `qa_plots.py` | A4 QA overview page (8 fields) for `plotqa` |
| `sphere_detection.py` | Evaluates the `*_QA.csv` + comparison table (field 8 of the QA page) |
| `create_report.py` | Builds the annual report (LaTeX → PDF) from the `numqa`/`plotqa` results |
| `rebuild_pdf.sh` | LaTeX-only recompilation of the existing build folder |
| `kinematics.py` | Forward kinematics, motor axes → clinical 6 DoF (without error propagation) |
| `kinematics_werror_v2.py` | Current kinematics incl. error propagation (exact rotation composition) |
| `radius_calibration.py` | Calibrates the lever arm `radius` of the kinematics against the sphere detection; its mean is stored as `SD_CALIB_DEFAULT` in `kinematics_werror_v2.py` (rerun after hardware changes) |
| `geometry_diagnostic.py` | Diagnostic helper script (not part of the workflow) |

### Configuration files

| File | Purpose | Must be maintained? |
|---|---|---|
| `etds_qa_2026_config.json` | Measurement metadata per ETD scan | **Yes, for every new measurement** |
| `report/report_config.json` | Report header data: role codes (authors/approvers/contact), institution, phantom, public contact URL | When roles change |
| `report/report_config.local.json` | Git-ignored: `people_file`, `lab_url`, `logo` for the internal, signable build | Locally, per installation |
| `report/history/L<n>_history.csv` | Previous years' values for the year-over-year comparison | Annually, manually |
| `zoom_box_config.json` | Zoom-box layouts of the interactive paper plots, per measurement series | No (writes itself) |
