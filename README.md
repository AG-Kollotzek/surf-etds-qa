# surf-etds-qa

QA-Auswertung für ein hausgebautes mechatronisches Phantom, das die Tracking-Performance von
Surface-Guided-Radiotherapy(SGRT)-Systemen (hier: Brainlab ExacTrac Dynamic, "ETD") an C-Arm-Linacs prüft.
Das Phantom fährt kontrollierte, bekannte Translationen/Rotationen ab; ein separates SURF-Terminal
protokolliert die tatsächlichen Achsstellungen. Dieses Repo vergleicht beide Zeitreihen (SURF "Ground Truth"
vs. ETD-Tracking), um Genauigkeit und Robustheit des SGRT-Systems zu quantifizieren.

Hintergrund und Methodik sind im zugehörigen Paper
beschrieben (Abschnitt "Mechatronic Phantom for Tracking Performance Evaluation").

## Funktionsprinzip

Das Phantom hat zwei aktive Achsen (`Pos_H` horizontal, `Pos_R` Rotation), eine dritte (`Pos_V`, vertikal/Pitch)
ist mechanisch an `Pos_H` gekoppelt. Über `kinematics.py` (Vorwärtskinematik) werden aus den drei
Motorpositionen die klinischen 6-DoF-Koordinaten (`True_Lateral/Longitudinal/Vertical/Pitch/Roll/Yaw`)
berechnet; `kinematics_werror.py` liefert dieselbe Transformation inkl. Fehlerfortpflanzung
(`uncertainties`-Package) für Unsicherheitsschläuche in den Plots.

Zwei unabhängige Logs pro Messung:
- **SURF-Terminal-CSV** (`data/raw/L*/surf_phantom/*.csv`): Motor-Rohpositionen + Temperaturen, 10 Hz.
- **ETD-Tracking-JSON** (`data/raw/L*/etds_scans/TrackingResult_*.json`): vom SGRT-System gemessene
  Shift-Werte (lateral/longitudinal/vertical/pitch/roll/yaw) + RMSE-Kennzahlen, eigene Zeitbasis.

Da beide Systeme unabhängig starten und leicht unterschiedlich schnell takten, enthält jede Messfahrt am
Anfang und Ende einen kurzen **Sync-Puls** (definierter kurzer Achsausschlag, der auf das Ausgangsniveau
zurückspringt). `DataConverter.align_and_crop_signals` findet diese Pulse in beiden Zeitreihen anhand des
Stillstands der Achsen (Plateau-Erkennung), berechnet daraus Skalierungsfaktor und Zeitversatz zwischen den
beiden Uhren und schneidet beide Datensätze auf die eigentliche Messfahrt zu.

## Ablauf / Pipeline

1. **`etds_qa_2026_config.json`** ordnet jeder Messung eine ID, den Linac, die Messgruppe
   (z. B. `mindev`, `maxdev`, `couch_90`), den Heizpad-Status (`OFF`/`32`°C) und die Zeitstempel der
   zugehörigen SURF- und ETD-Rohdateien zu.
2. **`etds_qa_evaluation.py`** ist der Haupteinstiegspunkt:
   ```
   python etds_qa_evaluation.py <plot|rmse> <linac_id> <RT|32|ALL>
   ```
   Gruppiert die Config-Einträge nach `(group, heatingpads)`, lädt pro Messung CSV+JSON über
   `DataConverter.ETDQAProcessor`, richtet die Zeitachsen aus (`align_and_crop_signals`), wendet die
   Kinematik an (`apply_kinematics`) und exportiert Zwischenstände nach
   `data/process/L<n>/<RT|32>/<group>/meas_<id>/`:
   - `01_after_align_*.csv` – Rohsignale nach Zeit-Alignment, vor Kinematik
   - `02_aligned_kin_applied_*.csv` – nach Kinematik-Transformation (inkl. `True_*`-Spalten)
   - `03_alldof_rmse.csv` – RMSE ETD vs. Phantom pro DoF
   - `metadata.json` – Kopie der Config-Metadaten dieser Messung
3. Im `plot`-Modus werden alle Läufe einer Gruppe gemittelt (`bin_and_average`) und interaktiv als
   3-Panel-Matplotlib-Plot (Translation, Rotation, RMSE) mit editierbaren Zoom-Boxen dargestellt
   (`plot_evaluation_results_interactive`); Zoom-Box-Layouts werden in `zoom_box_config.json`
   gespeichert. Auf Kommando (`save`) wird der Plot als PDF neben den CSVs abgelegt.
4. **`DataConverter.ETDQAProcessor.evaluate_plateaus_and_export`** erkennt zusätzlich beliebige
   Endpositions-Plateaus (Stillstand der Motoren fernab der Nullposition, unter Herausfiltern der
   Sync-Pulse) und exportiert pro Plateau die Mean/Std-Abweichung ETD vs. Phantom als CSV-Report.

## Repo-Struktur

```
DataConverter.py           Kernklasse ETDQAProcessor: CSV/JSON laden, Zeit-Alignment, Kinematik, Export, Plots
etds_qa_evaluation.py      CLI-Batch-Pipeline über alle Messungen einer Config (Export + interaktives Plotten)
etds_qa_2026_config.json   Messungs-Metadaten (Linac, Gruppe, Heizpads, Zeitstempel, Couch-Winkel)
zoom_box_config.json       Persistierte Zoom-Box-Layouts der interaktiven Plots (pro Gruppe)
kinematics.py              Vorwärtskinematik Motorachsen -> klinische 6-DoF-Koordinaten (ohne Fehler)
kinematics_werror.py       Dieselbe Kinematik inkl. Unsicherheitsfortpflanzung (uncertainties)
quick_plotter.py           Standalone Plotly-Viewer: ETD vs. Phantom vor/nach Alignment (6 Subplots)
old_scripts/                Vorgängerversionen (batch_evaluation, calculate_rmse, raw_data_export)
data/raw/L<n>/
  surf_phantom/*.csv        SURF-Terminal-Rohlogs (Motorpositionen, Temperaturen), 10 Hz
  etds_scans/*.json + .png  ETD-Tracking-Rohlogs (Shift-Werte, RMSE) + Screenshot je Scan
data/process/L<n>/<RT|32>/<group>/meas_<id>/
                             Exportierte Zwischenstände + finaler Report pro Messung/Gruppe
data/measurement.json       (Altbestand / Referenz einer Einzelmessung)
```

## Setup

```bash
pip install -r requirements.txt
```

Benötigt außerdem `kaleido` für den PNG-Export der Plotly-Plots (`export_all_plots`) sowie ein
funktionierendes LaTeX-freies Matplotlib-Backend für die interaktiven PDF-Plots.
