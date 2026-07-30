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

## Messfenster

Für die numerische Auswertung zählt nicht der gesamte exportierte Bereich, sondern nur die
eigentliche Messung. Sie ist über die beiden Sync-Pulse definiert:

```
Messfenster = [ Mitte(erster Sync-Puls) + 3 s , Mitte(letzter Sync-Puls) − 3 s ]
```

Der Abstand von 3 s hält die Sync-Bewegung selbst (samt Ein-/Ausschwingen) aus den Metriken heraus.
`align_and_crop_signals` legt die dafür nötigen Zeitstempel in `self.sync_info` ab; sie werden in die
`metadata.json` jeder Messung geschrieben (Roh-Zeiten beider Systeme + gemeinsame Zeitbasis +
Skalierungsfaktor), damit `numqa` und die Sphere-Detection-Auswertung ohne erneutes Alignment
darauf zugreifen können.

## Ablauf / Pipeline

1. **`etds_qa_2026_config.json`** ordnet jeder Messung eine ID, den Linac, `deflection` (1 = kleine,
   2 = große Auslenkung), `meas_couch_type` (`single angle`/`multi angle`), den Heizpad-Status
   (`OFF`/`32`°C), den Couch-Winkel und die Zeitstempel der zugehörigen Rohdateien zu.
2. **`etds_qa_evaluation.py`** ist der Haupteinstiegspunkt:
   ```
   python etds_qa_evaluation.py <funktion> <linac_id> [deflection: 1/2/all] [heatingpads: RT/32/all]
   ```
   Die beiden Filter sind optional; fehlend oder `all` bedeutet „alles". Funktionen:

   | Funktion | Wirkung |
   |---|---|
   | `process` | Alignment + Kinematik, schreibt die 01/02/03-CSVs und `metadata.json` |
   | `plotpaper` | wie `process`, zusätzlich der interaktive Publikationsplot je Messreihe |
   | `plotqa` | wie `process`, zusätzlich die A4-QA-Übersicht je Temperatursetting |
   | `numqa` | numerische Auswertung im Messfenster (setzt einen vorherigen `process`-Lauf voraus) |

   `plot` und `deviation` bleiben als Aliase für `plotpaper` bzw. `process` erhalten.
3. Export je Messung nach
   `data/process/L<n>/<single_angle|multi_angle>/L<n>_<RT|32>_Deflection<N>_Couch<Winkel>_<Datum>/`:
   - `01_after_align_*.csv` – Rohsignale nach Zeit-Alignment, vor Kinematik
   - `02_aligned_kin_applied_*.csv` – nach Kinematik-Transformation (inkl. `True_*`-Spalten)
   - `03_alldof_rmse.csv` – RMSE über den gesamten exportierten Bereich (Übersichtswert)
   - `03_alldof_parameters.csv` – **nur `numqa`**: MAE / RMSE / max. absoluter Fehler je DoF,
     ausschließlich innerhalb des Messfensters
   - `metadata.json` – Config-Metadaten + Sync-Zeitstempel + Messfenster
4. `numqa` führt die `03_alldof_parameters.csv` aller Messungen zu einer Linac-Tabelle
   `data/process/L<n>/ETDS_L<n>_numqa_<Datum>.csv` zusammen (Spalten: DoF, Unit, Deflection,
   Pad_Temperature, Meas_Couch_Type, Couch_Angle, MAE, RMSE, MaxAE, N_Samples).
5. `plotqa` erzeugt je Temperatursetting eine A4-Seite mit 8 Feldern: sechs DoF-Felder
   (Longitudinal, Lateral, Vertical, Roll, Pitch, Yaw) mit beiden Deflections auf gemeinsamer,
   auf den ersten Sync-Puls bezogener Zeitachse (grau schraffiert = Messfenster, gepunktet =
   Sync-Pulse), ein RMSE-Feld und die Sphere-Detection-Vergleichstabelle.
6. **`create_report.py`** setzt aus den Ergebnissen von `numqa` + `plotqa` den jährlichen
   QA-Report als PDF zusammen (siehe [Report](#report)).

> `numqa` und `plotqa` werten per Absprache nur `single angle`-Messreihen aus – die Couch-Serie ist
> laut Messprotokoll für die jährliche QA nicht erforderlich. Die Spalten `Meas_Couch_Type` und
> `Couch_Angle` sind trotzdem in der Tabelle enthalten, damit `multi angle` später ohne Schemawechsel
> dazukommen kann.

### Vorzeichen-Konvention

Die Vertikalachse des ETD-Systems zeigt entgegengesetzt zu der der Phantom-Kinematik (und zur
Sphere-Detection-Z-Achse). `qa_metrics.DOF_SPEC` ist die einzige Stelle, an der diese Zuordnung
(Phantom-Spalte, ETD-Spalte, Vorzeichen, Einheit) definiert ist; Plots, RMSE-Export und `numqa`
greifen alle darauf zu, damit die Konvention nicht auseinanderlaufen kann.

### Sphere-Detection

Die `*_QA.csv` neben jeder SURF-Messdatei enthält die röntgenbasiert eingemessene Kugelposition an
definierten Punkten. `sphere_detection.py` paart jede `Verschub`-Zeile mit der davorliegenden
Nullpositions-Zeile (Couch-Serie über `R_pos` unterschieden) und stellt gegenüber:
SD-Referenzkoordinaten, SD-Bewegungsvektor, Phantom-Bewegungsvektor (Kinematik auf `H/V/R_pos`),
Bewegungsfehler (SD − Phantom) und Surface-Tracking-Vektor (ETD am Ende des Messfensters).

## Report

```bash
python etds_qa_evaluation.py plotqa 1     # Plotseiten + 02-CSVs
python etds_qa_evaluation.py numqa 1      # Passraten- und Fehler-Tabellen
python create_report.py 1                 # PDF-Report für Linac 1
```

`create_report.py` sucht in `data/process/L<n>/` automatisch die neueste Auswertung (oder die von
`--date YYYY-MM-DD`), füllt die LaTeX-Vorlage in `report/template/` und kompiliert mit XeLaTeX
(`fontspec`) nach `report/output/ETDS_L<n>_QA_Report_<Datum>.pdf`. Aufbau: Overview-Seite,
zwei Detailseiten (Passraten je Messung, Fehlermetriken) und angehängt je eine Plotseite für RT
und 32 °C aus `data/process/L<n>/single_angle/`.

| Quelle | füllt |
|---|---|
| `ETDS_L<n>_passrate_summary_<Datum>.csv` | Passraten je DoF + Ampel-Box (Seite 1) |
| `ETDS_L<n>_numqa_<Datum>.csv` | max. MAE/RMSE/Absolutfehler (Seite 1), Fehlertabelle (Seite 3) |
| `ETDS_L<n>_passrate_<Datum>.csv` | Passraten je Messung (Seite 2) |
| `single_angle/L<n>_<RT\|32>_Couch*_<Datum>_QA.pdf` | Plotseiten im Anhang |
| `report/report_config.json` | „Tested by", Institution, Phantom (derzeit Dummy-Werte) |
| `report/history/L<n>_history.csv` | Vorjahreswerte im Jahresvergleich (aktuell leer) |

Bewertung je Zeile (1-mm-Kriterium): **Action** ab einem Action-Punkt oder pass < 95 %, **Watch**
bei 95–99 % pass mit 1–5 % watch, sonst **Pass**. Die Ampel-Box auf Seite 1 zeigt das schlechteste
Ergebnis aller sechs Freiheitsgrade, den größten absoluten Fehler und die über alle DoF gepoolte
Gesamt-Passrate. Die Toleranzwerte stammen aus `qa_metrics.TOLERANCE_ACCEPT/WATCH`.

`--author "Max Mustermann"` überschreibt den Namen einmalig, `--out` das Ziel-PDF, `--keep-build`
behält die Hilfsdateien in `report/build/L<n>/` für die Fehlersuche.

## Repo-Struktur

```
DataConverter.py           Kernklasse ETDQAProcessor: CSV/JSON laden, Zeit-Alignment, Kinematik, Export
etds_qa_evaluation.py      CLI + Orchestrierung (process/plotpaper/plotqa/numqa) + Paper-Plot
qa_metrics.py              DOF_SPEC (Zuordnung/Vorzeichen), Messfenster, MAE/RMSE/MaxAE, Tabellen-Aggregation
sphere_detection.py        Auswertung der *_QA.csv (röntgenbasierte Kugelposition) + Vergleichstabelle
qa_plots.py                A4-QA-Übersicht (8 Felder) für den plotqa-Modus
create_report.py           Baut den Jahres-QA-Report (LaTeX -> PDF) aus den numqa/plotqa-Ergebnissen
report/template/           LaTeX-Vorlage (main.tex + Report/*.tex + Logo), Platzhalter <<...>>
report/report_config.json  Report-Kopfdaten (Tested by, Institution, Phantom)
report/history/            Vorjahreswerte je Linac für den Jahresvergleich
report/output/             Fertige Report-PDFs
report/build/              Zwischenstand des LaTeX-Laufs (wird bei jedem Lauf neu erzeugt)
etds_qa_2026_config.json   Messungs-Metadaten (Linac, Deflection, Couch-Typ/-Winkel, Heizpads, Zeitstempel)
zoom_box_config.json       Persistierte Zoom-Box-Layouts der interaktiven Paper-Plots (pro Messreihe)
kinematics.py              Vorwärtskinematik Motorachsen -> klinische 6-DoF-Koordinaten (ohne Fehler)
kinematics_werror.py       Dieselbe Kinematik inkl. Unsicherheitsfortpflanzung (Kleinwinkel-Näherung, v1)
kinematics_werror_v2.py    Aktuelle Kinematik: exakte Rotationskomposition + Fehlerfortpflanzung
quick_plotter.py           Standalone Plotly-Viewer: ETD vs. Phantom vor/nach Alignment (6 Subplots)
old_scripts/               Vorgängerversionen (batch_evaluation, calculate_rmse, raw_data_export)
data/raw/L<n>/
  surf_phantom/*.csv        SURF-Terminal-Rohlogs (Motorpositionen, Temperaturen), 10 Hz
  surf_phantom/*_QA.csv     Sphere-Detection-Punkte (röntgenbasierte Referenzkoordinaten)
  etds_scans/*.json + .png  ETD-Tracking-Rohlogs (Shift-Werte, RMSE) + Screenshot je Scan
data/process/L<n>/
  <single_angle|multi_angle>/L<n>_<RT|32>_Deflection<N>_Couch<Winkel>_<Datum>/
                             Exportierte Zwischenstände je Messung (01/02/03 + metadata.json)
  ETDS_L<n>_numqa_<Datum>.csv  Zusammengeführte QA-Tabelle des Linacs
data/measurement.json       (Altbestand / Referenz einer Einzelmessung)
```

## Setup

```bash
pip install -r requirements.txt
```

Benötigt außerdem `kaleido` für den PNG-Export der Plotly-Plots (`export_all_plots`) sowie ein
funktionierendes LaTeX-freies Matplotlib-Backend für die interaktiven PDF-Plots.
