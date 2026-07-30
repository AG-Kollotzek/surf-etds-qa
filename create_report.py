"""Erzeugt den jaehrlichen ETDS-QA-Report (PDF) aus den Auswertungen in data/process.

Aufruf:
    python create_report.py 1                  # Linac 1, neueste Auswertung
    python create_report.py 1 --date 2026-07-29
    python create_report.py 1 --author "Max Mustermann"

Quellen (alle aus data/process/L<linac>/):
    ETDS_L<n>_passrate_summary_<datum>.csv   -> Passraten je DOF (Seite 1)
    ETDS_L<n>_passrate_<datum>.csv           -> Passraten je Messung (Seite 2)
    ETDS_L<n>_numqa_<datum>.csv              -> MAE/RMSE/Max-Fehler (Seite 3)
    single_angle/L<n>_{RT,32}_Couch*_<datum>_QA.pdf -> Plotseiten (Anhang)

Die .tex-Vorlagen liegen in report/template und enthalten <<PLATZHALTER>>, die hier
ersetzt werden. Kompiliert wird mit XeLaTeX (fontspec) in report/build/L<n>.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

import qa_metrics

# ==========================================
# KONFIGURATION & GLOBALE VARIABLEN
# ==========================================
PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_JSON_PATH = PROJECT_DIR / "etds_qa_2026_config.json"
REPORT_DIR = PROJECT_DIR / "report"
TEMPLATE_DIR = REPORT_DIR / "template"
BUILD_ROOT = REPORT_DIR / "build"
OUTPUT_DIR = REPORT_DIR / "output"
HISTORY_DIR = REPORT_DIR / "history"
REPORT_CONFIG_PATH = REPORT_DIR / "report_config.json"

# Reihenfolge wie im Muster-Report: erst Translationen, dann Rotationen.
DOF_ORDER = ['longitudinal', 'lateral', 'vertical', 'roll', 'pitch', 'yaw']
# Zeilenreihenfolge innerhalb eines DOF-Blocks: (Deflection, Pad-Temperatur)
MEAS_ORDER = [(1, '32C'), (2, '32C'), (1, 'RT'), (2, 'RT')]

# Fallbacks fuer Felder, die sich (noch) nicht aus den Daten ableiten lassen.
DUMMY_DEFAULTS = {
    "author": "Vorname Nachname",
    "institution": "tirol kliniken",
    "phantom": "SURF",
}

DESCRIPTION_TEXT = (
    "The aim of the annual quality assurance (QA) measurements is to evaluate the surface "
    "scanning accuracy of the ExacTrac Dynamic system using a mechanically movable and heated "
    "phantom. {n_series} measurement series are performed, comprising two different deflections: "
    "Deflection 1, in which the phantom is displaced in smaller increments (up to 20 mm and "
    "1{deg}), and Deflection 2, with larger increments (up to 50 mm and 2{deg}). Each deflection "
    "is carried out with both unheated and heated heating pads. The phantom is displaced "
    "according to predefined translational and rotational motion sequences. Before, between, and "
    "after each motion sequence, the position of the sphere, located at the center of the "
    "phantom, initially aligned with the treatment isocenter, is determined using sphere "
    "detection to verify the system's localization accuracy as part of the annual QA procedure."
)

NUM_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six",
             7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten"}


# ==========================================
# HILFSFUNKTIONEN (Formatierung & Bewertung)
# ==========================================
def fmt_rate(value):
    """Prozentraten werden im Report ganzzahlig dargestellt (wie im Muster-Report)."""
    return f"{float(value):.0f}"


def fmt_metric(value, digits=3):
    return f"{float(value):.{digits}f}"


def fmt_number(value):
    """Zahl ohne unnoetige Nachkommastelle: 100.0 -> '100', 96.55 -> '96.5'."""
    text = f"{float(value):.1f}"
    return text[:-2] if text.endswith(".0") else text


def unit_tex(unit):
    """CSV-Einheit -> LaTeX ('deg' wird als Gradzeichen gesetzt)."""
    return r"$^\circ$" if str(unit).strip().lower() in ("deg", "degree", "°") else str(unit)


def classify_rates(pass_rate, watch_rate, act_rate):
    """Gesamtbewertung einer Zeile nach den festgelegten Kriterien (1-mm-Kriterium).

    action : mindestens ein Action-Punkt (act > 0) ODER pass < 95 %
    watch  : 95 % <= pass < 99 % UND 1 % <= watch <= 5 % UND kein Action-Punkt
    pass   : sonst (>= 99 % pass, <= 1 % watch, 0 action)
    """
    if act_rate > 0 or pass_rate < 95:
        return 'action'
    if 95 <= pass_rate < 99 and 1 <= watch_rate <= 5:
        return 'watch'
    return 'pass'


VERDICT_CELL = {'pass': r"\PassCell", 'watch': r"\WatchCell", 'action': r"\ActionCell"}
VERDICT_COLOR = {'pass': "passgreen", 'watch': "watchyellow", 'action': "actionred"}
VERDICT_LABEL = {'pass': "Pass", 'watch': "Watch", 'action': "Action"}
VERDICT_RANK = {'pass': 0, 'watch': 1, 'action': 2}


# ==========================================
# QUELLEN FINDEN
# ==========================================
def _date_from_name(path):
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    return match.group(1) if match else ""


def find_latest_csv(process_dir, prefix, date_str=None, exclude=None):
    """Neueste (oder eine bestimmte) Auswertungs-CSV, z.B. 'ETDS_L1_passrate_summary'.

    'exclude' filtert Namensbestandteile heraus - noetig, weil das Praefix
    'ETDS_L1_passrate' auch auf die Summary-Datei passt.
    """
    candidates = sorted(process_dir.glob(f"{prefix}_*.csv"), key=_date_from_name)
    if exclude:
        candidates = [p for p in candidates if exclude not in p.name]
    if date_str:
        candidates = [p for p in candidates if _date_from_name(p) == date_str]
    if not candidates:
        hint = f" mit Datum {date_str}" if date_str else ""
        raise FileNotFoundError(f"Keine Datei '{prefix}_<datum>.csv'{hint} in {process_dir} gefunden. "
                                f"Bitte zuerst 'python etds_qa_evaluation.py numqa <linac>' ausfuehren.")
    return candidates[-1]


def find_latest_plot(single_angle_dir, linac, pad_folder, date_str=None):
    """Neueste QA-Plotseite (A4-PDF) fuer ein Temperatursetting, bevorzugt Couch 0."""
    candidates = sorted(single_angle_dir.glob(f"L{linac}_{pad_folder}_Couch*_QA.pdf"),
                        key=_date_from_name)
    if date_str:
        candidates = [p for p in candidates if _date_from_name(p) == date_str]
    if not candidates:
        return None
    newest_date = _date_from_name(candidates[-1])
    same_date = [p for p in candidates if _date_from_name(p) == newest_date]
    couch0 = [p for p in same_date if "_Couch0_" in p.name]
    return (couch0 or same_date)[-1]


def measurement_datetime(linac):
    """Messdatum/-zeit aus den Roh-Scans: etds_timestamp der Config -> TrackingResult-Datei.

    Rueckgabe (datum, zeit) als String oder (None, None), wenn nichts zuzuordnen ist.
    """
    scans_dir = PROJECT_DIR / f"data/raw/L{linac}/etds_scans"
    if not (CONFIG_JSON_PATH.exists() and scans_dir.exists()):
        return None, None

    with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
        full_config = json.load(f)

    stamps = [str(meta.get("etds_timestamp", "")) for meta in full_config.values()
              if str(meta.get("Linac")) == str(linac) and meta.get("meas_couch_type") == "single angle"]

    found = []
    for stamp in stamps:
        if len(stamp) != 6:
            continue
        pattern = f"TrackingResult_*_{stamp[0:2]}-{stamp[2:4]}-{stamp[4:6]}.json"
        for hit in scans_dir.glob(pattern):
            match = re.search(r"TrackingResult_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})", hit.name)
            if match:
                found.append(datetime.strptime(f"{match.group(1)} {match.group(2)}:{match.group(3)}",
                                               "%Y-%m-%d %H:%M"))
    if not found:
        return None, None
    first = min(found)
    return first.strftime("%d/%m/%Y"), first.strftime("%H:%M")


def load_history(linac):
    """Vorjahreswerte aus report/history/L<n>_history.csv (optional, ';'-getrennt).

    Zeilen, die mit '#' beginnen, werden ignoriert (Format-Beispiele in der Datei).
    """
    path = HISTORY_DIR / f"L{linac}_history.csv"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("year"):
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) >= 4:
            rows.append(parts[:4])
    return rows


# ==========================================
# TABELLEN BAUEN
# ==========================================
def sort_by_dof(df):
    """Zeilen in Report-Reihenfolge sortieren (DOF-Block, darin Deflection/Temperatur)."""
    order = {name: i for i, name in enumerate(DOF_ORDER)}
    meas = {key: i for i, key in enumerate(MEAS_ORDER)}
    df = df.copy()
    df['_dof'] = df['DoF'].map(lambda d: order.get(d, len(order)))
    if 'Deflection' in df.columns:
        df['_meas'] = [meas.get((int(d), str(t)), len(meas))
                       for d, t in zip(df['Deflection'], df['Pad_Temperature'])]
        return df.sort_values(['_dof', '_meas']).drop(columns=['_dof', '_meas'])
    return df.sort_values('_dof').drop(columns=['_dof'])


def build_summary_table(df_summary):
    """Passraten je DOF (Seite 1) inkl. Gesamtbewertung fuer die Ampel-Box."""
    lines = []
    verdicts = []
    for _, row in sort_by_dof(df_summary).iterrows():
        verdict = classify_rates(row['PassRate_AE'], row['WatchRate_AE'], row['ActRate_AE'])
        verdicts.append(verdict)
        lines.append(
            f"\\textcolor{{extdlabel}}{{{row['DoF']}}} & {fmt_rate(row['PassRate_AE'])} & "
            f"{fmt_rate(row['WatchRate_AE'])} & {fmt_rate(row['ActRate_AE'])} & "
            f"{VERDICT_CELL[verdict]} \\\\")
    overall = max(verdicts, key=lambda v: VERDICT_RANK[v]) if verdicts else 'pass'
    # Gesamt-Passrate ueber alle DOF: Rohpunkte poolen, nicht Prozentwerte mitteln.
    pooled = 100.0 * df_summary['N_Pass'].sum() / df_summary['N_Total'].sum()
    return "\n".join(lines), overall, pooled


def build_passrate_detail_table(df_rates):
    """Passraten je Messung (Seite 2), pro DOF durch \\midrule getrennt."""
    lines = []
    for i, (_, row) in enumerate(sort_by_dof(df_rates).iterrows()):
        if i and i % len(MEAS_ORDER) == 0:
            lines.append(r"\midrule")
        verdict = classify_rates(row['PassRate_AE'], row['WatchRate_AE'], row['ActRate_AE'])
        lines.append(
            f"\\textcolor{{extdlabel}}{{{row['DoF']}}} & {int(row['Deflection'])} & "
            f"{row['Pad_Temperature']} & {fmt_rate(row['PassRate_AE'])} & "
            f"{fmt_rate(row['WatchRate_AE'])} & {fmt_rate(row['ActRate_AE'])} & "
            f"{VERDICT_CELL[verdict]} \\\\")
    return "\n".join(lines)


def build_error_metrics_table(df_num):
    """MAE/RMSE/Max-Fehler je Messung (Seite 3), pro DOF durch \\midrule getrennt."""
    lines = []
    for i, (_, row) in enumerate(sort_by_dof(df_num).iterrows()):
        if i and i % len(MEAS_ORDER) == 0:
            lines.append(r"\midrule")
        lines.append(
            f"\\textcolor{{extdlabel}}{{{row['DoF']}}} & {unit_tex(row['Unit'])} & "
            f"{int(row['Deflection'])} & {row['Pad_Temperature']} & "
            f"{fmt_metric(row['Mean_Absolute_Error'])} & {fmt_metric(row['RMSE'])} & "
            f"{fmt_metric(row['Max_Absolute_Error'])} \\\\")
    return "\n".join(lines)


def max_metrics(df_num):
    """Groesste MAE/RMSE/Max-Absolut-Werte ueber alle Freiheitsgrade inkl. Einheit."""
    result = {}
    for key, column in (('mae', 'Mean_Absolute_Error'), ('rmse', 'RMSE'), ('max', 'Max_Absolute_Error')):
        row = df_num.loc[df_num[column].idxmax()]
        result[key] = (float(row[column]), unit_tex(row['Unit']))
    return result


def build_history_table(linac, year, maxima):
    """Vorjahresvergleich: gepflegte Historie + automatisch die Werte des aktuellen Laufs."""
    lines = []
    for row in load_history(linac):
        lines.append(" & ".join(row) + r" \\")
    lines.append(f"{year} & {fmt_metric(maxima['mae'][0])} {maxima['mae'][1]} & "
                 f"{fmt_metric(maxima['rmse'][0])} {maxima['rmse'][1]} & "
                 f"{fmt_metric(maxima['max'][0])} {maxima['max'][1]} \\\\")
    return "\n".join(lines)


def build_plot_pages(plot_files):
    """\\includepdf-Bloecke fuer die angehaengten QA-Plotseiten (Fusszeile bleibt erhalten)."""
    lines = []
    for name in plot_files:
        lines.append(r"\includepdf[pages=1-,scale=0.90,offset=0 12pt,"
                     r"pagecommand={\thispagestyle{fancy}}]{plots/" + name + "}")
    return "\n".join(lines) if lines else "% keine QA-Plotseiten gefunden"


# ==========================================
# VORLAGEN FUELLEN & KOMPILIEREN
# ==========================================
def render_templates(build_dir, replacements):
    """Vorlage nach build_dir kopieren und alle <<PLATZHALTER>> ersetzen."""
    if build_dir.exists():
        shutil.rmtree(build_dir)
    shutil.copytree(TEMPLATE_DIR, build_dir)

    unresolved = set()
    for tex in build_dir.rglob("*.tex"):
        text = tex.read_text(encoding="utf-8")
        for key, value in replacements.items():
            text = text.replace(f"<<{key}>>", str(value))
        unresolved.update(re.findall(r"<<([A-Z_]+)>>", text))
        tex.write_text(text, encoding="utf-8")
    if unresolved:
        print(f"   [!] Unersetzte Platzhalter: {', '.join(sorted(unresolved))}")


def compile_pdf(build_dir):
    """XeLaTeX-Lauf (fontspec). latexmk bevorzugt, sonst zwei xelatex-Durchgaenge."""
    if shutil.which("latexmk"):
        commands = [["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]]
    elif shutil.which("xelatex"):
        commands = [["xelatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]] * 2
    else:
        raise RuntimeError("Weder 'latexmk' noch 'xelatex' gefunden - bitte TeX-Distribution installieren.")

    for cmd in commands:
        proc = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
        if proc.returncode != 0:
            log = build_dir / "main.log"
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:] if log.exists() else []
            raise RuntimeError("LaTeX-Lauf fehlgeschlagen:\n" + "\n".join(tail) + "\n" + proc.stdout[-2000:])

    pdf = build_dir / "main.pdf"
    if not pdf.exists():
        raise RuntimeError(f"Kein PDF erzeugt in {build_dir}.")
    return pdf


# ==========================================
# HAUPTPROGRAMM
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Erzeugt den jaehrlichen ETDS-QA-Report als PDF.")
    parser.add_argument("linac", help="Linac-ID, z.B. 1")
    parser.add_argument("--date", help="Auswertungsdatum YYYY-MM-DD (Standard: neueste Auswertung)")
    parser.add_argument("--author", help="Name unter 'Tested by' (Standard: aus report_config.json)")
    parser.add_argument("--out", help="Zielpfad des PDFs (Standard: report/output/...)")
    parser.add_argument("--keep-build", action="store_true", help="Build-Ordner nicht aufraeumen")
    args = parser.parse_args()

    linac = args.linac
    process_dir = PROJECT_DIR / f"data/process/L{linac}"
    if not process_dir.exists():
        print(f"[X] Kein Auswertungsordner {process_dir}.")
        return 1

    settings = dict(DUMMY_DEFAULTS)
    if REPORT_CONFIG_PATH.exists():
        with open(REPORT_CONFIG_PATH, "r", encoding="utf-8") as f:
            settings.update(json.load(f))
    if args.author:
        settings["author"] = args.author

    # ---------- Quellen einlesen ----------
    try:
        summary_csv = find_latest_csv(process_dir, f"ETDS_L{linac}_passrate_summary", args.date)
        rates_csv = find_latest_csv(process_dir, f"ETDS_L{linac}_passrate", args.date, exclude="summary")
        numqa_csv = find_latest_csv(process_dir, f"ETDS_L{linac}_numqa", args.date)
    except FileNotFoundError as e:
        print(f"[X] {e}")
        return 1

    df_summary = pd.read_csv(summary_csv, sep=';', decimal='.')
    df_rates = pd.read_csv(rates_csv, sep=';', decimal='.')
    df_num = pd.read_csv(numqa_csv, sep=';', decimal='.')

    print(f"[i] Passraten je DOF   : {summary_csv.name}")
    print(f"[i] Passraten je Messung: {rates_csv.name}")
    print(f"[i] Fehlermetriken      : {numqa_csv.name}")

    # ---------- Plotseiten ----------
    single_angle_dir = process_dir / "single_angle"
    plot_files = []
    for pad_folder in ("RT", "32"):
        source = find_latest_plot(single_angle_dir, linac, pad_folder, args.date)
        if source is None:
            print(f"   [!] Keine QA-Plotseite fuer Pads={pad_folder} gefunden "
                  f"(erwartet: L{linac}_{pad_folder}_Couch*_QA.pdf in {single_angle_dir}). "
                  f"Bitte 'python etds_qa_evaluation.py plotqa {linac}' ausfuehren.")
            continue
        plot_files.append((f"L{linac}_{pad_folder}_QA.pdf", source))
        print(f"[i] Plotseite {pad_folder:>2}          : {source.name}")

    # ---------- Inhalte bauen ----------
    summary_rows, overall_verdict, pooled_passrate = build_summary_table(df_summary)
    maxima = max_metrics(df_num)
    meas_date, meas_time = measurement_datetime(linac)
    if meas_date is None:
        meas_date, meas_time = "TT/MM/JJJJ", "HH:MM"   # Dummy: kein Rohscan zugeordnet
        print("   [!] Messdatum/-zeit nicht aus data/raw ableitbar - Dummy eingesetzt.")
    data_date = _date_from_name(numqa_csv)
    year = data_date[:4] if data_date else datetime.now().strftime("%Y")
    now = datetime.now()
    n_series = len(df_rates.groupby(['Deflection', 'Pad_Temperature']))

    replacements = {
        "LINAC": linac,
        "MEAS_DATE": meas_date,
        "MEAS_TIME": meas_time,
        "CREATION_DATE": now.strftime("%d/%m/%Y"),
        "CREATION_TIME": now.strftime("%H:%M"),
        "AUTHOR": settings["author"],
        "INSTITUTION": settings["institution"],
        "PHANTOM": settings["phantom"],
        "OVERALL_PASSRATE": fmt_number(pooled_passrate),
        "VERDICT_COLOR": VERDICT_COLOR[overall_verdict],
        "VERDICT_LABEL": VERDICT_LABEL[overall_verdict],
        "MAX_ABS_ERROR": fmt_metric(maxima['max'][0]),
        "MAX_ABS_ERROR_UNIT": f"{fmt_metric(maxima['max'][0])} {maxima['max'][1]}",
        "MAX_MAE": f"{fmt_metric(maxima['mae'][0])} {maxima['mae'][1]}",
        "MAX_RMSE": f"{fmt_metric(maxima['rmse'][0])} {maxima['rmse'][1]}",
        "TOL_ACCEPT": fmt_number(qa_metrics.TOLERANCE_ACCEPT),
        "TOL_WATCH": fmt_number(qa_metrics.TOLERANCE_WATCH),
        "TABLE_PASSRATE_SUMMARY": summary_rows,
        "TABLE_PASSRATE_DETAIL": build_passrate_detail_table(df_rates),
        "TABLE_ERROR_METRICS": build_error_metrics_table(df_num),
        "TABLE_HISTORY": build_history_table(linac, year, maxima),
        "DESCRIPTION": DESCRIPTION_TEXT.format(n_series=NUM_WORDS.get(n_series, n_series),
                                               deg=r"$^\circ$"),
        "PLOT_PAGES": build_plot_pages([name for name, _ in plot_files]),
    }

    # ---------- Rendern & kompilieren ----------
    build_dir = BUILD_ROOT / f"L{linac}"
    render_templates(build_dir, replacements)
    (build_dir / "plots").mkdir(exist_ok=True)
    for name, source in plot_files:
        shutil.copyfile(source, build_dir / "plots" / name)

    try:
        pdf = compile_pdf(build_dir)
    except RuntimeError as e:
        print(f"[X] {e}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = Path(args.out) if args.out else OUTPUT_DIR / f"ETDS_L{linac}_QA_Report_{data_date}.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf, target)

    if not args.keep_build:
        for junk in set(build_dir.glob("main.*")) | set(build_dir.rglob("*.aux")):
            if junk.suffix not in (".tex", ".pdf"):
                junk.unlink()

    print(f"\n[✓] Report gespeichert: {target}")
    print(f"    Gesamtbewertung: {VERDICT_LABEL[overall_verdict]} "
          f"(Passrate gesamt {fmt_number(pooled_passrate)} %)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
