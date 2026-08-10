"""Kalibrierung des `radius`-Hebelarms gegen die roentgenbasierte Sphere-Detection.

HINTERGRUND
    Der aus Bauteilmassen aufsummierte `radius` in kinematics_werror_v2.SurfKinematics war
    systematisch zu klein. Folge: die Kinematik unterschaetzt `True_Longitudinal` mit der
    Auslenkung wachsend um 0.4-1.3 mm. geometry_diagnostic.py hatte das bereits eingegrenzt
    (Test A/Winkel unauffaellig, Test B/Distanz auffaellig), konnte den Betrag aber nur aus
    einer einzelnen Messreihe schaetzen.

    Jetzt liegen von allen vier Linacs stereoskopische kV-Einmessungen der Phantomkugel vor
    ("sphere detection", *_QA.csv). Die sind unabhaengig von Surface-Tracking und
    Achs-Sollwerten und taugen damit als Referenz fuer den Hebelarm.

METHODE
    Je Messpunkt wird der Bewegungsvektor Kugel(Verschub) - Kugel(Nullposition) mit dem
    Modellvektor verglichen. Weil bei V=0 exakt pitch_local=0 gilt, faellt die Referenzlage
    aus der Rechnung heraus - der Vergleich misst also direkt den Hebelarm.

    `radius` geht linear in die Translation ein:
        y_local = -(radius * sin(pitch)) + rollOffset - h*cos(pitch)
        z_local = -(radius * (1 - cos(pitch))) - h*sin(pitch)
    Ein Zuschlag delta wirkt damit als -delta*sin(pitch) auf y und -delta*(1-cos(pitch)) auf z.
    delta wird je Linac als Least-Squares ueber beide Kanaele und alle Punkte bestimmt.

    Nur 'single angle'-Eintraege der Config werden verwendet (kein multi angle) - bei
    gedrehter Couch mischen sich Couch-Winkelfehler in die Translation und wuerden den
    Hebelarm verfaelschen.

    Hinweis zur Gewichtung: die z-Empfindlichkeit ist sehr klein (1-cos(pitch) = 0.005..0.035
    pro mm gegenueber sin(pitch) = 0.10..0.26 bei y). Bei 0.1 mm Quantisierung der
    Sphere-Detection traegt z praktisch keine Information bei - im Least-Squares wird es
    ueber das Quadrat seiner Empfindlichkeit automatisch entsprechend klein gewichtet. Die
    Spalte 'd (nur z)' der Ausgabe zeigt das deutlich und dient nur als Plausibilitaetscheck.

NUTZUNG
    python radius_calibration.py            # Tabelle + Mittelwert
    python radius_calibration.py --csv      # zusaetzlich als CSV speichern

    Der ermittelte Mittelwert ist als SD_CALIB_DEFAULT in kinematics_werror_v2 hinterlegt.
    Dieses Skript rechnet immer gegen das UNkalibrierte Modell (sd_calib=0) und bleibt damit
    reproduzierbar, auch nachdem der Wert dort eingetragen wurde.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from uncertainties import unumpy as unp

from kinematics_werror_v2 import SurfKinematics, SD_CALIB_DEFAULT
from sphere_detection import load_sphere_detection, select_deflection_rows

CONFIG_JSON_PATH = "etds_qa_2026_config.json"


def single_angle_tables(full_config):
    """{linac: {surf_timestamp: [deflection, ...]}} fuer alle 'single angle'-Eintraege."""
    out = {}
    for meta in full_config.values():
        if meta.get("meas_couch_type") != "single angle":
            continue
        out.setdefault(meta["Linac"], {}) \
           .setdefault(meta["surf_timestamp"], []).append(int(meta["deflection"]))
    return out


def qa_path(linac, surf_stamp):
    hits = list(Path(f"data/raw/L{linac}/surf_phantom").rglob(f"*{surf_stamp}_QA.csv"))
    if not hits:
        raise FileNotFoundError(f"Keine _QA.csv fuer L{linac}, surf_timestamp {surf_stamp}.")
    return hits[0]


def model_yz(h, v, r_pos, sd_calib, terminal_version='legacy'):
    """(y, z) des Modells fuer eine Achsstellung. sd_calib als Radius-Zuschlag in mm."""
    if terminal_version == 'legacy':
        r_pos = -r_pos  # wie DataConverter.load_csv (Pos_R invertiert)
    kin = SurfKinematics(sd_calib=sd_calib)
    res = kin.calculate_task_space(np.array([h]), np.array([v]), np.array([r_pos]), 0.0)
    return (float(unp.nominal_values(res['True_Longitudinal'])[0]),
            float(unp.nominal_values(res['True_Vertical'])[0]))


def collect_points(full_config):
    """Ein Eintrag je (Linac, Messreihe, Deflection) mit gemessenem und modelliertem Vektor."""
    points = []
    for linac, tables in sorted(single_angle_tables(full_config).items()):
        for surf_stamp, deflections in sorted(tables.items()):
            df_sd = load_sphere_detection(qa_path(linac, surf_stamp))
            for deflection in sorted(set(deflections)):
                ref, defl = select_deflection_rows(df_sd, couch_angle=0.0, deflection=deflection)
                h, v, r = float(defl['H_pos']), float(defl['V_pos']), float(defl['R_pos'])

                # gemessen (Roentgen): Verschub minus Nullposition
                sd_y = float(defl['Sphere_Y']) - float(ref['Sphere_Y'])
                sd_z = float(defl['Sphere_Z']) - float(ref['Sphere_Z'])

                # Modell unkalibriert, und Empfindlichkeit pro mm delta
                y0, z0 = model_yz(h, v, r, sd_calib=0.0)
                y1, z1 = model_yz(h, v, r, sd_calib=1.0)
                ref_y0, ref_z0 = model_yz(float(ref['H_pos']), float(ref['V_pos']),
                                          float(ref['R_pos']), sd_calib=0.0)
                ref_y1, ref_z1 = model_yz(float(ref['H_pos']), float(ref['V_pos']),
                                          float(ref['R_pos']), sd_calib=1.0)

                points.append({
                    'linac': linac, 'surf': surf_stamp, 'deflection': deflection,
                    'H': h, 'V': v,
                    'sd_y': sd_y, 'sd_z': sd_z,
                    'mod_y': y0 - ref_y0, 'mod_z': z0 - ref_z0,
                    'sens_y': (y1 - ref_y1) - (y0 - ref_y0),
                    'sens_z': (z1 - ref_z1) - (z0 - ref_z0),
                })
    return points


def fit_delta(points, channels='yz'):
    """Least-Squares-Zuschlag auf radius (mm) ueber die gewaehlten Kanaele."""
    num = den = 0.0
    for p in points:
        for ch in channels:
            err, sens = p[f'sd_{ch}'] - p[f'mod_{ch}'], p[f'sens_{ch}']
            num += err * sens
            den += sens * sens
    return num / den if den else float('nan')


def rms(points, delta):
    r = []
    for p in points:
        for ch in ('y', 'z'):
            r.append(p[f'sd_{ch}'] - (p[f'mod_{ch}'] + delta * p[f'sens_{ch}']))
    return float(np.sqrt(np.mean(np.square(r))))


def main():
    with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
        full_config = json.load(f)

    points = collect_points(full_config)
    linacs = sorted({p['linac'] for p in points})

    print(f"\nSphere-Detection-Punkte: {len(points)} "
          f"({len(linacs)} Linacs, nur 'single angle')\n")

    print("=" * 78)
    print("KALIBRIERFAKTOR JE LINAC   (Zuschlag auf radius, in mm)")
    print("=" * 78)
    print(f"{'Linac':<8}{'n':>3}{'delta [mm]':>12}{'d (nur y)':>11}{'d (nur z)':>11}"
          f"{'RMS vor':>10}{'RMS nach':>10}")
    print("-" * 78)

    table = []
    for linac in linacs:
        sub = [p for p in points if p['linac'] == linac]
        d = fit_delta(sub)
        row = {
            'Linac': f"L{linac}", 'n_points': len(sub),
            'delta_mm': round(d, 3),
            'delta_y_only_mm': round(fit_delta(sub, 'y'), 3),
            'delta_z_only_mm': round(fit_delta(sub, 'z'), 3),
            'rms_before_mm': round(rms(sub, 0.0), 3),
            'rms_after_mm': round(rms(sub, d), 3),
        }
        table.append(row)
        print(f"{row['Linac']:<8}{row['n_points']:>3}{row['delta_mm']:>12.3f}"
              f"{row['delta_y_only_mm']:>11.3f}{row['delta_z_only_mm']:>11.3f}"
              f"{row['rms_before_mm']:>10.3f}{row['rms_after_mm']:>10.3f}")

    vals = np.array([r['delta_mm'] for r in table], dtype=float)
    mean, sd = float(vals.mean()), float(vals.std(ddof=1))
    print("-" * 78)
    print(f"{'MITTEL':<8}{len(points):>3}{mean:>12.3f}"
          f"{'':>11}{'':>11}{rms(points, 0.0):>10.3f}{rms(points, mean):>10.3f}")
    print(f"\n  SD zwischen den Linacs : {sd:.3f} mm")
    print(f"  Spanne                 : {vals.min():.3f} ... {vals.max():.3f} mm "
          f"(= {vals.ptp():.3f} mm)")
    print(f"  -> SD_CALIB_DEFAULT    : ufloat({mean:.3f}, {sd:.3f})")

    hinterlegt = float(unp.nominal_values(SD_CALIB_DEFAULT))
    status = "stimmt ueberein" if abs(hinterlegt - mean) < 5e-3 else "WEICHT AB"
    print(f"  in kinematics_werror_v2: {hinterlegt:.3f} mm  ({status})")

    print("\n" + "=" * 78)
    print("EINZELPUNKTE - Residuum longitudinal [mm]")
    print("=" * 78)
    print(f"{'Linac':<8}{'surf':<9}{'d':<3}{'H':>5}{'V':>5}"
          f"{'gemessen':>10}{'Modell':>9}{'vor':>8}{'nach':>8}")
    print("-" * 78)
    for p in points:
        after = p['sd_y'] - (p['mod_y'] + mean * p['sens_y'])
        print(f"L{p['linac']:<7}{p['surf']:<9}{p['deflection']:<3}{p['H']:>5.0f}{p['V']:>5.0f}"
              f"{p['sd_y']:>10.2f}{p['mod_y']:>9.2f}{p['sd_y']-p['mod_y']:>8.2f}{after:>8.2f}")

    if "--csv" in sys.argv:
        out = Path("report/radius_calibration.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(table)
        df.loc[len(df)] = {'Linac': 'MITTEL', 'n_points': len(points),
                           'delta_mm': round(mean, 3), 'delta_y_only_mm': None,
                           'delta_z_only_mm': None,
                           'rms_before_mm': round(rms(points, 0.0), 3),
                           'rms_after_mm': round(rms(points, mean), 3)}
        df.to_csv(out, index=False, sep=';', decimal='.')
        print(f"\n[✓] Tabelle gespeichert: {out}")


if __name__ == "__main__":
    main()
