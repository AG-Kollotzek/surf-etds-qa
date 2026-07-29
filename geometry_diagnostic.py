"""
Kleiner Diagnose-Test: welche Geometrie-Gruppe erklärt die wachsende Y/Pitch-Abweichung?

Hintergrund: bei reiner V-Bewegung (H fest) wächst in `True_Longitudinal` ein Restfehler
(ETD - Modell) an, der ~sin(pitch_local) folgt. pitch_local haengt NUR von hAxis_vertical/
hAxis_horizontal ab (ueber alpha_offset und hAxis_diagonal), waehrend die Y/Z-Translation
zusaetzlich radius = rotatationTable_height + phantomToTableDistance + phantomCenterDistance
+ sliderShift + hAxis_vertical enthaelt. Nur Pitch und Longitudinal reagieren ueberhaupt so
stark auf V - alle anderen DoF sind von diesem Term kaum/nicht betroffen, daher zeigen sie den
Fehler auch nicht.

Zwei unabhaengige Tests trennen die beiden Parameter-Gruppen:

  Test A (Winkel):    ETD-eigener 'pitch'-Wert vs. modelliertes pitch_local, ueber den
                       gesamten V-Ramp. R ist bei mindev/maxdev ~0-2 Grad, ETD's pitch ist
                       dort praktisch eine direkte, radius-unabhaengige Messung von
                       pitch_local. Baseline bei V=0 UND Steigung ueber den Ramp pruefen:
                         - Baseline != 0 bei V=0            -> alpha_offset / hAxis_vertical:horizontal-Verhaeltnis
                         - Steigung btw. deutlich != 1        -> hAxis_diagonal (hAxis_vertical/horizontal Betrag)
                       Bereits gepruefte Referenzmessungen (2026-07-15) zeigen: Baseline
                       0.04+/-0.04 Grad (V=0), Steigung 0.999-1.034 -> beides unauffaellig.

  Test B (Distanz):   Residuum (ETD - Modell) in True_Longitudinal vs. sin(pitch_local).
                       Ist die Steigung hier deutlich von 0 verschieden (bei allen bisherigen
                       Messungen der Fall, Vorzeichen konsistent), ist der `radius`-Summand
                       zu klein/gross - das kann Test A nicht erklaeren.

Wenn Test A unauffaellig bleibt (wie bisher) und Test B weiterhin einen konsistenten Ausschlag
zeigt: gezielt rotatationTable_height, phantomToTableDistance (im Code schon als "Nachmessen!"
markiert), phantomCenterDistance und sliderShift nachmessen - nicht hAxis_vertical/horizontal.

Nutzung:
    python geometry_diagnostic.py <linac_id> <deflection: 1/2>

Setzt KEINEN vorherigen 'process'-Lauf voraus - laedt Rohdaten direkt ueber die Config.
"""

import sys
import json
import numpy as np

from DataConverter import ETDQAProcessor
from etds_qa_evaluation import compute_pair_index
from kinematics_werror_v2 import SurfKinematics
from uncertainties import unumpy as unp

CONFIG_JSON_PATH = "etds_qa_2026_config.json"


def load_single_angle_measurement(full_config, target_linac, deflection):
    """Sucht den passenden single-angle-Eintrag (couch_angle=0) fuer Linac + Deflection."""
    for m_id, meta in full_config.items():
        if (meta.get("Linac") == target_linac
                and str(meta.get("deflection")) == str(deflection)
                and meta.get("meas_couch_type") == "single angle"):
            return m_id, meta
    raise ValueError(f"Keine single-angle-Messung fuer Linac {target_linac}, Deflection {deflection} gefunden.")


def find_raw_files(raw_base_dir, meta):
    etd_stamp = meta["etds_timestamp"]
    if len(etd_stamp) == 6 and "-" not in etd_stamp:
        etd_stamp = f"{etd_stamp[:2]}-{etd_stamp[2:4]}-{etd_stamp[4:]}"
    csv_files = [c for c in (raw_base_dir / "surf_phantom").rglob(f"*{meta['surf_timestamp']}.csv")
                 if not c.name.endswith("_QA.csv")]
    json_files = list((raw_base_dir / "etds_scans").rglob(f"*{etd_stamp}.json"))
    if not csv_files or not json_files:
        raise FileNotFoundError(f"Rohdaten fuer Messung fehlen (surf={meta['surf_timestamp']}, etd={meta['etds_timestamp']}).")
    return str(csv_files[0]), str(json_files[0])


def run_diagnostic(target_linac, deflection):
    from pathlib import Path

    with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
        full_config = json.load(f)
    m_id, meta = load_single_angle_measurement(full_config, target_linac, deflection)
    csv_path, json_path = find_raw_files(Path(f"data/raw/L{target_linac}"), meta)

    proc = ETDQAProcessor(terminal_version='legacy')
    proc.load_csv(csv_path)
    proc.load_json(json_path)
    proc.align_and_crop_signals(measurement_group='geometry_diagnostic',
                                pair_idx=compute_pair_index(full_config, m_id),
                                couch_angle=float(meta.get("couch_angle", 0.0)))
    proc.apply_kinematics(couch_angle=float(meta.get("couch_angle", 0.0)))

    ph = proc.df_csv
    et = proc.df_json
    etd_pitch = np.interp(ph['Time_Sec'], et['Time_Sec'], et['pitch'])
    etd_lon = np.interp(ph['Time_Sec'], et['Time_Sec'], et['longitudinal'])

    # pitch_local-Modell separat rekonstruieren (identisch zu SurfKinematics, Schritt 1)
    kin = SurfKinematics()
    alpha_offset = unp.nominal_values(unp.arctan(kin.hAxis_vertical / kin.hAxis_horizontal))
    hv = unp.nominal_values(kin.hAxis_vertical)
    hd = unp.nominal_values(kin.hAxis_diagonal)
    radius_n = unp.nominal_values(kin.radius)
    pitch_local_rad = np.arcsin((hv + ph['Pos_V'].values) / hd) - alpha_offset
    pitch_local_deg = np.degrees(pitch_local_rad)

    h_max = np.abs(ph['Pos_H'].values).max()
    v_max = ph['Pos_V'].values.max()

    print(f"\n=== Geometrie-Diagnose: Linac {target_linac}, Deflection {deflection} (ID {m_id}) ===")
    print(f"Motorbereich: H bis {h_max:.1f}mm, V bis {v_max:.1f}mm\n")

    # --- Test A: Baseline bei V=0 (reine H-Bewegung, pitch_local_model=0 exakt) ---
    mask_v0 = (ph['Pos_V'].values < 0.01) & (np.abs(ph['Pos_H'].values) > 1)
    if mask_v0.sum() >= 5:
        base_mean, base_std = etd_pitch[mask_v0].mean(), etd_pitch[mask_v0].std()
        print(f"Test A1 (Baseline, V=0, n={mask_v0.sum()}):")
        print(f"   ETD pitch = {base_mean:+.3f} +/- {base_std:.3f} deg  (Modell: exakt 0)")
        if abs(base_mean) > 3 * max(base_std, 0.05):
            print("   -> auffaellig: deutet auf alpha_offset / hAxis_vertical:horizontal-Verhaeltnis hin.")
        else:
            print("   -> unauffaellig (Baseline stimmt).")
    else:
        print("Test A1: zu wenige Punkte mit V=0 fuer eine Baseline-Aussage.")

    # --- Test A: Steigung ueber den V-Ramp ---
    mask_ramp = (np.abs(ph['Pos_H'].values) > h_max * 0.95) & \
                (ph['Pos_V'].values > 0.05 * v_max) & (ph['Pos_V'].values < 0.95 * v_max)
    n_ramp = mask_ramp.sum()
    if n_ramp >= 10:
        B, A = np.polyfit(pitch_local_deg[mask_ramp], etd_pitch[mask_ramp], 1)
        print(f"\nTest A2 (Steigung, n={n_ramp}):")
        print(f"   ETD_pitch = {A:+.3f} + {B:.4f} * pitch_local_model")
        if abs(B - 1.0) > 0.05:
            print(f"   -> Steigung {abs(B-1)*100:.1f}% von 1 entfernt: deutet auf hAxis_diagonal "
                  f"(hAxis_vertical/hAxis_horizontal Betrag) hin.")
        else:
            print("   -> unauffaellig (Steigung ~1, hAxis_vertical/horizontal wahrscheinlich in Ordnung).")
    else:
        print("\nTest A2: zu wenige Punkte im reinen V-Ramp fuer eine Steigungs-Aussage.")

    # --- Test B: Y-Residuum vs. sin(pitch_local) im reinen V-Ramp ---
    residual = etd_lon - ph['True_Longitudinal'].values
    if n_ramp >= 10:
        x = np.sin(pitch_local_rad[mask_ramp])
        y = residual[mask_ramp]
        b, a = np.polyfit(x, y, 1)
        delta_radius = -b
        print(f"\nTest B (Y-Residuum vs. sin(pitch_local), n={n_ramp}):")
        print(f"   residual_mm = {a:+.3f} + ({b:+.3f}) * sin(pitch_local)")
        print(f"   -> impliziertes Delta_radius = {delta_radius:+.2f} mm "
              f"(aktuell modelliert: {radius_n:.1f} mm, das sind {100*delta_radius/radius_n:+.1f}%)")
        if abs(delta_radius) > 2.0:
            print("   -> auffaellig: rotatationTable_height, phantomToTableDistance, "
                  "phantomCenterDistance, sliderShift nachmessen (in dieser Reihenfolge testen,\n"
                  "      phantomToTableDistance zuerst - im Code schon als unsicher markiert).")
        else:
            print("   -> unauffaellig.")
    else:
        print("\nTest B: zu wenige Punkte im reinen V-Ramp fuer eine Aussage.")

    print("\nFazit-Regel:")
    print("  Test A auffaellig  -> hAxis_vertical / hAxis_horizontal nachmessen (Winkelgeometrie).")
    print("  nur Test B auffaellig -> radius-Summanden nachmessen (reine Distanz/Hebelarm).")
    print("  Nur Pitch und Longitudinal reagieren hier ueberhaupt auf V - andere DoF sind kein")
    print("  Gegencheck fuer diese beiden Parametergruppen.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Nutzung: python geometry_diagnostic.py <linac_id> <deflection: 1/2>")
        print("Beispiel: python geometry_diagnostic.py 1 2")
        sys.exit(1)
    run_diagnostic(sys.argv[1], sys.argv[2])
