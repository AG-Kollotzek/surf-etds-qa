"""
Auswertung der Sphere-Detection-Referenzpunkte (*_QA.csv des SURF-Terminals).

Bei der Messung wird die Stahlkugel im Phantomzentrum an definierten Stellen ueber die
stereoskopische kV-Bildgebung des ExacTrac eingemessen ("sphere detection"). Damit gibt es
eine dritte, roentgenbasierte Positionsangabe neben Phantom-Kinematik (Soll) und
Surface-Tracking (Ist).

Struktur der *_QA.csv (eine Zeile je Messpunkt):
    Time_Sec;Point_ID;Mode;H_pos;V_pos;R_pos;Sphere_X;Sphere_Y;Sphere_Z;...

    0_Referenz               H,V,R = 0,0,0        <- Referenz fuer Deflection 1
    1_minVerschub            H,V,R = -10,20,~1    <- Deflection 1
    2_zeroposition           H,V,R = 0,0,0        <- Referenz fuer Deflection 2
    3_MaxVerschub            H,V,R = -30,50,~2    <- Deflection 2
    4_zeroposition           H,V,R = 0,0,0
    (bei Couch-Messungen folgen dieselben 5 Punkte nochmal mit R_pos ~ 90)

Jede Verschub-Zeile wird also mit der unmittelbar davor liegenden Nullpositions-Zeile gepaart.

Berechnete Groessen je Deflection:
    SD Reference Coords     (Sphere_X, Sphere_Y, Sphere_Z) der Referenzzeile
    SD Movement Vector      Sphere(Verschub) - Sphere(Referenz)          [roentgenbasiert]
    Phantom Movement Vector Kinematik(Verschub) - Kinematik(Referenz)    [Soll aus Achswerten]
    Movement Error Vector   SD Movement - Phantom Movement
    Surface Tracking Vector ETD-Shift am Ende des Messfensters           [Surface-Tracking]
"""

import numpy as np
import pandas as pd
from uncertainties import unumpy as unp

from kinematics_werror_v2 import SurfKinematics
from qa_metrics import DOF_BY_NAME, VECTOR_DOFS

# Ab diesem |R_pos| gilt eine Zeile als zur gedrehten Couch-Serie gehoerig.
_COUCH_R_THRESHOLD_DEG = 45.0


def qa_csv_path_for(surf_csv_path):
    """Pfad der zugehoerigen Sphere-Detection-Datei (gleicher Name + '_QA')."""
    p = str(surf_csv_path)
    if p.endswith('.csv'):
        p = p[:-4]
    return p + '_QA.csv'


def load_sphere_detection(path):
    """Laedt eine *_QA.csv. Leere Sphere-Spalten bleiben NaN."""
    df = pd.read_csv(path, sep=';', decimal='.')
    for col in ('Sphere_X', 'Sphere_Y', 'Sphere_Z', 'H_pos', 'V_pos', 'R_pos'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _is_zero_position(row):
    """Nullposition: alle Achsen auf 0 bzw. (bei gedrehter Couch) nur die Couch-Basisrotation."""
    point_id = str(row.get('Point_ID', '')).lower()
    if 'zeroposition' in point_id or 'referenz' in point_id:
        return True
    return abs(row.get('H_pos', 0.0)) < 1e-6 and abs(row.get('V_pos', 0.0)) < 1e-6


def _couch_group_mask(df, couch_angle):
    """Zeilen der gesuchten Couch-Serie. R_pos ist der Rohwert des Terminals:
    ~0-2 Grad bei Couch 0, ~90-92 Grad bei der -90 Grad-Serie."""
    r = df['R_pos'].abs()
    if abs(float(couch_angle)) < _COUCH_R_THRESHOLD_DEG:
        return r < _COUCH_R_THRESHOLD_DEG
    return r >= _COUCH_R_THRESHOLD_DEG


def select_deflection_rows(df_sd, couch_angle, deflection):
    """Liefert (referenz_zeile, verschub_zeile) fuer eine Deflection einer Couch-Serie.

    deflection 1 -> 'minVerschub', deflection 2 -> 'MaxVerschub'.
    Als Referenz dient die letzte Nullpositions-Zeile vor der Verschub-Zeile.
    """
    group = df_sd[_couch_group_mask(df_sd, couch_angle)]
    if group.empty:
        raise ValueError(f"Keine Sphere-Detection-Zeilen fuer couch_angle={couch_angle} gefunden.")

    keyword = 'minverschub' if int(deflection) == 1 else 'maxverschub'
    hits = [i for i, row in group.iterrows() if keyword in str(row.get('Point_ID', '')).lower()]
    if not hits:
        raise ValueError(f"Keine '{keyword}'-Zeile fuer Deflection {deflection} "
                         f"(couch_angle={couch_angle}) gefunden.")
    defl_idx = hits[0]

    ref_idx = None
    for i in group.index:
        if i >= defl_idx:
            break
        if _is_zero_position(group.loc[i]):
            ref_idx = i
    if ref_idx is None:
        raise ValueError(f"Keine Nullpositions-Zeile vor '{keyword}' "
                         f"(couch_angle={couch_angle}) gefunden.")

    return group.loc[ref_idx], group.loc[defl_idx]


def _sphere_vector(row):
    return np.array([row.get('Sphere_X', np.nan),
                     row.get('Sphere_Y', np.nan),
                     row.get('Sphere_Z', np.nan)], dtype=float)


def _phantom_vector(row, couch_angle, terminal_version='legacy'):
    """Klinische Koordinaten (x, y, z) fuer die Achswerte einer Sphere-Detection-Zeile.

    Wichtig: R_pos steht in der *_QA.csv als Rohwert des Terminals. Im Legacy-Modus wird
    Pos_R beim Laden der Messdaten invertiert (DataConverter.load_csv) - dieselbe Korrektur
    muss hier angewendet werden, sonst dreht die Rotation in die falsche Richtung.
    """
    r_pos = float(row['R_pos'])
    if terminal_version == 'legacy':
        r_pos = -r_pos

    kin = SurfKinematics()
    res = kin.calculate_task_space(np.array([float(row['H_pos'])]),
                                   np.array([float(row['V_pos'])]),
                                   np.array([r_pos]),
                                   float(couch_angle))
    return np.array([
        unp.nominal_values(res['True_Lateral'])[0],
        unp.nominal_values(res['True_Longitudinal'])[0],
        unp.nominal_values(res['True_Vertical'])[0],
    ], dtype=float)


def surface_tracking_vector(df_etd, t_eval):
    """ETD-Translationsvektor (x, y, z) zum Zeitpunkt t_eval, in Phantom-Konvention.

    t_eval ist ueblicherweise das Ende des Messfensters (letzter Sync-Puls minus 3 s), also
    der eingeschwungene Zustand in der ausgelenkten Position. Die ETD-Shiftwerte starten zu
    Scan-Beginn bei 0, der Wert entspricht damit direkt dem Bewegungsvektor.
    """
    t = df_etd['Time_Sec'].values
    out = []
    for name in VECTOR_DOFS:
        dof = DOF_BY_NAME[name]
        out.append(float(np.interp(t_eval, t, dof.etd_values(df_etd))))
    return np.array(out, dtype=float)


def fmt_vector(vec, decimals=2):
    """Tripel als '(x, y, z)' formatieren; fehlende Werte werden als '-' dargestellt."""
    if vec is None:
        return '-'
    parts = []
    for v in vec:
        parts.append('-' if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{decimals}f}")
    return '(' + ', '.join(parts) + ')'


def build_sphere_detection_table(sd_csv_path, deflection_runs, terminal_version='legacy'):
    """Vergleichstabelle der Sphere-Detection-Punkte fuer alle Deflections einer Messreihe.

    deflection_runs: Liste von dicts mit
        'deflection'   : 1 oder 2
        'couch_angle'  : Couch-Winkel der Messung
        'df_etd'       : ausgerichteter ETD-DataFrame (02_..._etd.csv)
        't_eval'       : Zeitpunkt fuer den Surface-Tracking-Vektor (Ende Messfenster)

    Rueckgabe: DataFrame mit einer Zeile je Deflection, Vektoren als formatierte Tripel.
    """
    df_sd = load_sphere_detection(sd_csv_path)

    rows = []
    for run in sorted(deflection_runs, key=lambda r: r['deflection']):
        ref_row, defl_row = select_deflection_rows(df_sd, run['couch_angle'], run['deflection'])

        sd_ref = _sphere_vector(ref_row)
        sd_move = _sphere_vector(defl_row) - sd_ref
        phantom_move = (_phantom_vector(defl_row, run['couch_angle'], terminal_version)
                        - _phantom_vector(ref_row, run['couch_angle'], terminal_version))
        error_move = sd_move - phantom_move
        surface_vec = surface_tracking_vector(run['df_etd'], run['t_eval'])

        rows.append({
            'Deflection': run['deflection'],
            'SD - Reference Coordinates': fmt_vector(sd_ref),
            'SD - Movement Vector': fmt_vector(sd_move),
            'Phantom Movement Vector': fmt_vector(phantom_move),
            'Movement Error Vector': fmt_vector(error_move),
            'Surface Tracking Vector': fmt_vector(surface_vec),
        })

    return pd.DataFrame(rows)
