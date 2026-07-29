"""
Numerische QA-Auswertung: Fehlermetriken zwischen ETD-Tracking (Ist) und Phantom-Kinematik (Soll).

Ablauf der Auswertungskette (dokumentiert die Zwischenschritte):

  raw CSV + raw JSON
        |  DataConverter.align_and_crop_signals  -> gemeinsame Zeitbasis + Sync-Zeitstempel
        v
  01_after_align_*.csv          (ausgerichtet, noch ohne Kinematik)
        |  DataConverter.apply_kinematics        -> True_* Spalten (klinische Koordinaten)
        v
  02_aligned_kin_applied_*.csv  (Basis dieser Auswertung)
        |  qa_metrics.compute_dof_metrics        -> nur innerhalb des Messfensters
        v
  03_alldof_parameters.csv      (MAE / RMSE / MaxAE je DoF)
        |  qa_metrics.build_numqa_table
        v
  ETDS_L<n>_numqa_<datum>.csv   (Gesamttabelle je Linac)
"""

import numpy as np
import pandas as pd

# Vorlaufzeit nach dem ersten bzw. vor dem letzten Sync-Puls, die nicht mitgewertet wird.
DEFAULT_WINDOW_MARGIN_SEC = 3.0

# Toleranzschwellen fuer die Pass/Watch/Act-Klassifikation des absoluten Fehlers je Sample.
# |AE| <= ACCEPT: pass (gruen). ACCEPT < |AE| <= WATCH: watch (gelb). |AE| > WATCH: act (rot).
# Einheit ist die des jeweiligen DoF (mm fuer Translation, Grad fuer Rotation).
TOLERANCE_ACCEPT = 1.0
TOLERANCE_WATCH = 2.0


class DofSpec:
    """Zuordnung eines Freiheitsgrads zwischen Phantom- und ETD-Datensatz.

    etd_sign ist der entscheidende Punkt: die Vertikalachse des ETD-Systems zeigt
    entgegengesetzt zur Vertikalachse der Phantom-Kinematik (und zur Sphere-Detection-Z-Achse).
    Ohne diese Korrektur ergibt sich fuer 'vertical' ein scheinbarer Fehler in der
    Groessenordnung der Auslenkung selbst statt der tatsaechlichen ~0.03 mm.
    """

    __slots__ = ('name', 'label', 'phantom_col', 'etd_col', 'etd_sign', 'unit')

    def __init__(self, name, label, phantom_col, etd_col, etd_sign, unit):
        self.name = name
        self.label = label
        self.phantom_col = phantom_col
        self.etd_col = etd_col
        self.etd_sign = etd_sign
        self.unit = unit

    def etd_values(self, df_etd):
        """ETD-Rohwerte in die Vorzeichen-Konvention der Phantom-Kinematik gebracht."""
        return self.etd_sign * df_etd[self.etd_col].values


# Einzige Quelle der Wahrheit fuer die DoF-Zuordnung. Reihenfolge = Plot-Reihenfolge (oben nach unten).
DOF_SPEC = [
    DofSpec('longitudinal', 'Longitudinal (Y)', 'True_Longitudinal', 'longitudinal', +1.0, 'mm'),
    DofSpec('lateral', 'Lateral (X)', 'True_Lateral', 'lateral', +1.0, 'mm'),
    DofSpec('vertical', 'Vertical (Z)', 'True_Vertical', 'vertical', -1.0, 'mm'),
    DofSpec('roll', 'Roll', 'True_Roll', 'roll', +1.0, 'deg'),
    DofSpec('pitch', 'Pitch', 'True_Pitch', 'pitch', +1.0, 'deg'),
    DofSpec('yaw', 'Yaw', 'True_Yaw', 'yaw', +1.0, 'deg'),
]

DOF_BY_NAME = {d.name: d for d in DOF_SPEC}

# Reihenfolge der Translations-DoF fuer Vektor-Tripel (x, y, z)
VECTOR_DOFS = ('lateral', 'longitudinal', 'vertical')


def measurement_window(sync_info, margin_sec=DEFAULT_WINDOW_MARGIN_SEC):
    """Zeitfenster der eigentlichen Messung in der gemeinsamen (ausgerichteten) Zeitbasis.

    Definition: beginnt margin_sec nach der Mitte des ersten Sync-Pulses und endet
    margin_sec vor der Mitte des letzten Sync-Pulses. Der Abstand sorgt dafuer, dass die
    Sync-Bewegung selbst (und ihr Ein-/Ausschwingen) nicht in die Fehlermetriken eingeht.

    sync_info: dict wie von DataConverter.align_and_crop_signals erzeugt bzw. aus metadata.json.
    """
    if not sync_info:
        raise ValueError(
            "Keine Sync-Informationen vorhanden. Bitte zuerst 'process' oder 'plotpaper'/'plotqa' "
            "ausfuehren - dabei werden die Sync-Zeitstempel in die metadata.json geschrieben.")

    t_start = float(sync_info['aligned_first_mid']) + margin_sec
    t_end = float(sync_info['aligned_last_mid']) - margin_sec

    if not t_end > t_start:
        raise ValueError(
            f"Messfenster leer: Sync-Pulse liegen nur {t_end - t_start + 2 * margin_sec:.1f}s "
            f"auseinander, das ist zu wenig fuer margin_sec={margin_sec}.")

    return t_start, t_end


def sample_etd_at(df_etd, dof, t):
    """ETD-Wert eines DoF zu einem Zeitpunkt t (lineare Interpolation, Konvention korrigiert)."""
    return float(np.interp(t, df_etd['Time_Sec'].values, dof.etd_values(df_etd)))


def sample_phantom_at(df_phantom, dof, t):
    """Phantom-Sollwert eines DoF zu einem Zeitpunkt t (lineare Interpolation)."""
    return float(np.interp(t, df_phantom['Time_Sec'].values, df_phantom[dof.phantom_col].values))


def compute_dof_metrics(df_phantom, df_etd, window, dofs=None):
    """MAE / RMSE / maximaler absoluter Fehler je DoF, ausschliesslich im Messfenster.

    Vorgehen je DoF:
      1. Phantom-Stuetzstellen (Soll) innerhalb des Fensters auswaehlen.
      2. ETD-Signal (Ist) auf genau diese Zeitpunkte interpolieren - die beiden Systeme
         haben unterschiedliche Abtastraten (~10 Hz Phantom, ~12 Hz ETD).
      3. Differenz = Ist(ETD) - Soll(Phantom), daraus die drei Metriken.

    Rueckgabe: DataFrame mit einer Zeile je DoF.
    """
    dofs = dofs or DOF_SPEC
    t_start, t_end = window

    t_ph = df_phantom['Time_Sec'].values
    mask = (t_ph >= t_start) & (t_ph <= t_end)
    if not mask.any():
        raise ValueError(f"Keine Phantom-Messpunkte im Fenster [{t_start:.2f}, {t_end:.2f}]s.")

    t_eval = t_ph[mask]
    t_etd = df_etd['Time_Sec'].values

    rows = []
    for dof in dofs:
        if dof.phantom_col not in df_phantom.columns or dof.etd_col not in df_etd.columns:
            continue

        soll = df_phantom[dof.phantom_col].values[mask]
        ist = np.interp(t_eval, t_etd, dof.etd_values(df_etd))
        diff = ist - soll

        rows.append({
            'DoF': dof.name,
            'Unit': dof.unit,
            'Mean_Absolute_Error': np.mean(np.abs(diff)),
            'RMSE': np.sqrt(np.mean(diff ** 2)),
            'Max_Absolute_Error': np.max(np.abs(diff)),
            'N_Samples': int(mask.sum()),
            'Window_Start_Sec': t_start,
            'Window_End_Sec': t_end,
        })

    return pd.DataFrame(rows)


def classify_ae(ae_values, accept=TOLERANCE_ACCEPT, watch=TOLERANCE_WATCH):
    """Pass/Watch/Act je Sample, anhand des absoluten Fehlers (elementweise)."""
    ae = np.asarray(ae_values, dtype=float)
    return np.where(ae <= accept, 'pass', np.where(ae <= watch, 'watch', 'act'))


def compute_ae_time_series(df_phantom, df_etd, window, dofs=None):
    """Absoluter Fehler |Ist(ETD) - Soll(Phantom)| je DoF und Zeitpunkt, nur im Messfenster.

    Selbe Interpolationslogik wie compute_dof_metrics (ETD auf die Phantom-Stuetzstellen
    interpoliert), nur dass hier jeder einzelne Zeitpunkt statt einer Kennzahl behalten wird -
    Basis fuer die Toleranzbaender/rot markierten Zeitfenster im plotqa-Plot und fuer die
    Pass/Watch-Raten. Wird als 03_alldof_ae_time.csv exportiert.

    Rueckgabe: DataFrame mit Time_Sec, AE_<dof> (Betrag) und State_<dof> (pass/watch/act).
    """
    dofs = dofs or DOF_SPEC
    t_start, t_end = window

    t_ph = df_phantom['Time_Sec'].values
    mask = (t_ph >= t_start) & (t_ph <= t_end)
    if not mask.any():
        raise ValueError(f"Keine Phantom-Messpunkte im Fenster [{t_start:.2f}, {t_end:.2f}]s.")

    t_eval = t_ph[mask]
    t_etd = df_etd['Time_Sec'].values

    out = {'Time_Sec': t_eval}
    for dof in dofs:
        if dof.phantom_col not in df_phantom.columns or dof.etd_col not in df_etd.columns:
            continue
        soll = df_phantom[dof.phantom_col].values[mask]
        ist = np.interp(t_eval, t_etd, dof.etd_values(df_etd))
        ae = np.abs(ist - soll)
        out[f'AE_{dof.name}'] = ae
        out[f'State_{dof.name}'] = classify_ae(ae)

    return pd.DataFrame(out)


def find_out_of_tolerance_intervals(ae_time_df, dof_name, state='act'):
    """Zusammenhängende Zeitintervalle [(t_start, t_end), ...], in denen State_<dof>==state ist.

    Wird genutzt, um die rot markierten Zeitbereiche im plotqa-Plot zu bestimmen.
    """
    col = f'State_{dof_name}'
    if ae_time_df is None or col not in ae_time_df.columns or ae_time_df.empty:
        return []

    t = ae_time_df['Time_Sec'].values
    is_state = (ae_time_df[col].values == state)
    if not is_state.any():
        return []

    edges = np.diff(is_state.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1

    intervals = []
    for s, e in zip(starts, ends):
        e = min(e, len(t) - 1)
        intervals.append((float(t[s]), float(t[e])))
    return intervals


def compute_pass_watch_rates(ae_time_df, dofs=None):
    """Pass-/Watch-/Act-Rate (%) je DoF, aus einer compute_ae_time_series-Tabelle."""
    dofs = dofs or DOF_SPEC
    rows = []
    for dof in dofs:
        state_col = f'State_{dof.name}'
        if state_col not in ae_time_df.columns:
            continue
        states = ae_time_df[state_col].values
        n = len(states)
        if n == 0:
            continue
        rows.append({
            'DoF': dof.name,
            'Unit': dof.unit,
            'PassRate_AE': 100.0 * np.sum(states == 'pass') / n,
            'WatchRate_AE': 100.0 * np.sum(states == 'watch') / n,
            'ActRate_AE': 100.0 * np.sum(states == 'act') / n,
            'N_Samples': n,
        })
    return pd.DataFrame(rows)


def build_passrate_table(records):
    """Fuehrt die Pass/Watch-Raten aller Messungen zu einer Linac-Tabelle zusammen.

    records: Liste von dicts mit 'meta' (metadata.json-Inhalt) und 'rates'
             (DataFrame aus compute_pass_watch_rates). Gleicher Aufbau/Sortierung wie
             build_numqa_table, nur mit den Raten-Spalten statt MAE/RMSE/MaxAE.
    """
    rows = []
    for rec in records:
        meta = rec['meta']
        pads = meta.get('heatingpads')
        for _, r in rec['rates'].iterrows():
            rows.append({
                'DoF': r['DoF'],
                'Unit': r['Unit'],
                'Deflection': meta.get('deflection'),
                'Pad_Temperature': 'RT' if pads == 'OFF' else f"{pads}C",
                'Meas_Couch_Type': meta.get('meas_couch_type'),
                'Couch_Angle': meta.get('couch_angle'),
                'PassRate_AE': r['PassRate_AE'],
                'WatchRate_AE': r['WatchRate_AE'],
                'ActRate_AE': r['ActRate_AE'],
                'N_Samples': r['N_Samples'],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    dof_order = {d.name: i for i, d in enumerate(DOF_SPEC)}
    df['_dof_order'] = df['DoF'].map(dof_order)
    df = df.sort_values(['_dof_order', 'Pad_Temperature', 'Deflection']).drop(columns='_dof_order')
    return df.reset_index(drop=True)


def pool_ae_time_series(ae_time_dfs, dofs=None):
    """Gesamt-Passrate je DoF, gepoolt über mehrere AE-Zeitreihen (z.B. alle Deflection/Pad-
    Kombinationen eines Linacs).

    Wichtig: hier werden die ROHEN Pass/Watch/Act-Punkte aus allen Messungen zusammengezählt,
    nicht die bereits pro Messung berechneten Prozentsätze gemittelt. Ein Mittelwert über
    Prozentsätze würde jede Messung gleich gewichten, unabhängig von ihrer Punktzahl (z.B.
    85 vs. 199 Samples) - das Pooling der Rohpunkte gewichtet automatisch korrekt nach
    Stichprobengröße.
    """
    dofs = dofs or DOF_SPEC
    rows = []
    for dof in dofs:
        state_col = f'State_{dof.name}'
        all_states = np.concatenate([df[state_col].values for df in ae_time_dfs if state_col in df.columns])
        if len(all_states) == 0:
            continue
        n = len(all_states)
        rows.append({
            'DoF': dof.name,
            'Unit': dof.unit,
            'N_Total': n,
            'N_Pass': int(np.sum(all_states == 'pass')),
            'N_Watch': int(np.sum(all_states == 'watch')),
            'N_Act': int(np.sum(all_states == 'act')),
            'PassRate_AE': 100.0 * np.sum(all_states == 'pass') / n,
            'WatchRate_AE': 100.0 * np.sum(all_states == 'watch') / n,
            'ActRate_AE': 100.0 * np.sum(all_states == 'act') / n,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    dof_order = {d.name: i for i, d in enumerate(DOF_SPEC)}
    df['_dof_order'] = df['DoF'].map(dof_order)
    df = df.sort_values('_dof_order').drop(columns='_dof_order')
    return df.reset_index(drop=True)


def build_numqa_table(records):
    """Fuehrt die 03_alldof_parameters.csv aller Messungen zu einer Linac-Tabelle zusammen.

    records: Liste von dicts mit den Schluesseln 'meta' (metadata.json-Inhalt) und
             'metrics' (DataFrame aus compute_dof_metrics).

    Sortierung: DoF-weise (alle Deflections/Pad-Settings eines Freiheitsgrads beieinander),
    das ist die Leserichtung einer QA-Tabelle.
    """
    rows = []
    for rec in records:
        meta = rec['meta']
        pads = meta.get('heatingpads')
        for _, m in rec['metrics'].iterrows():
            rows.append({
                'DoF': m['DoF'],
                'Unit': m['Unit'],
                'Deflection': meta.get('deflection'),
                'Pad_Temperature': 'RT' if pads == 'OFF' else f"{pads}C",
                'Meas_Couch_Type': meta.get('meas_couch_type'),
                'Couch_Angle': meta.get('couch_angle'),
                'Mean_Absolute_Error': m['Mean_Absolute_Error'],
                'RMSE': m['RMSE'],
                'Max_Absolute_Error': m['Max_Absolute_Error'],
                'N_Samples': m['N_Samples'],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    dof_order = {d.name: i for i, d in enumerate(DOF_SPEC)}
    df['_dof_order'] = df['DoF'].map(dof_order)
    df = df.sort_values(['_dof_order', 'Pad_Temperature', 'Deflection']).drop(columns='_dof_order')
    return df.reset_index(drop=True)
