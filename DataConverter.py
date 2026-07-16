import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from kinematics import SurfKinematics as SurfKinematicsNominal
from kinematics_werror import SurfKinematics
from uncertainties import unumpy as unp
import datetime
import os

MEASUREMENT_10032026 = {
    '1': {'Gruppe': 'All axes', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '160617',
          'CSV': '160448', 'title': 'H, V, R'},
    '2': {'Gruppe': 'All axes', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '161117',
          'CSV': '160950', 'title': 'H, V, R'},
    '3': {'Gruppe': 'All axes', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '161416',
          'CSV': '161302', 'title': 'H, V, R'},
    '4': {'Gruppe': 'Horizontal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '161724',
          'CSV': '161538', 'title': 'H'},
    '5': {'Gruppe': 'Horizontal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '162054',
          'CSV': '161859', 'title': 'H'},
    '6': {'Gruppe': 'Horizontal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '162326',
          'CSV': '162130', 'title': 'H'},
    '7': {'Gruppe': 'Vertical', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '162647',
          'CSV': '162451', 'title': 'V'},
    '8': {'Gruppe': 'Vertical', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '162922',
          'CSV': '162728', 'title': 'V'},
    '9': {'Gruppe': 'Vertical', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '163157',
          'CSV': '162951', 'title': 'V'},
    '10': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '163508',
           'CSV': '163325', 'title': 'R'},
    '11': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '163758',
           'CSV': '163613', 'title': 'R'},
    '12': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '164007',
           'CSV': '163820', 'title': 'R'},
    '14': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '173928',
           'CSV': '173645'},
    '15': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '174334',
           'CSV': '174056'},
    '16': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': 'OFF', 'ETD': '174832',
           'CSV': '174558'},
    '17': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area': 'Fitting', 'Heatingpads': 'OFF', 'ETD': '175259',
           'CSV': '175019'},
    '18': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area': 'OnlyFrontSurface', 'Heatingpads': 'OFF', 'ETD': '175635',
           'CSV': '175356'},
    '19': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area': 'Fiting', 'Heatingpads': 'OFF', 'ETD': '180108',
           'CSV': '175824'},
    '21': {'Gruppe': 'Vertical_Slide', 'Heatingpads': 'OFF', 'ETD': '182337', 'CSV': '182011'},
    '22': {'Gruppe': 'Vertical_Slide', 'Heatingpads': 'OFF', 'ETD': '182630', 'CSV': '182433'},
    '24': {'Gruppe': 'Horizontal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '190312',
           'CSV': '190114', 'title': 'H'},
    '25': {'Gruppe': 'Horizontal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '190609',
           'CSV': '190405', 'title': 'H'},
    '26': {'Gruppe': 'Horizontal', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '190831',
           'CSV': '190639', 'title': 'H'},
    '27': {'Gruppe': 'Vertical', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '191212',
           'CSV': '191006', 'title': 'V'},
    '28': {'Gruppe': 'Vertical', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '191421',
           'CSV': '191235', 'title': 'V'},
    '29': {'Gruppe': 'Vertical', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '191651',
           'CSV': '191507', 'title': 'V'},
    '30': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '192007',
           'CSV': '191806', 'title': 'R'},
    '31': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '192224',
           'CSV': '192031', 'title': 'R'},
    '32': {'Gruppe': 'Rotation', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '192441',
           'CSV': '192250', 'title': 'R'},
    '33': {'Gruppe': 'Variable_Geschwindigkeit', 'ROI_Area': 'PhantomWithBuffer', 'Heatingpads': '32', 'ETD': '192853',
           'CSV': '192615'},
    '34': {'Gruppe': 'Vertical_Slide', 'Heatingpads': '32', 'ETD': '193528', 'CSV': '193140'},
    '35': {'Gruppe': 'Vertical_Slide', 'Heatingpads': '32', 'ETD': '193801', 'CSV': '193557'},
}

class ETDQAProcessor:
    def __init__(self, terminal_version='legacy'):
        """
        terminal_version:
            'legacy' -> Invertiert Pos_H und Pos_R (für alte Datensätze)
            'v2'     -> Nutzt Rohdaten wie sie sind (für zukünftige korrigierte Version)
        """
        self.df_csv = None
        self.df_json = None
        self.time_offset = 0
        self.terminal_version = terminal_version

        self.kinematics_werror_dict = {}

    def load_csv(self, file_path):
        """Lädt das SURF-Terminal Log und korrigiert ggf. Vorzeichenfehler"""
        self.csv_filename = os.path.basename(file_path)  # NEU: Dateiname merken
        self.df_csv = pd.read_csv(file_path, sep=';', decimal='.')

        # --- SMART SIGN CORRECTION ---t
        if self.terminal_version == 'legacy':
            # Korrektur für Longitudinal (H) und Rotation (R)
            self.df_csv['Pos_H'] = self.df_csv['Pos_H'] * -1
            self.df_csv['Pos_R'] = self.df_csv['Pos_R'] * -1
            print(f"INFO: Legacy-Mode aktiv. Vorzeichen für Pos_H und Pos_R wurden invertiert.")

        return self.df_csv

    def load_json(self, file_path):
        """Lädt das ETD-Tracking Log und berechnet die 3D-Magnitude inkl. RMSE"""

        self.json_filename = os.path.basename(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)

        results = []
        self.raw_lost_timestamps = []
        for entry in data['trackingResults']:
            if entry.get('trackingLost', False):
                self.raw_lost_timestamps.append(entry['timestamp'])
            else:
                shifts = {k: float(v) for k, v in entry['shiftValues'].items()}
                shifts['timestamp_ms'] = entry['timestamp']

                # --- NEW: Extract RMSE values here ---
                # We use .get() with a fallback to np.nan in case a frame is missing the data
                shifts['rmse3d'] = entry.get('rmse3D', np.nan)
                shifts['rmse_temp'] = entry.get('rmseThermal', np.nan)

                results.append(shifts)

        self.df_json = pd.DataFrame(results)
        t0 = self.df_json['timestamp_ms'].iloc[0]
        self.df_json['Time_Sec'] = (self.df_json['timestamp_ms'] - self.df_json['timestamp_ms'].iloc[0]) / 1000.0
        self.lost_times_sec = [(t - t0) / 1000.0 for t in self.raw_lost_timestamps]

        # Internes Alignment nutzt Magnitude (unabhängig vom Vorzeichen)
        lat0, long0, vert0 = self.df_json['lateral'].iloc[0], self.df_json['longitudinal'].iloc[0], \
            self.df_json['vertical'].iloc[0]
        self.df_json['Vector_Mag'] = np.sqrt(
            (self.df_json['lateral'] - lat0) ** 2 +
            (self.df_json['longitudinal'] - long0) ** 2 +
            (self.df_json['vertical'] - vert0) ** 2
        )
        return self.df_json

    def apply_kinematics(self, couch_angle=0.0):
        """Berechnet klinische Koordinaten und bereitet Fehler-Tubes vor (OHNE Baseline-Nullung)."""
        print("Berechne kinematische Transformation in klinische Koordinaten...")
        kin = SurfKinematics()

        # Berechnung (ohne Nullen)
        res = kin.calculate_task_space(self.df_csv['Pos_H'].values,
                                       self.df_csv['Pos_V'].values,
                                       self.df_csv['Pos_R'].values, couch_angle)

        for key, uarray in res.items():
            self.df_csv[key] = unp.nominal_values(uarray)
            self.df_csv[f"{key}_std"] = unp.std_devs(uarray)
            self.df_csv[f"{key}_upper"] = self.df_csv[key] + self.df_csv[f"{key}_std"]
            self.df_csv[f"{key}_lower"] = self.df_csv[key] - self.df_csv[f"{key}_std"]

    def align_signals(self, sync_axis='Pos_H', threshold=4.5, csv_sync_window=None):
        """Synchronisiert und speichert die exakten Zeiten der Peaks für das spätere Cropping.
           csv_sync_window: (t_min, t_max) in Sekunden, um falsche Peaks in der CSV zu ignorieren."""

        def get_peak_info(times, values, thresh, window=None):
            values_check = values.copy()

            # NEU: Filtere falsche Peaks durch virtuelles Nullen aus
            if window is not None:
                t_min, t_max = window
                mask_out = (times < t_min) | (times > t_max)
                values_check.loc[mask_out] = 0.0

            v_smooth = pd.Series(values_check).rolling(window=5, center=True).median().fillna(0).values
            mask = np.abs(v_smooth) > thresh

            # Finde Kanten
            edges = np.diff(mask.astype(int), prepend=0, append=0)
            starts = np.where(edges == 1)[0]
            # Korrektur: Wir nehmen den Index des letzten 'True' Werts
            ends = np.where(edges == -1)[0] - 1

            if len(starts) < 2:
                return None, None, None, None

            # --- BIAS-FIX: FLOAT-PRÄZISION STATT INT-RUNDUNG ---
            # Wir nehmen den zeitlichen Mittelpunkt zwischen Start und Ende der Flanken
            t_first_mid = (times.iloc[starts[0]] + times.iloc[ends[0]]) / 2.0
            t_last_mid = (times.iloc[starts[-1]] + times.iloc[ends[-1]]) / 2.0

            t_first_start = times.iloc[starts[0]]
            t_last_end = times.iloc[ends[-1]]

            return t_first_mid, t_last_mid, t_first_start, t_last_end

        # Peak-Zeiten ermitteln (CSV übergibt jetzt das window!)
        csv_first, csv_last, csv_start, csv_end = get_peak_info(self.df_csv['Time_Sec'], self.df_csv[sync_axis],
                                                                threshold, window=csv_sync_window)
        json_first, json_last, _, _ = get_peak_info(self.df_json['Time_Sec'], self.df_json['Vector_Mag'], threshold,
                                                    window=None)

        if csv_first is None or json_first is None:
            raise ValueError("Nicht genügend 5mm Peaks für Sync gefunden.")

        scale_factor = (csv_last - csv_first) / (json_last - json_first)
        self.df_json['Time_Sec'] = (self.df_json['Time_Sec'] - json_first) * scale_factor + csv_first
        self.time_offset = 0

        # Tracking-Lost Zeiten an das CSV-Alignment anpassen
        if hasattr(self, 'lost_times_sec') and self.lost_times_sec:
            self.lost_times_aligned = [(t - json_first) * scale_factor + csv_first for t in self.lost_times_sec]
        else:
            self.lost_times_aligned = []

        # WICHTIG: Start und Ende für die Baseline-Korrektur speichern
        self.sync_t_start = csv_start
        self.sync_t_end = csv_end

        print(f"Sync erfolgreich (Skalierung: {scale_factor:.6f}, Referenz-Peak: {csv_first:.2f}s)")
        return scale_factor

    def apply_baseline_and_crop(self):
        """Nullt die Achsen präzise an den Sync-Peaks und schneidet Vorlauf ab."""
        if not hasattr(self, 'sync_t_start'):
            print("FEHLER: Führe zuerst align_signals() aus!")
            return

        t_start = self.sync_t_start
        t_end = self.sync_t_end
        csv_time = self.df_csv['Time_Sec']

        # 1. Smarte Maske: Wir nehmen exakt das Fenster 1 bis 6 Sekunden VOR dem ersten Peak
        # und 1 bis 6 Sekunden NACH dem letzten Peak (um aus den Flanken raus zu sein).
        base_mask = ((csv_time >= t_start - 6.0) & (csv_time <= t_start - 1.0)) | \
                    ((csv_time >= t_end + 1.0) & (csv_time <= t_end + 6.0))

        if not base_mask.any():
            base_mask = (csv_time < t_start)  # Fallback

        print(f"Führe Baseline-Korrektur durch (Referenzfenster: {base_mask.sum()} Punkte)...")

        # 2. Offset von allen kinematischen Spalten (inkl. Schläuche) abziehen
        keys = ['True_Lateral', 'True_Longitudinal', 'True_Vertical', 'True_Pitch', 'True_Roll', 'True_Yaw']
        for key in keys:
            offset = np.median(self.df_csv.loc[base_mask, key])
            self.df_csv[key] -= offset
            self.df_csv[f"{key}_upper"] -= offset
            self.df_csv[f"{key}_lower"] -= offset

        # 3. Cropping: Wir schneiden alles ab, was mehr als 5 Sekunden vor dem ersten Peak liegt
        crop_time = t_start - 5.0
        self.df_csv = self.df_csv[self.df_csv['Time_Sec'] >= crop_time].reset_index(drop=True)
        self.df_json = self.df_json[self.df_json['Time_Sec'] >= crop_time].reset_index(drop=True)

        if hasattr(self, 'lost_times_aligned'):
            self.lost_times_aligned = [t for t in self.lost_times_aligned if t >= crop_time]

        print(f"-> Daten gecroppt. Plot startet nun exakt 5s vor dem ersten Peak.")

    def plot_sync_check(self):
        """Visualisierung aller 6 DoF inkl. Unsicherheits-Schläuchen."""

        def _add_uncertainty_trace(fig, df, key, color, name):
            # 1. Der Unsicherheits-Schlauch (Transparente Fläche)
            fig.add_trace(go.Scatter(
                x=np.concatenate([df['Time_Sec'], df['Time_Sec'][::-1]]),
                y=np.concatenate([df[f"{key}_upper"], df[f"{key}_lower"][::-1]]),
                fill='toself',
                fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                name=f"{name} Uncert."
            ))
            # 2. Die nominelle SURF-Linie (gestrichelt)
            fig.add_trace(go.Scatter(
                x=df['Time_Sec'], y=df[key],
                name=f"SURF {name}",
                line=dict(color=color, width=2, dash='dash')
            ))

        # --- FIGUR 1: TRANSLATIONEN (X, Y, Z) ---
        fig_trans = go.Figure()

        # ETD Daten (Durchgezogen)
        fig_trans.add_trace(go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['lateral'], name="ETD Lateral (X)",
                                       line=dict(color='red')))
        fig_trans.add_trace(go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['longitudinal'], name="ETD Long. (Y)",
                                       line=dict(color='green')))
        fig_trans.add_trace(go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['vertical'], name="ETD Vert. (Z)",
                                       line=dict(color='blue')))

        # SURF Daten (Gestrichelt + Schlauch)
        _add_uncertainty_trace(fig_trans, self.df_csv, 'True_Lateral', 'rgb(255, 0, 0)', 'Lat (X)')
        _add_uncertainty_trace(fig_trans, self.df_csv, 'True_Longitudinal', 'rgb(0, 255, 0)', 'Long (Y)')
        _add_uncertainty_trace(fig_trans, self.df_csv, 'True_Vertical', 'rgb(0, 0, 255)', 'Vert (Z)')

        fig_trans.update_layout(title="6 DoF Check: Translationen", xaxis_title="Zeit [s]", yaxis_title="Position [mm]",
                                hovermode="x unified")
        fig_trans.show()

        # --- FIGUR 2: ROTATIONEN (Pitch, Roll, Yaw) ---
        fig_rot = go.Figure()

        # ETD Daten
        fig_rot.add_trace(go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['pitch'], name="ETD Pitch",
                                     line=dict(color='orange')))
        fig_rot.add_trace(
            go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['roll'], name="ETD Roll", line=dict(color='purple')))
        fig_rot.add_trace(
            go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['yaw'], name="ETD Yaw", line=dict(color='brown')))

        # SURF Daten
        _add_uncertainty_trace(fig_rot, self.df_csv, 'True_Pitch', 'rgb(255, 165, 0)', 'Pitch')
        _add_uncertainty_trace(fig_rot, self.df_csv, 'True_Roll', 'rgb(128, 0, 128)', 'Roll')
        _add_uncertainty_trace(fig_rot, self.df_csv, 'True_Yaw', 'rgb(165, 42, 42)', 'Yaw')

        fig_rot.update_layout(title="6 DoF Check: Rotationen", xaxis_title="Zeit [s]", yaxis_title="Winkel [°]",
                              hovermode="x unified")
        fig_rot.show()

    def evaluate_plateaus_and_export(self, output_file="QA_Evaluation_Report.csv"):
        """
        Erkennt beliebige Endpositionen (Plateaus) anhand des Stillstands der Motoren.
        Filtert die Initialisierungs-Peaks (den ersten und letzten) automatisch heraus.
        """
        import datetime
        print("Starte flexible Plateau-Erkennung (Kinetische Analyse)...")

        times = self.df_csv['Time_Sec'].values

        # 1. Gesamtauslenkung berechnen (Abstand von der Nullposition)
        # Kombiniert alle klinischen Achsen, um JEDE Art von Bewegung (Translation/Rotation) zu erfassen.
        auslenkung = np.sqrt(self.df_csv['True_Lateral'] ** 2 +
                             self.df_csv['True_Longitudinal'] ** 2 +
                             self.df_csv['True_Vertical'] ** 2) + \
                     np.abs(self.df_csv['True_Pitch']) + \
                     np.abs(self.df_csv['True_Roll']) + \
                     np.abs(self.df_csv['True_Yaw'])

        # 2. Geschwindigkeit / Bewegung der Hardware-Achsen berechnen
        # np.diff gibt uns die Änderung zum vorherigen Zeitschritt.
        dH = np.diff(self.df_csv['Pos_H'].values, prepend=self.df_csv['Pos_H'].values[0])
        dV = np.diff(self.df_csv['Pos_V'].values, prepend=self.df_csv['Pos_V'].values[0])
        dR = np.diff(self.df_csv['Pos_R'].values, prepend=self.df_csv['Pos_R'].values[0])

        # Absolute Positionsänderung pro Zeitschritt
        motor_delta = np.abs(dH) + np.abs(dV) + np.abs(dR)
        motor_delta_smooth = pd.Series(motor_delta).rolling(window=3, center=True).median().fillna(0).values

        # 3. Maske für: "Wir stehen an einer Endposition"
        # - motor_delta_smooth < 0.005: Die Achsen bewegen sich nicht (Stillstand).
        # - auslenkung > 0.5: Wir befinden uns fernab der Baseline.
        mask_plateau = (motor_delta_smooth < 0.005) & (auslenkung > 0.5)

        edges = np.diff(mask_plateau.astype(int), prepend=0, append=0)
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0] - 1

        # 4. Sammle alle echten Plateaus
        potential_plateaus = []
        for s, e in zip(starts, ends):
            duration = times[e] - times[s]
            # Wir suchen Plateaus, die durch den Roboter bedingt ~2 Sekunden dauern
            # (Toleranz 1.0 bis 3.5 Sekunden, um minimale Messschwankungen abzufangen)
            if 1.0 <= duration <= 3.5:
                potential_plateaus.append((s, e, duration))

        if not potential_plateaus:
            print("WARNUNG: Keine Plateaus gefunden!")
            return None

        # 5. --- DER WICHTIGSTE SCHRITT: Sync-Peaks entfernen ---
        if len(potential_plateaus) >= 3:
            print(f"-> {len(potential_plateaus)} Plateaus im Signal gefunden. Entferne ersten und letzten Peak.")
            # Wir nehmen alles ab Index 1 bis zum vorletzten Element (Index -1)
            valid_plateaus = potential_plateaus[1:-1]
        else:
            print("WARNUNG: Zu wenige Plateaus gefunden, um Sync-Peaks abzuziehen.")
            valid_plateaus = potential_plateaus

        # 6. Auswertung der übrig gebliebenen (gültigen) Peaks
        dof_mapping = {
            'Lateral_X': ('lateral', 'True_Lateral'),
            'Longitudinal_Y': ('longitudinal', 'True_Longitudinal'),
            'Vertical_Z': ('vertical', 'True_Vertical'),
            'Pitch': ('pitch', 'True_Pitch'),
            'Roll': ('roll', 'True_Roll'),
            'Yaw': ('yaw', 'True_Yaw')
        }

        results = []

        for i, (s, e, duration) in enumerate(valid_plateaus):
            # Wir zentrieren uns auf die Mitte des 2-Sekunden-Plateaus und
            # werten exakt ein 1.2 Sekunden langes Fenster aus (±0.6s).
            t_center = (times[s] + times[e]) / 2.0
            t_start = t_center - 0.6
            t_end = t_center + 0.6
            eval_duration = t_end - t_start

            mask_csv = (self.df_csv['Time_Sec'] >= t_start) & (self.df_csv['Time_Sec'] <= t_end)
            mask_json = (self.df_json['Time_Sec'] >= t_start) & (self.df_json['Time_Sec'] <= t_end)

            peak_stats = {
                ('Meta', 'Peak_ID'): i + 1,
                ('Meta', 'Dauer_s'): round(eval_duration, 2)
            }

            for dof, (col_etd, col_surf) in dof_mapping.items():
                val_etd = self.df_json.loc[mask_json, col_etd].values
                val_surf = self.df_csv.loc[mask_csv, col_surf].values

                if len(val_etd) == 0 or len(val_surf) == 0:
                    peak_stats[(dof, 'Mean_Diff')] = np.nan
                    peak_stats[(dof, 'Std_Diff')] = np.nan
                    continue

                mean_diff = np.mean(val_etd) - np.mean(val_surf)
                std_diff = np.sqrt(np.std(val_etd) ** 2 + np.std(val_surf) ** 2)

                peak_stats[(dof, 'Mean_Diff')] = round(mean_diff, 4)
                peak_stats[(dof, 'Std_Diff')] = round(std_diff, 4)

            results.append(peak_stats)

        # 7. DataFrame erstellen und exportieren
        df_results = pd.DataFrame(results)
        df_results.columns = pd.MultiIndex.from_tuples(df_results.columns)

        mess_tag, csv_code, json_code = "Unbekannt", "Unbekannt", "Unbekannt"
        if hasattr(self, 'csv_filename'):
            try:
                parts = self.csv_filename.replace('.csv', '').split('_')
                if len(parts[-2]) == 8:
                    mess_tag = f"{parts[-2][:4]}-{parts[-2][4:6]}-{parts[-2][6:]}"
                csv_code = parts[-1]
            except Exception:
                pass
        if hasattr(self, 'json_filename'):
            try:
                parts = self.json_filename.replace('.json', '').split('_')
                if len(parts) >= 3:
                    json_code = parts[2].replace('-', '')
            except Exception:
                pass

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# --- SURF-ETD QA Evaluierungsbericht ---\n")
            f.write(f"# Erstellt am: {timestamp}\n")
            f.write(f"# Messung von Tag: {mess_tag}\n")
            f.write(f"# CSV-Code: {csv_code}\n")
            f.write(f"# JSON-Code: {json_code}\n")
            f.write(f"# Auswertung: Dynamische Kinetik-Erkennung\n")
            f.write(f"# Evaluierungsfenster: {round(eval_duration, 2)}s (mittig im Endpunkt zentriert)\n")
            f.write(f"# Alle Translationswerte in [mm], Rotationswerte in [Grad]\n")
            f.write(f"# ---------------------------------------\n")

            df_results.to_csv(f, index=False, lineterminator='\n')

        print(f"-> Report mit {len(results)} validen Messpunkten gespeichert in '{output_file}'.")
        return df_results

    def export_all_plots(self, output_dir, prefix=""):
        """
        Erzeugt drei Plotly-Graphen (Translation, Rotation, Raw-Motors)
        und speichert sie direkt als PNG-Dateien im angegebenen Ordner ab.
        """
        import os
        print("Generiere und speichere Plots...")

        def _add_uncertainty_trace(fig, df, key, color, name):
            fig.add_trace(go.Scatter(
                x=np.concatenate([df['Time_Sec'], df['Time_Sec'][::-1]]),
                y=np.concatenate([df[f"{key}_upper"], df[f"{key}_lower"][::-1]]),
                fill='toself',
                fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip", showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=df['Time_Sec'], y=df[key], name=f"SURF {name}",
                line=dict(color=color, width=2, dash='dash')
            ))

        # --- PLOT 1: TRANSLATIONEN ---
        fig_trans = go.Figure()
        fig_trans.add_trace(
            go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['lateral'], name="ETD X", line=dict(color='red')))
        fig_trans.add_trace(go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['longitudinal'], name="ETD Y",
                                       line=dict(color='green')))
        fig_trans.add_trace(
            go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['vertical'], name="ETD Z", line=dict(color='blue')))
        _add_uncertainty_trace(fig_trans, self.df_csv, 'True_Lateral', 'rgb(255, 0, 0)', 'Lat (X)')
        _add_uncertainty_trace(fig_trans, self.df_csv, 'True_Longitudinal', 'rgb(0, 255, 0)', 'Long (Y)')
        _add_uncertainty_trace(fig_trans, self.df_csv, 'True_Vertical', 'rgb(0, 0, 255)', 'Vert (Z)')
        fig_trans.update_layout(title=f"{prefix}: Translationen (ETD vs SURF)", xaxis_title="Zeit [s]",
                                yaxis_title="Position [mm]")

        # --- PLOT 2: ROTATIONEN ---
        fig_rot = go.Figure()
        fig_rot.add_trace(go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['pitch'], name="ETD Pitch",
                                     line=dict(color='orange')))
        fig_rot.add_trace(
            go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['roll'], name="ETD Roll", line=dict(color='purple')))
        fig_rot.add_trace(
            go.Scatter(x=self.df_json['Time_Sec'], y=self.df_json['yaw'], name="ETD Yaw", line=dict(color='brown')))
        _add_uncertainty_trace(fig_rot, self.df_csv, 'True_Pitch', 'rgb(255, 165, 0)', 'Pitch')
        _add_uncertainty_trace(fig_rot, self.df_csv, 'True_Roll', 'rgb(128, 0, 128)', 'Roll')
        _add_uncertainty_trace(fig_rot, self.df_csv, 'True_Yaw', 'rgb(165, 42, 42)', 'Yaw')
        fig_rot.update_layout(title=f"{prefix}: Rotationen (ETD vs SURF)", xaxis_title="Zeit [s]",
                              yaxis_title="Winkel [°]")

        # --- PLOT 3: RAW MOTOR VALUES DER TEST UNIT ---
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(x=self.df_csv['Time_Sec'], y=self.df_csv['Pos_H'], name="Motor H (Long/Lat)",
                                     line=dict(color='blue')))
        fig_raw.add_trace(go.Scatter(x=self.df_csv['Time_Sec'], y=self.df_csv['Pos_V'], name="Motor V (Vert/Pitch)",
                                     line=dict(color='green')))
        fig_raw.add_trace(go.Scatter(x=self.df_csv['Time_Sec'], y=self.df_csv['Pos_R'], name="Motor R (Rotation)",
                                     line=dict(color='red')))
        fig_raw.update_layout(title=f"{prefix}: Rohdaten TestUnit (Hardware)", xaxis_title="Zeit [s]",
                              yaxis_title="Position [mm / Grad]")

        # --- SPEICHERN ---
        try:
            fig_trans.write_image(os.path.join(output_dir, f"{prefix}_Plot_Translation.png"), scale=2)
            fig_rot.write_image(os.path.join(output_dir, f"{prefix}_Plot_Rotation.png"), scale=2)
            fig_raw.write_image(os.path.join(output_dir, f"{prefix}_Plot_RawMotors.png"), scale=2)
        except ValueError as e:
            print(f"WARNUNG: Plot konnte nicht gespeichert werden. Fehlt 'kaleido'? Error: {e}")

    @staticmethod
    def get_interp_data(df_json, df_csv, t_common):
        """
        Interpoliert JSON- und CSV-Signale auf ein gemeinsames Zeitraster.
        Nutzt die bereits von apply_kinematics berechneten Spalten.
        """
        interp_results = {}

        # 1. ETD (JSON) Signale interpolieren
        for col in ['lateral', 'longitudinal', 'vertical', 'pitch', 'yaw', 'roll']:
            interp_results[f'et_{col}'] = np.interp(t_common, df_json['Time_Sec'], df_json[col])

        # Mapping von deinen CSV-Spaltennamen auf die kurzen Keys
        csv_mapping = {
            'X': 'True_Lateral',
            'Y': 'True_Longitudinal',
            'Z': 'True_Vertical',
            'pitch': 'True_Pitch',
            'roll': 'True_Roll',
            'yaw': 'True_Yaw'
        }

        # 2. Phantom Nominalwerte AND die systematischen Fehler (_std) interpolieren
        for key, csv_name in csv_mapping.items():
            # Nominalwert der Plattform
            interp_results[f'ihd_{key}_nom'] = np.interp(t_common, df_csv['Time_Sec'], df_csv[csv_name])

            # Systematischer Fehler (aus deiner kinematics_werror) direkt aus der _std Spalte laden!
            interp_results[f'ihd_{key}_sys_err'] = np.interp(t_common, df_csv['Time_Sec'], df_csv[f"{csv_name}_std"])

        return interp_results

    @staticmethod
    def bin_and_average_measurements(processed_runs, t_common):
        """
        Mittelt die Durchgänge einer Gruppe (Kopie deiner Logik aus batch_evaluation).
        Berechnet die Standardabweichung (statistischer Fehler) zwischen den Runs.
        """
        mean_curves = {}
        stat_errors = {}

        # Keys, die wir mitteln wollen
        keys_to_average = [
            'et_lateral', 'et_longitudinal', 'et_vertical', 'et_pitch', 'et_yaw', 'et_roll',
            'ihd_X_nom', 'ihd_Y_nom', 'ihd_Z_nom', 'ihd_pitch_nom', 'ihd_roll_nom', 'ihd_yaw_nom',
            'ihd_X_sys_err', 'ihd_Y_sys_err', 'ihd_Z_sys_err', 'ihd_pitch_sys_err', 'ihd_roll_sys_err',
            'ihd_yaw_sys_err'
        ]

        for key in keys_to_average:
            # Sammle die Kurven aller Durchgänge für diesen Key
            matrix = np.array([run[key] for run in processed_runs])

            # Mittelwert über die Achse der Durchgänge (axis=0)
            mean_curves[key] = np.mean(matrix, axis=0)

            # Für die Fehlerrechnung der DoFs bestimmen wir die rein statistische Abweichung (STDEV) zwischen den Läufen
            # Wir mappen das hier direkt auf die kurzen Namen für den späteren CSV-Export
            short_key = key.replace('et_', '').replace('ihd_', '').replace('_nom', '')
            if short_key in ['lateral', 'longitudinal', 'vertical']:
                # Mapping auf X, Y, Z für den Tabellenexport
                short_key = {'lateral': 'X', 'longitudinal': 'Y', 'vertical': 'Z'}[short_key]

            if key.startswith('et_'):
                # Statistischer Fehler der ETD-Messungen untereinander
                stat_errors[f'et_{short_key}_stat'] = np.std(matrix, axis=0)
            elif '_nom' in key:
                # Statistischer Fehler der Phantom-Läufe untereinander
                stat_errors[f'ihd_{short_key}_stat'] = np.std(matrix, axis=0)

        return mean_curves, stat_errors