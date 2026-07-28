import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from kinematics import SurfKinematics as SurfKinematicsNominal
from kinematics_werror_v2 import SurfKinematics
from uncertainties import unumpy as unp
import datetime
import os


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
        # Wird von align_and_crop_signals gefüllt (Sync-Puls-Zeiten, definieren das Messfenster)
        self.sync_info = None

        self.kinematics_werror_dict = {}

    def load_csv(self, file_path):
        """Lädt das SURF-Terminal Log und korrigiert ggf. Vorzeichenfehler"""
        self.csv_filename = os.path.basename(file_path)  # NEU: Dateiname merken
        self.df_csv = pd.read_csv(file_path, sep=';', decimal='.')

        # --- SMART SIGN CORRECTION ---t
        if self.terminal_version == 'legacy':
            # Korrektur für Longitudinal (H) und Rotation (R)
            #self.df_csv['Pos_H'] = self.df_csv['Pos_H'] * -1
            self.df_csv['Pos_R'] = self.df_csv['Pos_R'] * -1
            print(f"INFO: Legacy-Mode aktiv. Vorzeichen für Pos_R wurden invertiert.")

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

    def align_and_crop_signals(self, measurement_group='mindev', sync_axis='Pos_H', threshold=3.0, pad_sec=5.0,
                               etds_timestamp=None, surf_timestamp=None):
        """
        1. Findet Peaks in CSV und JSON.
        2. Berechnet den Skalierungsfaktor und Offset (Alignment der Zeitachsen).
        3. Skaliert die JSON-Zeitachse passend auf die CSV-Zeitachse.
        4. Schneidet (croppt) beide DataFrames exakt auf [Peak_1 - pad_sec ... Peak_2 + pad_sec] zu.
        (Keine Baseline-Nullung!)

        etds_timestamp/surf_timestamp (Format HHMMSS, aus der Config): werden benötigt, wenn
        sich mehrere ETD-Scans dieselbe CSV teilen (z.B. die Couch-Rotations-Serie) - darüber
        wird das richtige Sync-Puls-Paar in der CSV gefunden (siehe Schritt 3).
        """

        def get_peaks(times, level, min_dur=0.5, max_dur=5.0, vel_tol=0.6,
                      amp_min=2.5, amp_max=8.0, level_tol=1.5):
            """
            Findet die Sync-Pulse (kurzer 5-mm-Hub, der auf das Ausgangsniveau zurückkehrt).

            Statt "Ausschlag gegen ein gleitendes Median-Baseline" (das bei 2-s-Pulsen
            zusammenbricht) wird hier über den Stillstand der Achsen segmentiert:
            Ein Sync-Puls ist ein kurzes Plateau, dessen Nachbar-Plateaus dasselbe
            Niveau haben. Damit sind Sync-Pulse eindeutig von den Fahrten in die
            Messposition unterscheidbar - die enden auf einem anderen Niveau.

            times: 1D-Zeitachse, level: (N, k)-Array der Achsen/DoF.
            """
            t = np.asarray(times, dtype=float)
            L = np.atleast_2d(np.asarray(level, dtype=float))
            if L.shape[0] != len(t):
                L = L.T

            dt = np.gradient(t)
            dt[dt == 0] = np.nan
            speed = np.sum(np.abs(np.gradient(L, axis=0)), axis=1) / dt
            speed = pd.Series(speed).rolling(5, center=True, min_periods=1).median().bfill().ffill().values

            still = speed < vel_tol
            edges = np.diff(still.astype(int), prepend=0, append=0)
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0] - 1

            plateaus = [[s, e, np.median(L[s:e + 1], axis=0)]
                        for s, e in zip(starts, ends) if t[e] - t[s] > 0.2]

            # ETD-Rauschen zerhackt lange Plateaus -> gleiche Niveaus wieder verschmelzen
            merged = []
            for p in plateaus:
                if merged and np.linalg.norm(p[2] - merged[-1][2]) < level_tol:
                    merged[-1][1] = p[1]
                    merged[-1][2] = np.median(L[merged[-1][0]:p[1] + 1], axis=0)
                else:
                    merged.append(p)

            valid_peaks = []
            for i in range(1, len(merged) - 1):
                s, e, lvl = merged[i]
                duration = t[e] - t[s]
                if not (min_dur <= duration <= max_dur):
                    continue

                prev_lvl, next_lvl = merged[i - 1][2], merged[i + 1][2]
                if np.linalg.norm(prev_lvl - next_lvl) > level_tol:
                    continue  # kein Rücksprung -> Fahrt in die Messposition, kein Sync-Puls

                a_in = np.linalg.norm(lvl - prev_lvl)
                a_out = np.linalg.norm(lvl - next_lvl)
                if not (amp_min <= min(a_in, a_out) and max(a_in, a_out) <= amp_max):
                    continue

                valid_peaks.append({'mid': (t[s] + t[e]) / 2.0, 'start': t[s], 'end': t[e]})
            return valid_peaks

        # --- 1. Sync-Pulse in CSV finden (Hardware-Achsen) ---
        csv_peaks = get_peaks(self.df_csv['Time_Sec'].values,
                              self.df_csv[['Pos_H', 'Pos_V', 'Pos_R']].values)

        # --- 2. Sync-Pulse in JSON finden (ETD-Translationen) ---
        json_peaks = get_peaks(self.df_json['Time_Sec'].values,
                               self.df_json[['lateral', 'longitudinal', 'vertical']].values)

        # --- 3. Welches Paar aus der CSV gehört zu dieser Messung? ---
        # Mehrere ETD-Scans können sich dieselbe CSV teilen (z.B. die Couch-Rotations-Serie,
        # ein CSV-Lauf mit 4 Sync-Puls-Paaren für 4 ETD-Scans). Das richtige Paar wird über
        # den bekannten Wanduhr-Versatz zwischen SURF- und ETD-Start gefunden: das Paar, dessen
        # Zeitstempel in der CSV am nächsten an (etds_start - surf_start) liegt. Das ist robust
        # gegen Gruppennamen (anders als eine Heuristik auf Basis von measurement_group).
        pair_idx = 0
        n_pairs = len(csv_peaks) // 2

        if n_pairs > 1:
            if etds_timestamp is None or surf_timestamp is None:
                raise ValueError(
                    f"CSV enthält {n_pairs} Sync-Puls-Paare - ohne etds_timestamp/surf_timestamp "
                    f"kann das richtige Paar nicht eindeutig bestimmt werden.")

            def _hhmmss_to_sec(stamp):
                s = str(stamp)
                return int(s[0:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6])

            offset_sec = _hhmmss_to_sec(etds_timestamp) - _hhmmss_to_sec(surf_timestamp)
            pair_mids = [(csv_peaks[2 * i]['mid'] + csv_peaks[2 * i + 1]['mid']) / 2.0 for i in range(n_pairs)]
            pair_idx = int(np.argmin([abs(mid - offset_sec) for mid in pair_mids]))

        if len(csv_peaks) < (pair_idx * 2 + 2):
            raise ValueError(
                f"Nicht genügend Peaks in CSV! (Suche Paar {pair_idx + 1}, aber nur {len(csv_peaks)} gefunden)")
        if len(json_peaks) < 2:
            raise ValueError(f"Nicht genügend Peaks in JSON! (Gefunden: {len(json_peaks)}, Erwartet: mind. 2)")

        csv_first = csv_peaks[pair_idx * 2]
        csv_last = csv_peaks[pair_idx * 2 + 1]
        json_first = json_peaks[0]
        json_last = json_peaks[-1]

        # --- 4. Zeitskalierung & Alignment (Dein bewährtes Verfahren) ---
        scale_factor = (csv_last['mid'] - csv_first['mid']) / (json_last['mid'] - json_first['mid'])
        self.df_json['Time_Sec'] = (self.df_json['Time_Sec'] - json_first['mid']) * scale_factor + csv_first['mid']

        # --- 4b. Sync-Zeitstempel festhalten ---
        # Nach dem Alignment liegen die Sync-Pulse beider Systeme per Konstruktion exakt
        # aufeinander, d.h. die gemeinsame (CSV-)Zeitbasis der Sync-Mitten ist csv_first/csv_last.
        # Trotzdem werden die ETD-Rohzeiten mitgespeichert: damit lässt sich später ohne erneutes
        # Alignment in die unveränderte TrackingResult-JSON zurückspringen.
        # Diese Zeitstempel definieren das Messfenster (siehe qa_metrics.measurement_window).
        self.sync_info = {
            'aligned_first_mid': float(csv_first['mid']),
            'aligned_last_mid': float(csv_last['mid']),
            'phantom_first_mid': float(csv_first['mid']),
            'phantom_last_mid': float(csv_last['mid']),
            'etd_raw_first_mid': float(json_first['mid']),
            'etd_raw_last_mid': float(json_last['mid']),
            'scale_factor': float(scale_factor),
            'csv_pair_index': int(pair_idx),
        }

        # Tracking Lost Zeiten mitskalieren
        if hasattr(self, 'lost_times_sec') and self.lost_times_sec:
            self.lost_times_aligned = [(t - json_first['mid']) * scale_factor + csv_first['mid'] for t in
                                       self.lost_times_sec]
        else:
            self.lost_times_aligned = []

        # --- 5. Cropping: Exakt 5s vor dem ersten und 5s nach dem letzten Peak ---
        crop_start = csv_first['mid'] - pad_sec
        crop_end = csv_last['mid'] + pad_sec

        self.df_csv = self.df_csv[
            (self.df_csv['Time_Sec'] >= crop_start) & (self.df_csv['Time_Sec'] <= crop_end)].reset_index(drop=True)
        self.df_json = self.df_json[
            (self.df_json['Time_Sec'] >= crop_start) & (self.df_json['Time_Sec'] <= crop_end)].reset_index(drop=True)

        if hasattr(self, 'lost_times_aligned'):
            self.lost_times_aligned = [t for t in self.lost_times_aligned if crop_start <= t <= crop_end]

        print(
            f"-> Alignment & Cropping erfolgreich (Scale: {scale_factor:.6f} | Spanne: {crop_start:.1f}s bis {crop_end:.1f}s). Keine Baseline-Korrektur.")
        return scale_factor


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