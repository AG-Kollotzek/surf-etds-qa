import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.patches import ConnectionPatch
import gc

from DataConverter import ETDQAProcessor

# ==========================================
# KONFIGURATION & GLOBALE VARIABLEN
# ==========================================
CONFIG_JSON_PATH = "etds_qa_2026_config.json"
ZOOM_CONFIG_FILE = "zoom_box_config.json"

# Spalten-Konfiguration für Export
ETD_COLS = ['Time_Sec', 'lateral', 'longitudinal', 'vertical', 'pitch', 'yaw', 'roll']
PHANTOM_COLS = ['Time_Sec', 'True_Lateral', 'True_Longitudinal', 'True_Vertical', 'True_Pitch', 'True_Yaw', 'True_Roll']

DOF_PAIRS = {
    'lateral': ('True_Lateral', 'lateral'),
    'longitudinal': ('True_Longitudinal', 'longitudinal'),
    'vertical': ('True_Vertical', 'vertical'),
    'pitch': ('True_Pitch', 'pitch'),
    'yaw': ('True_Yaw', 'yaw'),
    'roll': ('True_Roll', 'roll')
}


# ==========================================
# HILFSFUNKTIONEN (Export & RMSE)
# ==========================================
def load_box_config():
    if os.path.exists(ZOOM_CONFIG_FILE):
        with open(ZOOM_CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_box_config(config_data):
    with open(ZOOM_CONFIG_FILE, 'w') as f:
        json.dump(config_data, f, indent=4)


def save_7d_array(df, columns, save_path):
    if df is not None and not df.empty:
        # Nur existierende Spalten exportieren
        valid_cols = [col for col in columns if col in df.columns]
        df_clean = df[valid_cols].copy()
        df_clean.to_csv(save_path, index=False, sep=';', decimal='.')


def calculate_and_save_rmse(df_etd, df_phantom, out_dir):
    """Interpoliert ETD auf Phantom und berechnet RMSE für alle 6 DoF"""
    if df_etd.empty or df_phantom.empty:
        return None

    t_phantom = df_phantom['Time_Sec'].values
    t_etd = df_etd['Time_Sec'].values
    rmse_results = {}

    for dof_name, (phantom_col, etd_col) in DOF_PAIRS.items():
        if phantom_col not in df_phantom.columns or etd_col not in df_etd.columns:
            continue

        phantom_vals = df_phantom[phantom_col].values
        etd_vals = df_etd[etd_col].values

        etd_interpolated = np.interp(t_phantom, t_etd, etd_vals, left=etd_vals[0], right=etd_vals[-1])
        diff = phantom_vals - etd_interpolated
        rmse_results[dof_name] = np.sqrt(np.mean(diff ** 2))

    df_rmse = pd.DataFrame([rmse_results])
    output_path = out_dir / "03_alldof_rmse.csv"
    df_rmse.to_csv(output_path, index=False, sep=";", decimal=".")

    return rmse_results


def bin_and_average(data_frames, bin_ms=200):
    bin_sec = bin_ms / 1000.0
    aligned_frames = []

    for df in data_frames:
        df_align = df.copy()
        t_start = df_align['Time_Sec'].iloc[0]
        df_align['Time_Relative'] = df_align['Time_Sec'] - t_start
        aligned_frames.append(df_align)

    combined = pd.concat(aligned_frames, ignore_index=True)
    combined['Time_Bin'] = np.round(combined['Time_Relative'] / bin_sec) * bin_sec

    grouped = combined.groupby('Time_Bin')
    mean_df = grouped.mean().reset_index()
    std_df = grouped.std().reset_index().fillna(0)

    return mean_df, std_df


# ==========================================
# INTERAKTIVE PLOT-ROUTINE
# ==========================================
def plot_evaluation_results_interactive(
        t_kin, trans_et, trans_ihd, rot_et, rot_ihd,
        std_trans, std_rot,
        t_rmse, rmse3d, rmse_temp, save_path=None, group_name="Unbekannte Gruppe"
):
    C_X_PITCH = '#D55E00'
    C_Y_YAW = '#56B4E9'
    C_Z_ROLL = '#009E73'
    C_RMSE3D = '#CC79A7'
    C_RMSETMP = '#E69F00'

    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
        'font.size': 12, 'axes.edgecolor': 'black', 'axes.linewidth': 1.2,
        'legend.frameon': True, 'legend.edgecolor': 'black'
    })

    movement_mag = np.abs(trans_ihd['X']) + np.abs(trans_ihd['Y']) + np.abs(trans_ihd['Z'])
    t_peak = t_kin[np.argmax(movement_mag)]

    window_size = max(1, int(len(movement_mag) / 20))
    min_var = float('inf')
    t_flat_idx = 0
    for i in range(0, len(movement_mag) - window_size, window_size):
        var = np.var(movement_mag[i:i + window_size])
        if var < min_var:
            min_var = var
            t_flat_idx = i + window_size // 2
    t_flat = t_kin[t_flat_idx]

    window_sec = 4.0

    POS_MAP = {
        'upper-left': [0.05, 0.60, 0.25, 0.35], 'top-left': [0.05, 0.60, 0.25, 0.35],
        'upper-right': [0.60, 0.60, 0.25, 0.35], 'top-right': [0.60, 0.60, 0.25, 0.35],
        'upper-mid': [0.375, 0.60, 0.25, 0.35],
        'lower-left': [0.05, 0.05, 0.25, 0.35], 'bottom-left': [0.05, 0.05, 0.25, 0.35],
        'lower-right': [0.70, 0.05, 0.25, 0.35], 'bottom-right': [0.70, 0.05, 0.25, 0.35],
        'lower-mid': [0.375, 0.05, 0.25, 0.35]
    }

    zooms = {
        'peak': {
            'ax_idx': 0, 'xlim': [max(0, t_peak - window_sec / 2), t_peak + window_sec / 2],
            'ylim': None, 'pos': POS_MAP['upper-left'], 'active': False
        },
        'flat': {
            'ax_idx': 0, 'xlim': [max(0, t_flat - window_sec / 2), t_flat + window_sec / 2],
            'ylim': None, 'pos': POS_MAP['upper-right'], 'active': False
        }
    }

    box_config = load_box_config()
    if group_name in box_config:
        zooms = box_config[group_name]
    else:
        box_config[group_name] = zooms
        save_box_config(box_config)

    fig = None

    while True:
        if fig is not None:
            plt.close(fig)
            gc.collect()

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, dpi=100)
        fig.suptitle(f"ETDS outputs vs. SURF IHD (Group: {group_name})", fontsize=16, fontweight='bold')
        fig.tight_layout(rect=[0.02, 0.02, 1, 0.98])
        line_w = 1.2

        # Translation
        axes[0].plot(t_kin, trans_ihd['X'], label='IHD X', color=C_X_PITCH, linestyle='--', linewidth=line_w)
        axes[0].plot(t_kin, trans_ihd['Y'], label='IHD Y', color=C_Y_YAW, linestyle='--', linewidth=line_w)
        axes[0].plot(t_kin, trans_ihd['Z'], label='IHD Z', color=C_Z_ROLL, linestyle='--', linewidth=line_w)
        axes[0].plot(t_kin, trans_et['X'], label='ET X', color=C_X_PITCH, linestyle='-', linewidth=line_w)
        axes[0].plot(t_kin, trans_et['Y'], label='ET Y', color=C_Y_YAW, linestyle='-', linewidth=line_w)
        axes[0].plot(t_kin, trans_et['Z'], label='ET Z', color=C_Z_ROLL, linestyle='-', linewidth=line_w)
        axes[0].set_ylabel('Translation [mm]')
        axes[0].legend(loc='upper right', ncol=1, fontsize=9)
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[0].fill_between(t_kin, trans_et['X'] - std_trans['X'], trans_et['X'] + std_trans['X'], color=C_X_PITCH,
                             alpha=0.2, linewidth=0)
        axes[0].fill_between(t_kin, trans_et['Y'] - std_trans['Y'], trans_et['Y'] + std_trans['Y'], color=C_Y_YAW,
                             alpha=0.2, linewidth=0)
        axes[0].fill_between(t_kin, trans_et['Z'] - std_trans['Z'], trans_et['Z'] + std_trans['Z'], color=C_Z_ROLL,
                             alpha=0.2, linewidth=0)

        # Rotation
        axes[1].plot(t_kin, rot_ihd['pitch'], label='IHD pitch', color=C_X_PITCH, linestyle='--', linewidth=line_w)
        axes[1].plot(t_kin, rot_ihd['yaw'], label='IHD yaw', color=C_Y_YAW, linestyle='--', linewidth=line_w)
        axes[1].plot(t_kin, rot_ihd['roll'], label='IHD roll', color=C_Z_ROLL, linestyle='--', linewidth=line_w)
        axes[1].plot(t_kin, rot_et['pitch'], label='ET pitch', color=C_X_PITCH, linestyle='-', linewidth=line_w)
        axes[1].plot(t_kin, rot_et['yaw'], label='ET yaw', color=C_Y_YAW, linestyle='-', linewidth=line_w)
        axes[1].plot(t_kin, rot_et['roll'], label='ET roll', color=C_Z_ROLL, linestyle='-', linewidth=line_w)
        axes[1].set_ylabel('Rotation [°]')
        axes[1].legend(loc='upper right', ncol=1, fontsize=9)
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].fill_between(t_kin, rot_et['pitch'] - std_rot['pitch'], rot_et['pitch'] + std_rot['pitch'],
                             color=C_X_PITCH, alpha=0.2, linewidth=0)
        axes[1].fill_between(t_kin, rot_et['yaw'] - std_rot['yaw'], rot_et['yaw'] + std_rot['yaw'], color=C_Y_YAW,
                             alpha=0.2, linewidth=0)
        axes[1].fill_between(t_kin, rot_et['roll'] - std_rot['roll'], rot_et['roll'] + std_rot['roll'], color=C_Z_ROLL,
                             alpha=0.2, linewidth=0)

        current_ymin, current_ymax = axes[1].get_ylim()
        axes[1].set_ylim(min(current_ymin, -0.5), max(current_ymax, 0.5))

        # RMSE
        axes[2].plot(t_rmse, rmse3d, label='RMSE 3D', color=C_RMSE3D, linestyle='-', linewidth=line_w)
        axes[2].plot(t_rmse, rmse_temp, label='RMSE Temp', color=C_RMSETMP, linestyle='-', linewidth=line_w)
        axes[2].set_ylabel('RMSE')
        axes[2].set_xlabel('Time [s]')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, linestyle=':', alpha=0.6)

        # Zoom Boxes
        for z_name, z_data in zooms.items():
            if not z_data.get('active', True):
                continue
            try:
                ax_idx = int(z_data['ax_idx'])
                box_pos = [float(x) for x in z_data['pos']]
                xlims = tuple(float(x) for x in z_data['xlim'])
            except (KeyError, TypeError, ValueError):
                continue

            ax_main = axes[ax_idx]
            for artist in list(ax_main.get_children()):
                if isinstance(artist, ConnectionPatch):
                    artist.remove()

            axins = ax_main.inset_axes(box_pos)
            axins.set_facecolor((1.0, 1.0, 1.0, 0.6))

            if ax_idx == 0:
                axins.plot(t_kin, trans_ihd['X'], color=C_X_PITCH, linestyle='--')
                axins.plot(t_kin, trans_ihd['Y'], color=C_Y_YAW, linestyle='--')
                axins.plot(t_kin, trans_ihd['Z'], color=C_Z_ROLL, linestyle='--')
                axins.plot(t_kin, trans_et['X'], color=C_X_PITCH)
                axins.plot(t_kin, trans_et['Y'], color=C_Y_YAW)
                axins.plot(t_kin, trans_et['Z'], color=C_Z_ROLL)
                axins.fill_between(t_kin, trans_et['X'] - std_trans['X'], trans_et['X'] + std_trans['X'],
                                   color=C_X_PITCH, alpha=0.2)
                axins.fill_between(t_kin, trans_et['Y'] - std_trans['Y'], trans_et['Y'] + std_trans['Y'], color=C_Y_YAW,
                                   alpha=0.2)
                axins.fill_between(t_kin, trans_et['Z'] - std_trans['Z'], trans_et['Z'] + std_trans['Z'],
                                   color=C_Z_ROLL, alpha=0.2)
            elif ax_idx == 1:
                axins.plot(t_kin, rot_ihd['pitch'], color=C_X_PITCH, linestyle='--')
                axins.plot(t_kin, rot_ihd['yaw'], color=C_Y_YAW, linestyle='--')
                axins.plot(t_kin, rot_ihd['roll'], color=C_Z_ROLL, linestyle='--')
                axins.plot(t_kin, rot_et['pitch'], color=C_X_PITCH)
                axins.plot(t_kin, rot_et['yaw'], color=C_Y_YAW)
                axins.plot(t_kin, rot_et['roll'], color=C_Z_ROLL)
                axins.fill_between(t_kin, rot_et['pitch'] - std_rot['pitch'], rot_et['pitch'] + std_rot['pitch'],
                                   color=C_X_PITCH, alpha=0.2)
                axins.fill_between(t_kin, rot_et['yaw'] - std_rot['yaw'], rot_et['yaw'] + std_rot['yaw'], color=C_Y_YAW,
                                   alpha=0.2)
                axins.fill_between(t_kin, rot_et['roll'] - std_rot['roll'], rot_et['roll'] + std_rot['roll'],
                                   color=C_Z_ROLL, alpha=0.2)

            axins.set_xlim(xlims)
            has_manual_ylim = False
            if z_data.get('ylim') is not None:
                ylims = z_data['ylim']
                if len(ylims) == 2 and ylims[0] is not None and ylims[1] is not None:
                    try:
                        axins.set_ylim(tuple(float(y) for y in ylims))
                        has_manual_ylim = True
                    except (ValueError, TypeError):
                        pass

            if not has_manual_ylim:
                mask = (t_kin >= xlims[0]) & (t_kin <= xlims[1])
                if mask.any():
                    if ax_idx == 0:
                        y_vals = np.concatenate([trans_ihd['X'][mask], trans_ihd['Y'][mask], trans_ihd['Z'][mask],
                                                 trans_et['X'][mask], trans_et['Y'][mask], trans_et['Z'][mask]])
                    else:
                        y_vals = np.concatenate([rot_ihd['pitch'][mask], rot_ihd['yaw'][mask], rot_ihd['roll'][mask],
                                                 rot_et['pitch'][mask], rot_et['yaw'][mask], rot_et['roll'][mask]])
                    ymin, ymax = y_vals.min(), y_vals.max()
                    margin = max(0.1, (ymax - ymin) * 0.15)
                    axins.set_ylim(ymin - margin, ymax + margin)

            axins.set_xticklabels([])

            try:
                inset_result = mark_inset(ax_main, axins, loc1=1, loc2=2, fc="none", ec="black", lw=1)
                if isinstance(inset_result, tuple) and len(inset_result) >= 1:
                    connects = inset_result[1:]
                    for c in connects:
                        c.set_visible(False)
            except Exception:
                from matplotlib.patches import Rectangle
                xlim_main, ylim_main = axins.get_xlim(), axins.get_ylim()
                ax_main.add_patch(
                    Rectangle((xlim_main[0], ylim_main[0]), xlim_main[1] - xlim_main[0], ylim_main[1] - ylim_main[0],
                              facecolor='none', edgecolor='black', linewidth=1))

            # Dynamic connectors
            x_main_min, x_main_max = axins.get_xlim()
            y_main_min, y_main_max = axins.get_ylim()
            x_main_center = (x_main_min + x_main_max) / 2.0
            x_box_center_fraction = z_data['pos'][0] + (z_data['pos'][2] / 2.0)

            trans_fraction_to_data = ax_main.transAxes + ax_main.transData.inverted()
            x_box_center_data = trans_fraction_to_data.transform((x_box_center_fraction, 0.5))[0]

            if x_box_center_data > x_main_center:
                cp_top = ConnectionPatch(xyA=(x_main_max, y_main_max), xyB=(0.0, 1.0), coordsA="data",
                                         coordsB="axes fraction", axesA=ax_main, axesB=axins, color="black",
                                         linewidth=0.8, alpha=0.6)
                cp_bottom = ConnectionPatch(xyA=(x_main_max, y_main_min), xyB=(0.0, 0.0), coordsA="data",
                                            coordsB="axes fraction", axesA=ax_main, axesB=axins, color="black",
                                            linewidth=0.8, alpha=0.6)
            else:
                cp_top = ConnectionPatch(xyA=(x_main_min, y_main_max), xyB=(1.0, 1.0), coordsA="data",
                                         coordsB="axes fraction", axesA=ax_main, axesB=axins, color="black",
                                         linewidth=0.8, alpha=0.6)
                cp_bottom = ConnectionPatch(xyA=(x_main_min, y_main_min), xyB=(1.0, 0.0), coordsA="data",
                                            coordsB="axes fraction", axesA=ax_main, axesB=axins, color="black",
                                            linewidth=0.8, alpha=0.6)

            ax_main.add_artist(cp_top)
            ax_main.add_artist(cp_bottom)

        plt.show(block=False)
        plt.pause(0.1)

        print(f"\n--- Editor: {group_name} ---")
        print(" [save]   Save and proceed | [Enter] Skip | [exit] Abort")
        print(" [box off]  Entfernt Boxen | 'peak 1 15.0 20.0 lower-right' etc.")

        cmd = input("Command: ").strip().lower()

        if cmd == "exit":
            plt.close(fig)
            return 'exit'
        if cmd == "":
            box_config[group_name] = zooms
            save_box_config(box_config)
            plt.close(fig)
            return 'skipped'
        elif cmd == "save":
            box_config[group_name] = zooms
            save_box_config(box_config)
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
                print(f"   [✓] Saved PDF to {save_path}")
            plt.close(fig)
            return 'saved'
        elif cmd == "box off":
            zooms.clear()
            continue
        else:
            try:
                parts = cmd.split()
                box_name = parts[0]
                if box_name in zooms:
                    if len(parts) >= 2 and parts[1] == 'off':
                        zooms[box_name]['active'] = False
                        continue
                    elif len(parts) >= 2 and parts[1] == 'on':
                        zooms[box_name]['active'] = True
                        continue
                if len(parts) >= 4:
                    box_name, ax_idx = parts[0], int(parts[1])
                    tmin, tmax = float(parts[2]), float(parts[3])
                    ymin, ymax, new_pos = None, None, None
                    rem_parts = parts[4:]
                    if rem_parts and rem_parts[-1] in POS_MAP:
                        new_pos = POS_MAP[rem_parts.pop()]
                    if len(rem_parts) == 2:
                        ymin, ymax = float(rem_parts[0]), float(rem_parts[1])

                    if box_name in ['peak', 'flat']:
                        if box_name not in zooms:
                            zooms[box_name] = {
                                'pos': POS_MAP['upper-left'] if box_name == 'peak' else POS_MAP['upper-right']}
                        zooms[box_name]['ax_idx'] = ax_idx
                        zooms[box_name]['xlim'] = [tmin, tmax]
                        zooms[box_name]['ylim'] = [ymin, ymax] if ymin is not None else None
                        if new_pos is not None:
                            zooms[box_name]['pos'] = new_pos
            except Exception as e:
                print(f"Error parsing input: {e}")


# ==========================================
# HAUPTSKRIPT & BATCH-LOGIK
# ==========================================
def main():
    if len(sys.argv) < 4:
        print("Nutzung: python etds_qa_evaluation.py <mode: plot/rmse> <linac_id> <pads: RT/32>")
        print("Beispiel: python etds_qa_evaluation.py plot 1 32")
        return

    mode = sys.argv[1].lower()
    target_linac = sys.argv[2]

    # Pads übersetzen (RT in der Kommandozeile entspricht OFF im JSON)
    target_pads_input = sys.argv[3].upper()
    target_pads = "OFF" if target_pads_input == "RT" else target_pads_input

    # Pfade laden
    with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
        full_config = json.load(f)

    # 1. Filtern und Gruppieren
    # Gruppiert nach: (Group_Name, Pads)
    groups = {}
    for m_id, meta in full_config.items():
        if meta.get("Linac") != target_linac: continue

        # NEU: Wenn "ALL" übergeben wurde, ignorieren wir den Pad-Filter
        if target_pads_input != "ALL" and meta.get("heatingpads") != target_pads:
            continue

        g_key = (meta.get("group"), meta.get("heatingpads"))
        if g_key not in groups:
            groups[g_key] = []

        # Speichere die Messungs-ID und Metadaten für die Verarbeitung
        groups[g_key].append((m_id, meta))

    if not groups:
        print(f"Keine passenden Messungen für Linac {target_linac} und Pads {target_pads_input} gefunden.")
        return

    # Basis-Verzeichnisse für In- und Output
    raw_base_dir = Path(f"data/raw/L{target_linac}")
    process_base_dir = Path(f"data/process/L{target_linac}")

    for (group_name, pad_status), messungen in groups.items():
        pad_folder_name = "32" if pad_status == "32" else "RT"
        print(f"\n[{group_name.upper()} | Pads: {pad_folder_name}] ---> Starte Export & Alignment")

        all_json_dfs = []
        reference_csv_df = None

        for m_id, meta in messungen:
            # Ordnerstruktur aufbauen (z.B. data/process/L1/32/mindev/meas_1)
            out_dir = process_base_dir / pad_folder_name / group_name / f"meas_{str(m_id).zfill(2)}"
            out_dir.mkdir(parents=True, exist_ok=True)

            with open(out_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=4)

            etd_stamp = meta["etds_timestamp"]
            surf_stamp = meta["surf_timestamp"]

            # Formatieren des ETD-Timestamps falls nötig (z.B. 161959 -> 16-19-59)
            if len(etd_stamp) == 6 and "-" not in etd_stamp:
                etd_stamp = f"{etd_stamp[:2]}-{etd_stamp[2:4]}-{etd_stamp[4:]}"

            # Suche die passenden Rohdateien in den Subordnern
            csv_files = list((raw_base_dir / "surf_phantom").rglob(f"*{surf_stamp}.csv"))
            json_files = list((raw_base_dir / "etds_scans").rglob(f"*{etd_stamp}.json"))

            if not csv_files or not json_files:
                print(f"   [!] FEHLT: Rohdaten für ID {m_id}. Überspringe.")
                continue

            try:
                proc = ETDQAProcessor(terminal_version='legacy')
                proc.load_csv(str(csv_files[0]))
                proc.load_json(str(json_files[0]))

                # 1. Signale abgleichen (Zeitskalierung & Peak-Ermittlung auf Pos_H)
                proc.align_and_crop_signals(measurement_group=group_name)

                # Export 01 (Rohdaten nach Alignment / Vor Kinematik)
                save_7d_array(proc.df_json, ETD_COLS, out_dir / "01_after_align_etd.csv")
                save_7d_array(proc.df_csv, PHANTOM_COLS, out_dir / "01_after_align_phantom.csv")

                # 2. Kinematische Transformation in klinische Koordinaten berechnen (erzeugt True_* Spalten)
                c_angle = float(meta.get("couch_angle", 0.0))
                proc.apply_kinematics(couch_angle=c_angle)

                # Export 02 (Aligned + Kinematik angewendet)
                save_7d_array(proc.df_json, ETD_COLS, out_dir / "02_aligned_kin_applied_etd.csv")
                save_7d_array(proc.df_csv, PHANTOM_COLS, out_dir / "02_aligned_kin_applied_phantom.csv")

                # Export 03 (RMSE)
                calculate_and_save_rmse(proc.df_json, proc.df_csv, out_dir)

                # Daten für Plotting im Arbeitsspeicher ablegen
                all_json_dfs.append(proc.df_json)
                if reference_csv_df is None:
                    reference_csv_df = proc.df_csv

                print(f"   ✅ ID {m_id} verarbeitet und in {out_dir} gespeichert.")
            except Exception as e:
                print(f"   [X] FEHLER bei ID {m_id}: {e}")

        # Plot Generierung (Wenn --mode plot)
        if mode == "plot" and all_json_dfs:
            print(f"\nGeneriere Plot für {group_name}...")
            mean_df, std_df = bin_and_average(all_json_dfs)
            t_kin = mean_df['Time_Bin'].values

            def get_interp(col):
                csv_time = reference_csv_df['Time_Sec'].values - reference_csv_df['Time_Sec'].iloc[0]
                arr = reference_csv_df[
                    f'{col}_nominal'].values if f'{col}_nominal' in reference_csv_df.columns else np.array(
                    [getattr(v, 'n', v) for v in reference_csv_df[col]])
                return np.interp(t_kin, csv_time, arr)

            trans_et = {'X': mean_df['lateral'].values, 'Y': mean_df['longitudinal'].values,
                        'Z': -mean_df['vertical'].values}
            trans_ihd = {'X': get_interp('True_Lateral'), 'Y': get_interp('True_Longitudinal'),
                         'Z': get_interp('True_Vertical')}
            rot_et = {'pitch': mean_df['pitch'].values, 'yaw': mean_df['yaw'].values, 'roll': mean_df['roll'].values}
            rot_ihd = {'pitch': get_interp('True_Pitch'), 'yaw': get_interp('True_Yaw'),
                       'roll': get_interp('True_Roll')}

            std_trans = {'X': std_df['lateral'].values, 'Y': std_df['longitudinal'].values,
                         'Z': std_df['vertical'].values}
            std_rot = {'pitch': std_df['pitch'].values, 'yaw': std_df['yaw'].values, 'roll': std_df['roll'].values}

            rmse3d_vals = mean_df['rmse3d'].values if 'rmse3d' in mean_df.columns else np.zeros_like(t_kin)
            rmse_temp_vals = mean_df['rmse_temp'].values if 'rmse_temp' in mean_df.columns else np.zeros_like(t_kin)

            plot_out_dir = process_base_dir / pad_folder_name / group_name
            date_str = datetime.now().strftime("%Y-%m-%d")
            save_path = plot_out_dir / f"{group_name}_Plot_{date_str}.pdf"

            res = plot_evaluation_results_interactive(
                t_kin=t_kin, trans_et=trans_et, trans_ihd=trans_ihd, rot_et=rot_et, rot_ihd=rot_ihd,
                std_trans=std_trans, std_rot=std_rot, t_rmse=t_kin, rmse3d=rmse3d_vals, rmse_temp=rmse_temp_vals,
                save_path=save_path, group_name=group_name
            )

            if res == 'exit':
                print("Abbruch durch Benutzer.")
                break


if __name__ == "__main__":
    main()