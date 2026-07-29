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
import qa_metrics
from qa_metrics import DOF_SPEC, DOF_BY_NAME
import sphere_detection
from qa_plots import plot_qa_overview

# ==========================================
# KONFIGURATION & GLOBALE VARIABLEN
# ==========================================
CONFIG_JSON_PATH = "etds_qa_2026_config.json"
ZOOM_CONFIG_FILE = "zoom_box_config.json"

# Spalten-Konfiguration für Export
# rmse3d/rmse_temp werden mitexportiert, damit die 02-CSV ohne die Roh-JSON auswertbar ist.
ETD_COLS = ['Time_Sec', 'lateral', 'longitudinal', 'vertical', 'pitch', 'yaw', 'roll',
            'rmse3d', 'rmse_temp']
PHANTOM_COLS = ['Time_Sec', 'True_Lateral', 'True_Longitudinal', 'True_Vertical', 'True_Pitch', 'True_Yaw', 'True_Roll']

# Modi der Kommandozeile. 'plot'/'deviation' bleiben als Aliase der frueheren Namen erhalten.
MODE_ALIASES = {'plot': 'plotpaper', 'deviation': 'process'}
VALID_MODES = ('process', 'plotpaper', 'plotqa', 'numqa')


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
    """RMSE über den gesamten exportierten Bereich (Übersichtswert, kein Messfenster).

    Nutzt DOF_SPEC, damit die Vorzeichen-Konvention (insbesondere die invertierte
    ETD-Vertikalachse) identisch zu Plots und numqa-Auswertung ist. Die eigentliche
    QA-Auswertung im Messfenster passiert in qa_metrics.compute_dof_metrics.
    """
    if df_etd.empty or df_phantom.empty:
        return None

    t_phantom = df_phantom['Time_Sec'].values
    t_etd = df_etd['Time_Sec'].values
    rmse_results = {}

    for dof in DOF_SPEC:
        if dof.phantom_col not in df_phantom.columns or dof.etd_col not in df_etd.columns:
            continue

        phantom_vals = df_phantom[dof.phantom_col].values
        etd_vals = dof.etd_values(df_etd)

        etd_interpolated = np.interp(t_phantom, t_etd, etd_vals, left=etd_vals[0], right=etd_vals[-1])
        diff = etd_interpolated - phantom_vals
        rmse_results[dof.name] = np.sqrt(np.mean(diff ** 2))

    df_rmse = pd.DataFrame([rmse_results])
    output_path = out_dir / "03_alldof_rmse.csv"
    df_rmse.to_csv(output_path, index=False, sep=";", decimal=".")

    return rmse_results


# ==========================================
# INTERAKTIVE PLOT-ROUTINE
# ==========================================
def plot_evaluation_results_interactive(
        t_ihd, t_et, trans_et, trans_ihd, rot_et, rot_ihd,
        t_rmse, rmse3d, rmse_temp, save_path=None, group_name="Unbekannte Gruppe"
):
    """Publikationsplot einer einzelnen Messung (ein ETD-Scan, keine Mittelung mehrerer

    Läufe mehr) - daher getrennte Zeitachsen für Phantom (t_ihd) und ETD (t_et) statt eines
    gemeinsamen Binning-Rasters, und keine Unsicherheits-/Fehlerbänder (die stellten früher
    die Streuung zwischen mehreren gemittelten ETD-Läufen dar, die es hier nicht mehr gibt).
    """
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
    t_peak = t_ihd[np.argmax(movement_mag)]

    window_size = max(1, int(len(movement_mag) / 20))
    min_var = float('inf')
    t_flat_idx = 0
    for i in range(0, len(movement_mag) - window_size, window_size):
        var = np.var(movement_mag[i:i + window_size])
        if var < min_var:
            min_var = var
            t_flat_idx = i + window_size // 2
    t_flat = t_ihd[t_flat_idx]

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
        axes[0].plot(t_ihd, trans_ihd['X'], label='IHD X', color=C_X_PITCH, linestyle='--', linewidth=line_w)
        axes[0].plot(t_ihd, trans_ihd['Y'], label='IHD Y', color=C_Y_YAW, linestyle='--', linewidth=line_w)
        axes[0].plot(t_ihd, trans_ihd['Z'], label='IHD Z', color=C_Z_ROLL, linestyle='--', linewidth=line_w)
        axes[0].plot(t_et, trans_et['X'], label='ET X', color=C_X_PITCH, linestyle='-', linewidth=line_w)
        axes[0].plot(t_et, trans_et['Y'], label='ET Y', color=C_Y_YAW, linestyle='-', linewidth=line_w)
        axes[0].plot(t_et, trans_et['Z'], label='ET Z', color=C_Z_ROLL, linestyle='-', linewidth=line_w)
        axes[0].set_ylabel('Translation [mm]')
        axes[0].legend(loc='upper right', ncol=1, fontsize=9)
        axes[0].grid(True, linestyle=':', alpha=0.6)

        # Rotation
        axes[1].plot(t_ihd, rot_ihd['pitch'], label='IHD pitch', color=C_X_PITCH, linestyle='--', linewidth=line_w)
        axes[1].plot(t_ihd, rot_ihd['yaw'], label='IHD yaw', color=C_Y_YAW, linestyle='--', linewidth=line_w)
        axes[1].plot(t_ihd, rot_ihd['roll'], label='IHD roll', color=C_Z_ROLL, linestyle='--', linewidth=line_w)
        axes[1].plot(t_et, rot_et['pitch'], label='ET pitch', color=C_X_PITCH, linestyle='-', linewidth=line_w)
        axes[1].plot(t_et, rot_et['yaw'], label='ET yaw', color=C_Y_YAW, linestyle='-', linewidth=line_w)
        axes[1].plot(t_et, rot_et['roll'], label='ET roll', color=C_Z_ROLL, linestyle='-', linewidth=line_w)
        axes[1].set_ylabel('Rotation [°]')
        axes[1].legend(loc='upper right', ncol=1, fontsize=9)
        axes[1].grid(True, linestyle=':', alpha=0.6)

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
                axins.plot(t_ihd, trans_ihd['X'], color=C_X_PITCH, linestyle='--')
                axins.plot(t_ihd, trans_ihd['Y'], color=C_Y_YAW, linestyle='--')
                axins.plot(t_ihd, trans_ihd['Z'], color=C_Z_ROLL, linestyle='--')
                axins.plot(t_et, trans_et['X'], color=C_X_PITCH)
                axins.plot(t_et, trans_et['Y'], color=C_Y_YAW)
                axins.plot(t_et, trans_et['Z'], color=C_Z_ROLL)
            elif ax_idx == 1:
                axins.plot(t_ihd, rot_ihd['pitch'], color=C_X_PITCH, linestyle='--')
                axins.plot(t_ihd, rot_ihd['yaw'], color=C_Y_YAW, linestyle='--')
                axins.plot(t_ihd, rot_ihd['roll'], color=C_Z_ROLL, linestyle='--')
                axins.plot(t_et, rot_et['pitch'], color=C_X_PITCH)
                axins.plot(t_et, rot_et['yaw'], color=C_Y_YAW)
                axins.plot(t_et, rot_et['roll'], color=C_Z_ROLL)

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
                mask_ihd = (t_ihd >= xlims[0]) & (t_ihd <= xlims[1])
                mask_et = (t_et >= xlims[0]) & (t_et <= xlims[1])
                if mask_ihd.any() or mask_et.any():
                    if ax_idx == 0:
                        y_vals = np.concatenate([trans_ihd['X'][mask_ihd], trans_ihd['Y'][mask_ihd], trans_ihd['Z'][mask_ihd],
                                                 trans_et['X'][mask_et], trans_et['Y'][mask_et], trans_et['Z'][mask_et]])
                    else:
                        y_vals = np.concatenate([rot_ihd['pitch'][mask_ihd], rot_ihd['yaw'][mask_ihd], rot_ihd['roll'][mask_ihd],
                                                 rot_et['pitch'][mask_et], rot_et['yaw'][mask_et], rot_et['roll'][mask_et]])
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
def meas_couch_type_folder(meta):
    """Einzige verbleibende Ordnerebene unter L<n>/: 'single_angle' oder 'multi_angle'."""
    return str(meta.get("meas_couch_type", "unknown")).replace(" ", "_")


def group_label(meta, linac, pad_folder_name):
    """Beschreibender Name einer physischen Messreihe (Deflection/Couch/Pads), ohne Datum.
    Stabiler Key für die Zoom-Box-Konfiguration und Basis für Ordner-/Plotnamen - bleibt über
    mehrere Verarbeitungsläufe (verschiedene Tage) hinweg für dieselbe Messreihe gleich."""
    deflection = meta.get("deflection", "?")
    couch_angle = int(meta.get("couch_angle", 0))
    return f"L{linac}_{pad_folder_name}_Deflection{deflection}_Couch{couch_angle}"


def measurement_folder_name(meta, linac, pad_folder_name, date_str):
    """Voller Ordnername für einen einzelnen Verarbeitungslauf, ersetzt die frühere
    group-Ordner/meas_XX-Verschachtelung (z.B. 'L1_RT_Deflection1_Couch-90_2026-07-28')."""
    return f"{group_label(meta, linac, pad_folder_name)}_{date_str}"


def _parse_optional_filter(argv, idx):
    """Liest sys.argv[idx], falls vorhanden. Leerer String oder 'all' (auch fehlendes
    Argument) bedeutet: kein Filter, also alle Werte zulassen."""
    if len(argv) <= idx:
        return None
    val = argv[idx].strip()
    if val == "" or val.lower() == "all":
        return None
    return val


def select_measurements(full_config, target_linac, deflection_filter=None, pads_filter=None):
    """Alle Config-Einträge eines Linacs, gefiltert nach Deflection und Heatingpads.

    pads_filter kommt in Kommandozeilen-Schreibweise ('RT'/'32'); 'RT' entspricht 'OFF'
    in der Config. None bedeutet jeweils 'kein Filter'.
    """
    pads_target = None
    if pads_filter is not None:
        pads_target = "OFF" if pads_filter.upper() == "RT" else pads_filter.upper()

    selected = []
    for m_id, meta in full_config.items():
        if meta.get("Linac") != target_linac:
            continue
        if deflection_filter is not None and str(meta.get("deflection", "")) != deflection_filter:
            continue
        if pads_target is not None and str(meta.get("heatingpads", "")).upper() != pads_target:
            continue
        selected.append((m_id, meta))

    return selected


def compute_pair_index(full_config, m_id):
    """Position dieses Eintrags unter allen Einträgen mit derselben surf_timestamp (CSV-
    Datei), sortiert nach numerischer Config-ID.

    Mehrere ETD-Scans teilen sich oft dieselbe CSV-Aufnahme (z.B. Deflection 1+2 in einem
    Durchlauf, oder bei der Couch-Serie sogar 4 Scans) - die CSV enthält dann entsprechend
    viele Sync-Puls-Paare. Die Config-IDs geben die tatsächliche Fahrreihenfolge vor (per
    Absprache), das ist robuster als ein Wall-Clock-Abgleich von etds_timestamp/surf_timestamp:
    letzterer bricht, sobald ein Sync-Puls in CSV oder JSON falsch/fehlend erkannt wird (z.B.
    weil eine schiefstehende Achse den Rücksprung-Puls auf mehrere ETD-Kanäle verteilt und er
    dadurch nicht als eigenständiges, sauberes Ereignis auftaucht) - dann verschieben sich alle
    nachfolgenden Zeit-Offsets und die Zuordnung wird falsch.
    """
    surf_stamp = full_config[m_id]["surf_timestamp"]
    siblings = sorted(
        (mid for mid, m in full_config.items() if m.get("surf_timestamp") == surf_stamp),
        key=lambda mid: int(mid))
    return siblings.index(m_id)


def process_measurement(m_id, meta, target_linac, raw_base_dir, process_base_dir, date_str, full_config):
    """Verarbeitet eine einzelne Messung: Alignment, Kinematik, Export 01/02/03.

    Rückgabe: record-dict mit den ausgerichteten DataFrames und dem Sync-Zeitstempel-Block,
    oder None wenn die Rohdaten fehlen. Fehler beim Verarbeiten werden nach oben gereicht.
    """
    pad_folder_name = "32" if meta.get("heatingpads") == "32" else "RT"
    g_label = group_label(meta, target_linac, pad_folder_name)
    couch_folder = meas_couch_type_folder(meta)

    etd_stamp = meta["etds_timestamp"]
    surf_stamp = meta["surf_timestamp"]

    # Formatieren des ETD-Timestamps falls nötig (z.B. 161959 -> 16-19-59)
    if len(etd_stamp) == 6 and "-" not in etd_stamp:
        etd_stamp = f"{etd_stamp[:2]}-{etd_stamp[2:4]}-{etd_stamp[4:]}"

    csv_files = list((raw_base_dir / "surf_phantom").rglob(f"*{surf_stamp}.csv"))
    json_files = list((raw_base_dir / "etds_scans").rglob(f"*{etd_stamp}.json"))
    # Die Sphere-Detection-Datei heißt wie die Messdatei mit '_QA' und darf nicht als Messdatei
    # eingesammelt werden.
    csv_files = [c for c in csv_files if not c.name.endswith("_QA.csv")]

    if not csv_files or not json_files:
        print(f"   [!] FEHLT: Rohdaten für ID {m_id}. Überspringe.")
        return None

    out_dir = process_base_dir / couch_folder / measurement_folder_name(meta, target_linac, pad_folder_name, date_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = ETDQAProcessor(terminal_version='legacy')
    proc.load_csv(str(csv_files[0]))
    proc.load_json(str(json_files[0]))

    # 1. Signale abgleichen (Zeitskalierung anhand der Sync-Pulse + Feinjustierung)
    proc.align_and_crop_signals(measurement_group=g_label,
                                pair_idx=compute_pair_index(full_config, m_id),
                                couch_angle=float(meta.get("couch_angle", 0.0)))

    # Sync-Zeitstempel in die Metadaten übernehmen - sie definieren das Messfenster und
    # werden von numqa und der Sphere-Detection-Auswertung wiederverwendet.
    meta_out = dict(meta)
    meta_out["sync"] = proc.sync_info
    meta_out["measurement_window_margin_sec"] = qa_metrics.DEFAULT_WINDOW_MARGIN_SEC
    t_start, t_end = qa_metrics.measurement_window(proc.sync_info)
    meta_out["measurement_window_sec"] = [t_start, t_end]
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta_out, f, indent=4)

    # Export 01 (Rohdaten nach Alignment / Vor Kinematik)
    save_7d_array(proc.df_json, ETD_COLS, out_dir / "01_after_align_etd.csv")
    save_7d_array(proc.df_csv, PHANTOM_COLS, out_dir / "01_after_align_phantom.csv")

    # 2. Kinematische Transformation in klinische Koordinaten (erzeugt True_* Spalten)
    proc.apply_kinematics(couch_angle=float(meta.get("couch_angle", 0.0)))

    # Export 02 (Aligned + Kinematik angewendet)
    save_7d_array(proc.df_json, ETD_COLS, out_dir / "02_aligned_kin_applied_etd.csv")
    save_7d_array(proc.df_csv, PHANTOM_COLS, out_dir / "02_aligned_kin_applied_phantom.csv")

    # Export 03 (RMSE über den gesamten Bereich - Übersicht, nicht die QA-Metrik)
    calculate_and_save_rmse(proc.df_json, proc.df_csv, out_dir)

    return {
        'm_id': m_id,
        'meta': meta_out,
        'out_dir': out_dir,
        'g_label': g_label,
        'couch_folder': couch_folder,
        'pad_folder': pad_folder_name,
        'df_phantom': proc.df_csv,
        'df_etd': proc.df_json,
        'sync_info': proc.sync_info,
        'surf_csv': str(csv_files[0]),
        'window': (t_start, t_end),
    }


def run_processing(measurements, target_linac, raw_base_dir, process_base_dir, date_str, full_config):
    """Verarbeitet alle ausgewählten Messungen und sammelt die Ergebnis-Records ein.

    full_config (ungefiltert!) wird für compute_pair_index gebraucht: die Paar-Zuordnung muss
    auch dann korrekt bleiben, wenn measurements durch einen Deflection-/Pad-Filter nur einen
    Teil der Geschwister-Einträge enthält.
    """
    records = []
    for m_id, meta in measurements:
        pad_folder_name = "32" if meta.get("heatingpads") == "32" else "RT"
        g_label = group_label(meta, target_linac, pad_folder_name)
        print(f"\n[{g_label}] ---> Starte Export & Alignment")
        try:
            rec = process_measurement(m_id, meta, target_linac, raw_base_dir, process_base_dir, date_str, full_config)
            if rec is None:
                continue
            records.append(rec)
            print(f"   ✅ ID {m_id} verarbeitet und in {rec['out_dir']} gespeichert.")
        except Exception as e:
            print(f"   [X] FEHLER bei ID {m_id}: {e}")
    return records


# ------------------------------------------
# Modus: plotpaper (Publikationsplots, ein PDF je Messreihe)
# ------------------------------------------
def run_plotpaper(records, process_base_dir, date_str):
    """Publikationsplot je Messreihe. Jede Messreihe ist genau ein ETD-Scan (kein Mitteln
    mehrerer Läufe mehr), Phantom- und ETD-Kurve werden direkt auf ihrer eigenen, bereits
    ausgerichteten Zeitachse geplottet - kein gemeinsames Binning-Raster nötig."""
    for rec in records:
        g_label = rec['g_label']
        print(f"\nGeneriere Paper-Plot für {g_label}...")

        df_ihd = rec['df_phantom']
        df_et = rec['df_etd']
        t_ihd = df_ihd['Time_Sec'].values - df_ihd['Time_Sec'].iloc[0]
        t_et = df_et['Time_Sec'].values - df_ihd['Time_Sec'].iloc[0]

        def ihd_vals(col):
            return np.array([getattr(v, 'n', v) for v in df_ihd[col]])

        # ETD-Werte über DOF_SPEC holen, damit die Vorzeichen-Konvention (invertierte
        # ETD-Vertikalachse) identisch zu RMSE- und numqa-Auswertung ist.
        def et_vals(dof_name):
            return DOF_BY_NAME[dof_name].etd_sign * df_et[DOF_BY_NAME[dof_name].etd_col].values

        trans_et = {'X': et_vals('lateral'), 'Y': et_vals('longitudinal'), 'Z': et_vals('vertical')}
        trans_ihd = {'X': ihd_vals('True_Lateral'), 'Y': ihd_vals('True_Longitudinal'),
                     'Z': ihd_vals('True_Vertical')}
        rot_et = {'pitch': et_vals('pitch'), 'yaw': et_vals('yaw'), 'roll': et_vals('roll')}
        rot_ihd = {'pitch': ihd_vals('True_Pitch'), 'yaw': ihd_vals('True_Yaw'),
                   'roll': ihd_vals('True_Roll')}

        rmse3d_vals = df_et['rmse3d'].values if 'rmse3d' in df_et.columns else np.zeros_like(t_et)
        rmse_temp_vals = df_et['rmse_temp'].values if 'rmse_temp' in df_et.columns else np.zeros_like(t_et)

        save_path = process_base_dir / rec['couch_folder'] / f"{g_label}_{date_str}_Plot.pdf"

        res = plot_evaluation_results_interactive(
            t_ihd=t_ihd, t_et=t_et, trans_et=trans_et, trans_ihd=trans_ihd, rot_et=rot_et, rot_ihd=rot_ihd,
            t_rmse=t_et, rmse3d=rmse3d_vals, rmse_temp=rmse_temp_vals,
            save_path=save_path, group_name=g_label
        )

        if res == 'exit':
            print("Abbruch durch Benutzer.")
            return


# ------------------------------------------
# Modus: plotqa (QA-Übersicht, ein A4-PDF je Temperatursetting)
# ------------------------------------------
def run_plotqa(records, target_linac, process_base_dir, date_str):
    """Ein PDF je (Pads x Couch-Winkel). Beide Deflections liegen in denselben Achsen.

    Ausgewertet werden laut Absprache nur single-angle-Messreihen (die Couch-Serie ist laut
    Messprotokoll für die jährliche QA nicht erforderlich).
    """
    qa_records = [r for r in records if r['meta'].get('meas_couch_type') == 'single angle']
    skipped = len(records) - len(qa_records)
    if skipped:
        print(f"\n[i] plotqa: {skipped} multi-angle-Messung(en) übersprungen (nur single angle).")

    if not qa_records:
        print("Keine single-angle-Messungen für plotqa gefunden.")
        return

    groups = {}
    for rec in qa_records:
        key = (rec['pad_folder'], rec['meta'].get('couch_angle'))
        groups.setdefault(key, []).append(rec)

    for (pad_folder, couch_angle), recs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        recs = sorted(recs, key=lambda r: r['meta'].get('deflection', 0))
        print(f"\nGeneriere QA-Übersicht für Pads={pad_folder}, Couch={couch_angle}...")

        # AE-Zeitreihe (Pass/Watch/Act) je Messung berechnen und exportieren - Basis für die
        # Toleranzbaender/rot markierten Zeitfenster im Plot und (separat) für numqa.
        for r in recs:
            try:
                ae_df = qa_metrics.compute_ae_time_series(r['df_phantom'], r['df_etd'], r['window'])
                ae_df.to_csv(r['out_dir'] / "03_alldof_ae_time.csv", index=False, sep=';', decimal='.')
                r['ae_time'] = ae_df
            except ValueError as e:
                print(f"   [!] AE-Zeitreihe nicht berechenbar für {r['g_label']}: {e}")
                r['ae_time'] = None

        # Sphere-Detection-Tabelle aus der zugehörigen *_QA.csv (alle Deflections gemeinsam).
        sd_table = None
        try:
            sd_path = sphere_detection.qa_csv_path_for(recs[0]['surf_csv'])
            if os.path.exists(sd_path):
                runs = [{'deflection': r['meta'].get('deflection'),
                         'couch_angle': r['meta'].get('couch_angle', 0),
                         'df_etd': r['df_etd'],
                         't_eval': r['window'][1]} for r in recs]
                sd_table = sphere_detection.build_sphere_detection_table(sd_path, runs)
            else:
                print(f"   [!] Keine Sphere-Detection-Datei gefunden: {sd_path}")
        except Exception as e:
            print(f"   [!] Sphere-Detection-Tabelle nicht erstellbar: {e}")

        title = f"ETDS QA Overview - L{target_linac} | Pads {pad_folder} | Couch {couch_angle}deg"
        save_path = process_base_dir / recs[0]['couch_folder'] / \
            f"L{target_linac}_{pad_folder}_Couch{couch_angle}_{date_str}_QA.pdf"

        plot_qa_overview(recs, sd_table=sd_table, title=title, save_path=save_path)
        print(f"   [✓] QA-Übersicht gespeichert: {save_path}")


# ------------------------------------------
# Modus: numqa (numerische Auswertung im Messfenster)
# ------------------------------------------
def find_latest_measurement_dir(process_base_dir, couch_folder, g_label):
    """Neuester Verarbeitungslauf einer Messreihe (Ordner '<g_label>_<datum>')."""
    parent = process_base_dir / couch_folder
    if not parent.exists():
        return None
    candidates = sorted([d for d in parent.iterdir() if d.is_dir() and d.name.startswith(g_label + "_")])
    return candidates[-1] if candidates else None


def run_numqa(measurements, target_linac, process_base_dir, date_str):
    """Liest die exportierten 02-CSVs, berechnet die Metriken im Messfenster und aggregiert.

    Setzt einen vorherigen Lauf von 'process'/'plotpaper'/'plotqa' voraus - dabei entstehen
    die 02-CSVs und die Sync-Zeitstempel in der metadata.json.
    """
    qa_measurements = [(m, meta) for m, meta in measurements
                       if meta.get('meas_couch_type') == 'single angle']
    skipped = len(measurements) - len(qa_measurements)
    if skipped:
        print(f"[i] numqa: {skipped} multi-angle-Messung(en) übersprungen (nur single angle).")

    if not qa_measurements:
        print("Keine single-angle-Messungen für numqa gefunden.")
        return

    records = []
    rate_records = []
    ae_time_dfs = []
    for m_id, meta in qa_measurements:
        pad_folder_name = "32" if meta.get("heatingpads") == "32" else "RT"
        g_label = group_label(meta, target_linac, pad_folder_name)
        couch_folder = meas_couch_type_folder(meta)

        meas_dir = find_latest_measurement_dir(process_base_dir, couch_folder, g_label)
        if meas_dir is None:
            print(f"   [X] ID {m_id} ({g_label}): kein Verarbeitungsordner gefunden. "
                  f"Bitte zuerst 'process' oder 'plotqa' ausfuehren.")
            continue

        ph_path = meas_dir / "02_aligned_kin_applied_phantom.csv"
        etd_path = meas_dir / "02_aligned_kin_applied_etd.csv"
        meta_path = meas_dir / "metadata.json"

        missing = [p.name for p in (ph_path, etd_path, meta_path) if not p.exists()]
        if missing:
            print(f"   [X] ID {m_id} ({g_label}): fehlende Datei(en) {missing} in {meas_dir}. "
                  f"Bitte zuerst 'process' oder 'plotqa' ausfuehren.")
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_stored = json.load(f)

        if not meta_stored.get("sync"):
            print(f"   [X] ID {m_id} ({g_label}): metadata.json enthaelt keine Sync-Zeitstempel "
                  f"(alter Verarbeitungslauf). Bitte 'process' erneut ausfuehren.")
            continue

        df_ph = pd.read_csv(ph_path, sep=';', decimal='.')
        df_etd = pd.read_csv(etd_path, sep=';', decimal='.')

        try:
            window = qa_metrics.measurement_window(
                meta_stored["sync"],
                meta_stored.get("measurement_window_margin_sec", qa_metrics.DEFAULT_WINDOW_MARGIN_SEC))
            metrics = qa_metrics.compute_dof_metrics(df_ph, df_etd, window)
            # Self-contained (numqa setzt kein plotqa voraus): AE-Zeitreihe hier selbst neu
            # berechnen statt sich auf eine evtl. vorhandene 03_alldof_ae_time.csv zu verlassen.
            ae_df = qa_metrics.compute_ae_time_series(df_ph, df_etd, window)
        except ValueError as e:
            print(f"   [X] ID {m_id} ({g_label}): {e}")
            continue

        metrics.to_csv(meas_dir / "03_alldof_parameters.csv", index=False, sep=';', decimal='.')
        ae_df.to_csv(meas_dir / "03_alldof_ae_time.csv", index=False, sep=';', decimal='.')
        rates = qa_metrics.compute_pass_watch_rates(ae_df)

        records.append({'meta': meta_stored, 'metrics': metrics})
        rate_records.append({'meta': meta_stored, 'rates': rates})
        ae_time_dfs.append(ae_df)
        print(f"   ✅ ID {m_id} ({g_label}): Fenster {window[0]:.2f}-{window[1]:.2f}s, "
              f"{int(metrics['N_Samples'].iloc[0])} Punkte -> 03_alldof_parameters.csv + 03_alldof_ae_time.csv")

    if not records:
        print("\nKeine auswertbaren Messungen - keine Gesamttabelle erzeugt.")
        return

    table = qa_metrics.build_numqa_table(records)
    out_path = process_base_dir / f"ETDS_L{target_linac}_numqa_{date_str}.csv"
    table.to_csv(out_path, index=False, sep=';', decimal='.')
    print(f"\n[✓] Gesamttabelle ({len(table)} Zeilen) gespeichert: {out_path}")

    rate_table = qa_metrics.build_passrate_table(rate_records)
    rate_path = process_base_dir / f"ETDS_L{target_linac}_passrate_{date_str}.csv"
    rate_table.to_csv(rate_path, index=False, sep=';', decimal='.')
    print(f"[✓] Passraten-Tabelle ({len(rate_table)} Zeilen) gespeichert: {rate_path}")

    # Gesamt-Passrate je DoF, über alle Deflection/Pad-Kombinationen gepoolt (Rohpunkte
    # zusammengezählt, nicht die Prozentsätze der Einzelmessungen gemittelt - siehe
    # qa_metrics.pool_ae_time_series).
    summary_table = qa_metrics.pool_ae_time_series(ae_time_dfs)
    summary_path = process_base_dir / f"ETDS_L{target_linac}_passrate_summary_{date_str}.csv"
    summary_table.to_csv(summary_path, index=False, sep=';', decimal='.')
    print(f"[✓] Passraten-Summary ({len(summary_table)} Zeilen) gespeichert: {summary_path}")


def main():
    if len(sys.argv) < 3:
        print("Nutzung: python etds_qa_evaluation.py <funktion> <linac_id> [deflection: 1/2/all] [heatingpads: RT/32/all]")
        print("  Funktionen:")
        print("    process    - Alignment + Kinematik, schreibt 01/02/03-CSVs und metadata.json")
        print("    plotpaper  - wie process, zusaetzlich interaktiver Publikationsplot je Messreihe")
        print("    plotqa     - wie process, zusaetzlich QA-Uebersicht (A4) je Temperatursetting")
        print("    numqa      - numerische Auswertung im Messfenster (setzt einen process-Lauf voraus)")
        print("  Beispiel: python etds_qa_evaluation.py plotqa 1 all")
        print("  Beispiel: python etds_qa_evaluation.py numqa 1")
        return

    raw_mode = sys.argv[1].lower()
    mode = MODE_ALIASES.get(raw_mode, raw_mode)
    if mode not in VALID_MODES:
        print(f"Unbekannte Funktion '{sys.argv[1]}'. Erlaubt: {', '.join(VALID_MODES)}.")
        return
    if raw_mode in MODE_ALIASES:
        print(f"[i] '{raw_mode}' ist ein Alias fuer '{mode}'.")

    target_linac = sys.argv[2]

    # Optionale Filter: fehlendes Argument oder "all" -> kein Filter (alles verarbeiten)
    target_deflection_input = _parse_optional_filter(sys.argv, 3)
    target_pads_input = _parse_optional_filter(sys.argv, 4)

    with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
        full_config = json.load(f)

    measurements = select_measurements(full_config, target_linac,
                                       target_deflection_input, target_pads_input)
    if not measurements:
        print(f"Keine passenden Messungen für Linac {target_linac} "
              f"(deflection={target_deflection_input or 'all'}, heatingpads={target_pads_input or 'all'}) gefunden.")
        return

    raw_base_dir = Path(f"data/raw/L{target_linac}")
    process_base_dir = Path(f"data/process/L{target_linac}")
    date_str = datetime.now().strftime("%Y-%m-%d")

    if mode == "numqa":
        run_numqa(measurements, target_linac, process_base_dir, date_str)
        return

    records = run_processing(measurements, target_linac, raw_base_dir, process_base_dir, date_str, full_config)
    if not records:
        print("\nKeine Messung erfolgreich verarbeitet.")
        return

    if mode == "plotpaper":
        run_plotpaper(records, process_base_dir, date_str)
    elif mode == "plotqa":
        run_plotqa(records, target_linac, process_base_dir, date_str)


if __name__ == "__main__":
    main()