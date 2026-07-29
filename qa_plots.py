"""
QA-Uebersichtsplot (plotqa): eine A4-Seite je Temperatursetting.

Aufbau (8 Felder von oben nach unten):
    1-6  je ein Freiheitsgrad (Longitudinal, Lateral, Vertical, Roll, Pitch, Yaw).
         Beide Deflections liegen in denselben Achsen, Phantom-Soll gestrichelt,
         ETD-Ist durchgezogen. Der graue schraffierte Bereich markiert das Messfenster,
         die duennen gestrichelten Linien die Mitten der beiden Sync-Pulse. Um die
         Phantom-Kurve liegt ein Toleranz-Schlauch (gruen = accept, gelb = watch, siehe
         qa_metrics.TOLERANCE_ACCEPT/WATCH); Zeitabschnitte, in denen der ETD-Fehler die
         Watch-Schwelle ueberschreitet ('act'), sind rot hinterlegt. Die Schraffur des
         Messfenster-Balkens (/// bzw. \\\) kennzeichnet zusaetzlich zur Farbe, welcher
         Deflection er zugehoert - bleibt so auch in Graustufen-Ausdrucken unterscheidbar.
         Nur das 7. Feld traegt noch Zahlen auf der Zeitachse, die DoF-Felder darueber
         teilen sich die x-Achse.
    7    RMSE-3D und RMSE-Thermal des ETD-Systems.
    8    Vergleichstabelle der Sphere-Detection-Punkte (roentgenbasierte Referenz).

Gemeinsame Zeitachse: alle Messungen werden auf die Mitte ihres ERSTEN Sync-Pulses
bezogen (t = 0). Dadurch starten beide Deflections deckungsgleich; ihre Messfenster enden
unterschiedlich spaet, weil die grosse Auslenkung laenger dauert.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from qa_metrics import DOF_SPEC, TOLERANCE_ACCEPT, TOLERANCE_WATCH, find_out_of_tolerance_intervals

# Farben je Deflection (colorblind-safe, konsistent mit den Paper-Plots)
DEFLECTION_COLORS = {1: '#D55E00', 2: '#0072B2'}
_FALLBACK_COLOR = '#555555'

# Schraffur je Deflection fuer den Messfenster-Balken - zusaetzlich zur Farbe, damit die
# beiden Deflections auch in Graustufen-Ausdrucken unterscheidbar bleiben.
DEFLECTION_HATCHES = {1: '///', 2: '\\\\\\'}
_FALLBACK_HATCH = '...'

# Toleranz-Schlauch (accept/watch) und Act-Markierung um die Phantom-Kurve
_COLOR_ACCEPT = '#2ca02c'
_COLOR_WATCH = '#f2c744'
_COLOR_ACT = '#d62728'

# Mindestspanne der y-Achse in den DoF-Feldern, damit der Toleranz-Schlauch bei kleinen
# Auslenkungen (z.B. Roll/Yaw bei Deflection 1) nicht eine winzige, verwirrende Achse sprengt.
MIN_Y_SPAN = 6.0

A4_PORTRAIT_INCHES = (8.27, 11.69)


def _color_for(deflection):
    return DEFLECTION_COLORS.get(deflection, _FALLBACK_COLOR)


def _hatch_for(deflection):
    return DEFLECTION_HATCHES.get(deflection, _FALLBACK_HATCH)


def _relative_time(values, t_zero):
    return np.asarray(values, dtype=float) - t_zero


def _draw_measurement_window(ax, t_win_start, t_win_end, color, schraff):
    """Grau schraffierter Hintergrund fuer den ausgewerteten Zeitraum (dezent)."""
    ax.axvspan(t_win_start, t_win_end, facecolor='0.6', alpha=0.15,
               hatch=schraff, edgecolor=color, linewidth=0.0, zorder=0)


def _draw_act_bars(ax, ae_time_df, dof_name, t_zero):
    """Rot hinterlegte Zeitabschnitte, in denen der Fehler > TOLERANCE_WATCH liegt."""
    if ae_time_df is None:
        return
    for t_start, t_end in find_out_of_tolerance_intervals(ae_time_df, dof_name, state='act'):
        ax.axvspan(t_start - t_zero, t_end - t_zero, facecolor=_COLOR_ACT, alpha=0.28,
                   linewidth=0.0, zorder=0.3)


def _draw_tolerance_tube(ax, t_phantom, phantom_vals):
    """Toleranz-Schlauch um die Phantom-Kurve: gruen (accept), gelb (watch)."""
    ax.fill_between(t_phantom, phantom_vals - TOLERANCE_ACCEPT, phantom_vals + TOLERANCE_ACCEPT,
                    color=_COLOR_ACCEPT, alpha=0.3, linewidth=0.0, zorder=0.5)
    ax.fill_between(t_phantom, phantom_vals - TOLERANCE_WATCH, phantom_vals - TOLERANCE_ACCEPT,
                    color=_COLOR_WATCH, alpha=0.35, linewidth=0.0, zorder=0.5)
    ax.fill_between(t_phantom, phantom_vals + TOLERANCE_ACCEPT, phantom_vals + TOLERANCE_WATCH,
                    color=_COLOR_WATCH, alpha=0.35, linewidth=0.0, zorder=0.5)


def _enforce_min_y_span(ax, min_span=MIN_Y_SPAN):
    """Erweitert die y-Achse symmetrisch um ihren Mittelpunkt auf mindestens min_span."""
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    if span < min_span:
        center = (ymin + ymax) / 2.0
        ax.set_ylim(center - min_span / 2.0, center + min_span / 2.0)


def _draw_sync_lines(ax, t_sync_first, t_sync_last, color):
    for t in (t_sync_first, t_sync_last):
        ax.axvline(t, color=color, linestyle=':', linewidth=0.9, alpha=0.8, zorder=1)


def _render_sd_table(ax, sd_table):
    """Sphere-Detection-Vergleichstabelle als Matplotlib-Tabelle."""
    ax.axis('off')

    if sd_table is None or sd_table.empty:
        ax.text(0.5, 0.5, 'Keine Sphere-Detection-Daten verfuegbar',
                ha='center', va='center', fontsize=8, style='italic', color='0.35')
        return

    header_map = {
        'Deflection': 'Defl.',
        'SD - Reference Coordinates': 'SD Reference\nCoordinates',
        'SD - Movement Vector': 'SD Movement\nVector',
        'Phantom Movement Vector': 'Phantom Movement\nVector',
        'Movement Error Vector': 'Movement Error\n(SD - Phantom)',
        'Surface Tracking Vector': 'Surface Tracking\nVector',
    }
    columns = [header_map.get(c, c) for c in sd_table.columns]
    cell_text = sd_table.astype(str).values.tolist()

    table = ax.table(cellText=cell_text, colLabels=columns, cellLoc='center', loc='lower center')
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('0.75')
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor('0.92')
            cell.set_text_props(fontweight='bold')
        elif col == 0:
            cell.set_text_props(fontweight='bold')

    # Als Text statt set_title, damit die Ueberschrift nicht mit der Achsenbeschriftung
    # des darueberliegenden RMSE-Feldes kollidiert.
    ax.text(0.5, 0.97, 'Sphere detection comparison  -  vectors as (x, y, z) in mm',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=8.5, fontweight='bold')


def plot_qa_overview(records, sd_table=None, title="ETDS QA Overview", save_path=None):
    """Erzeugt die A4-QA-Uebersicht und speichert sie als PDF.

    records: Liste der Messungs-Records (je Deflection einer), jeweils mit
             'df_phantom', 'df_etd', 'sync_info', 'window' und 'meta'.
    sd_table: DataFrame aus sphere_detection.build_sphere_detection_table (oder None).
    """
    prev_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # reine Datei-Ausgabe, kein interaktives Fenster noetig
    try:
        plt.rcParams.update({
            'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
            'font.size': 8, 'axes.edgecolor': 'black', 'axes.linewidth': 0.9,
            'legend.frameon': True, 'legend.edgecolor': 'black',
        })

        fig, axes = plt.subplots(
            8, 1, figsize=A4_PORTRAIT_INCHES, dpi=150,
            gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1, 1, 1.2]})
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.985)
        # Feste Raender statt tight_layout: die Tabelle im letzten Feld wuerde die
        # automatische Layout-Berechnung sonst verzerren. hspace klein, da die DoF-Felder
        # (1-6) keine eigene Zahlen-Zeitachse mehr tragen und dadurch enger zusammenruecken.
        fig.subplots_adjust(left=0.115, right=0.975, top=0.955, bottom=0.025, hspace=0.22)

        # Zeitbezug: Mitte des ersten Sync-Pulses jeder Messung wird auf t = 0 gelegt.
        prepared = []
        for rec in records:
            sync = rec['sync_info']
            t_zero = float(sync['aligned_first_mid'])
            win_start, win_end = rec['window']
            prepared.append({
                'deflection': rec['meta'].get('deflection'),
                'color': _color_for(rec['meta'].get('deflection')),
                'schraff': _hatch_for(rec['meta'].get('deflection')),
                'df_phantom': rec['df_phantom'],
                'df_etd': rec['df_etd'],
                'ae_time': rec.get('ae_time'),
                't_zero': t_zero,
                't_phantom': _relative_time(rec['df_phantom']['Time_Sec'].values, t_zero),
                't_etd': _relative_time(rec['df_etd']['Time_Sec'].values, t_zero),
                'sync_first_rel': 0.0,
                'sync_last_rel': float(sync['aligned_last_mid']) - t_zero,
                'win_start_rel': win_start - t_zero,
                'win_end_rel': win_end - t_zero,
            })

        # --- Felder 1-6: die sechs Freiheitsgrade ---
        for ax, dof in zip(axes[:6], DOF_SPEC):
            for p in prepared:
                _draw_measurement_window(ax, p['win_start_rel'], p['win_end_rel'], p['color'], p['schraff'])
                _draw_act_bars(ax, p['ae_time'], dof.name, p['t_zero'])
                _draw_tolerance_tube(ax, p['t_phantom'], p['df_phantom'][dof.phantom_col].values)
                _draw_sync_lines(ax, p['sync_first_rel'], p['sync_last_rel'], p['color'])

                ax.plot(p['t_phantom'], p['df_phantom'][dof.phantom_col].values,
                        color=p['color'], linestyle='--', linewidth=0.6,
                        label=f"IHD D{p['deflection']}")
                ax.plot(p['t_etd'], dof.etd_values(p['df_etd']),
                        color=p['color'], linestyle='-', linewidth=0.7,
                        label=f"ET D{p['deflection']}")

            ax.set_ylabel(f"{dof.label}\n[{dof.unit}]", fontsize=7.5)
            ax.grid(True, linestyle=':', alpha=0.5, linewidth=0.4)
            ax.tick_params(labelsize=7, labelbottom=False)
            _enforce_min_y_span(ax)
            ax.legend(loc='upper right', fontsize=5.5, ncol=2, handlelength=1.8,
                      borderpad=0.3, labelspacing=0.25, columnspacing=0.9)

        # --- Feld 7: RMSE-Kanaele des ETD-Systems ---
        ax_rmse = axes[6]
        for p in prepared:
            df_etd = p['df_etd']
            if 'rmse3d' in df_etd.columns:
                ax_rmse.plot(p['t_etd'], df_etd['rmse3d'].values, color=p['color'],
                             linestyle='-', linewidth=1.0, label=f"RMSE 3D D{p['deflection']}")
            if 'rmse_temp' in df_etd.columns:
                ax_rmse.plot(p['t_etd'], df_etd['rmse_temp'].values, color=p['color'],
                             linestyle='--', linewidth=1.0, label=f"RMSE Temp D{p['deflection']}")
            _draw_measurement_window(ax_rmse, p['win_start_rel'], p['win_end_rel'], p['color'], p['schraff'])
            _draw_sync_lines(ax_rmse, p['sync_first_rel'], p['sync_last_rel'], p['color'])

        ax_rmse.set_ylabel('RMSE', fontsize=7.5)
        ax_rmse.set_xlabel('Time relative to first sync pulse [s]', fontsize=8)
        ax_rmse.grid(True, linestyle=':', alpha=0.5, linewidth=0.6)
        ax_rmse.tick_params(labelsize=7)
        ax_rmse.legend(loc='upper right', fontsize=5.5, ncol=2, handlelength=1.8,
                       borderpad=0.3, labelspacing=0.25, columnspacing=0.9)

        # Gemeinsame x-Grenzen fuer die Zeitreihen-Felder
        if prepared:
            x_min = min(float(np.min(p['t_phantom'])) for p in prepared)
            x_max = max(float(np.max(p['t_phantom'])) for p in prepared)
            for ax in axes[:7]:
                ax.set_xlim(x_min, x_max)

        # Zusaetzlicher Abstand vor Feld 8: bei dem knappen hspace (siehe oben) sitzt die
        # Tabellen-Ueberschrift sonst zu nah an der x-Achsenbeschriftung von Feld 7.
        pos7 = axes[7].get_position()
        gap = 0.025
        axes[7].set_position([pos7.x0, pos7.y0, pos7.width, pos7.height - gap])

        # --- Feld 8: Sphere-Detection-Vergleichstabelle ---
        _render_sd_table(axes[7], sd_table)

        if save_path is not None:
            fig.savefig(save_path, format='pdf')
        plt.close(fig)
        return save_path
    finally:
        matplotlib.use(prev_backend, force=False)
