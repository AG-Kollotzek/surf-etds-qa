r"""
QA-Uebersichtsplot (plotqa): zwei A4-Seiten je Temperatursetting.

SEITE 1 - Zeitreihen-Uebersicht, Aufbau (8 Felder von oben nach unten):
    1-6  je ein Freiheitsgrad (Longitudinal, Lateral, Vertical, Roll, Pitch, Yaw).
         Beide Deflections liegen in denselben Achsen, Phantom-Soll gestrichelt,
         ETD-Ist durchgezogen. Der graue schraffierte Bereich markiert das Messfenster
         (Schraffur /// bzw. \\\ zusaetzlich zur Farbe, damit beide Deflections auch in
         Graustufen-Ausdrucken unterscheidbar bleiben). Die Mitten der beiden Sync-Pulse
         sind als kleine schwarze Marker am oberen Rahmen der Box markiert - je Deflection
         ein eigenes Symbol (Dreieck/Pfeil bzw. Hohlkreis, siehe DEFLECTION_MARKERS). Um
         die Phantom-Kurve liegt ein Toleranz-Schlauch (gruen = accept, gelb = watch,
         siehe qa_metrics.TOLERANCE_ACCEPT/WATCH); Zeitabschnitte, in denen der ETD-Fehler
         die Watch-Schwelle ueberschreitet ('act'), sind rot hinterlegt.
         Nur das 7. Feld traegt noch Zahlen auf der Zeitachse, die DoF-Felder darueber
         teilen sich die x-Achse.
    7    RMSE-3D und RMSE-Thermal des ETD-Systems.
    8    Vergleichstabelle der Sphere-Detection-Punkte (roentgenbasierte Referenz).

Statt einer Legende je Feld gibt es eine einzige gemeinsame Legende ganz unten auf der
Seite (siehe _build_master_legend_handles), die alle Elemente (ETD-Tracking, RMSE-Linien,
Sync-Peaks, Messfenster, Toleranz-Schlauch) einmal fuer beide Deflections beschreibt.

Gemeinsame Zeitachse: alle Messungen werden auf die Mitte ihres ERSTEN Sync-Pulses
bezogen (t = 0). Dadurch starten beide Deflections deckungsgleich; ihre Messfenster enden
unterschiedlich spaet, weil die grosse Auslenkung laenger dauert.

SEITE 2 - Spider-Plot der Sphere-Detection-Abweichungen (plot_sphere_detection_spider):
    Radarplot mit den drei Translationsachsen (Lateral, Longitudinal, Vertical). Aufgetragen
    sind die BETRAEGE zweier Abweichungen je Deflection, jeweils an der ausgelenkten Position
    (Ende des Messfensters):
        |SD - Phantom|           roentgenbasierte Kugelposition gegen das Achs-Soll
        |SD - Surface Tracking|  roentgenbasierte Kugelposition gegen das ETD-Tracking
    Deflection 1 in Rottoenen, Deflection 2 in Blautoenen (SPIDER_SHADES); der dunklere Ton
    ist jeweils der Vergleich gegen das Phantom-Soll. Zusaetzlich sind die Toleranzringe aus
    qa_metrics.TOLERANCE_ACCEPT/WATCH eingezeichnet. Unter dem Radarplot stehen dieselben
    Werte noch einmal als Tabelle, da sich Betraege im Radar nicht ablesen lassen.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

from qa_metrics import DOF_SPEC, TOLERANCE_ACCEPT, TOLERANCE_WATCH, find_out_of_tolerance_intervals

# Farben je Deflection (colorblind-safe, konsistent mit den Paper-Plots)
DEFLECTION_COLORS = {1: '#D55E00', 2: '#0072B2'}
_FALLBACK_COLOR = '#555555'

# Schraffur je Deflection fuer den Messfenster-Balken - zusaetzlich zur Farbe, damit die
# beiden Deflections auch in Graustufen-Ausdrucken unterscheidbar bleiben.
DEFLECTION_HATCHES = {1: '///', 2: '\\\\\\'}
_FALLBACK_HATCH = '...'

# Symbol je Deflection fuer die Sync-Marker am Rahmen (Farbe immer schwarz) - damit beide
# Sync-Punkte auch bei Ueberlappung und in Graustufen-Ausdrucken unterscheidbar bleiben.
DEFLECTION_MARKERS = {1: 'v', 2: 'o'}
_FALLBACK_MARKER = 's'

# Groesse der Sync-Marker am Rahmen: der Hohlkreis (D2) wirkt bei gleicher Groesse wie das
# Dreieck (D1) optisch kleiner/leichter und wird daher etwas grosszuegiger dimensioniert.
DEFLECTION_MARKER_SIZES = {1: 5.0, 2: 7.0}
_FALLBACK_MARKER_SIZE = 5.0

# Toleranz-Schlauch (accept/watch) und Act-Markierung um die Phantom-Kurve
_COLOR_ACCEPT = '#2ca02c'
_COLOR_WATCH = '#f2c744'
_COLOR_ACT = '#d62728'

# Mindestspanne der y-Achse in den DoF-Feldern, damit der Toleranz-Schlauch bei kleinen
# Auslenkungen (z.B. Roll/Yaw bei Deflection 1) nicht eine winzige, verwirrende Achse sprengt.
MIN_Y_SPAN = 6.0

# Skalierungsfaktor fuer die gemeinsame Legende (Schrift, Handles, Abstaende) - 25% groesser
# als der Basiswert, damit sie ganz oben auf der Seite besser lesbar ist.
_LEGEND_SCALE = 1.25

A4_PORTRAIT_INCHES = (8.27, 11.69)

# --- Spider-Plot (Seite 2): Sphere-Detection-Abweichungen ---------------------------------
# Achsen des Spider-Plots, Reihenfolge = Komponentenreihenfolge der Vektoren aus
# sphere_detection.build_sphere_detection_vectors (= qa_metrics.VECTOR_DOFS).
SPIDER_AXIS_LABELS = ('Lateral (X)', 'Longitudinal (Y)', 'Vertical (Z)')

# Je Deflection zwei Abstufungen: Deflection 1 rot, Deflection 2 blau (Vorgabe). Der
# dunklere Ton steht jeweils fuer den Vergleich gegen das Achs-Soll (Phantom), der hellere
# fuer den Vergleich gegen das Surface-Tracking.
SPIDER_SHADES = {
    1: {'vs_phantom': '#B2182B', 'vs_surface': '#EF8A62'},
    2: {'vs_phantom': '#2166AC', 'vs_surface': '#67A9CF'},
}
_SPIDER_FALLBACK_SHADES = {'vs_phantom': '#444444', 'vs_surface': '#999999'}

# Linienstil/Marker trennen die beiden Vergleichsarten zusaetzlich zur Helligkeit, damit
# der Plot auch im Graustufen-Ausdruck lesbar bleibt.
_SPIDER_STYLES = {
    'vs_phantom': {'linestyle': '-', 'marker': 'o'},
    'vs_surface': {'linestyle': '--', 'marker': 's'},
}


def _shades_for(deflection):
    return SPIDER_SHADES.get(deflection, _SPIDER_FALLBACK_SHADES)


def _nice_ticks(vmax, n=4):
    """Runde Radialticks bis knapp ueber vmax (ohne die 0, die faellt mit dem Zentrum zusammen)."""
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    raw = vmax / n
    exp = np.floor(np.log10(raw))
    frac = raw / (10 ** exp)
    nice = 1.0 if frac <= 1 else 2.0 if frac <= 2 else 5.0 if frac <= 5 else 10.0
    step = nice * (10 ** exp)
    ticks = np.arange(step, step * (n + 1.5), step)
    return ticks[ticks <= step * (np.ceil(vmax / step) + 0.001)]


def _spider_series(sd_vectors):
    """Die vier Kurven des Spider-Plots als Liste von dicts (Betraege der Abweichungen).

    Reihenfolge: je Deflection zuerst der Vergleich gegen das Phantom-Soll, dann gegen das
    Surface-Tracking. Serien ohne jeden gueltigen Wert (z.B. leere Sphere-Spalten in der
    *_QA.csv) werden weggelassen.
    """
    series = []
    for entry in sorted(sd_vectors, key=lambda e: e.get('deflection', 0)):
        deflection = entry.get('deflection')
        shades = _shades_for(deflection)
        for key, desc in (('error_vs_phantom', 'SD - Phantom'),
                          ('error_vs_surface', 'SD - Surface Tracking')):
            values = np.abs(np.asarray(entry[key], dtype=float))
            if not np.any(np.isfinite(values)):
                continue
            style_key = 'vs_phantom' if key == 'error_vs_phantom' else 'vs_surface'
            series.append({
                'label': f"D{deflection}: |{desc}|",
                'values': values,
                'color': shades[style_key],
                **_SPIDER_STYLES[style_key],
            })
    return series


def _render_spider(ax, series):
    """Radarplot der Betragsabweichungen auf den drei Translationsachsen."""
    n_axes = len(SPIDER_AXIS_LABELS)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)
    closed = np.concatenate([angles, angles[:1]])

    ax.set_theta_offset(np.pi / 2)   # erste Achse nach oben
    ax.set_theta_direction(-1)       # im Uhrzeigersinn

    if not series:
        ax.text(0.5, 0.5, 'Keine Sphere-Detection-Daten verfuegbar',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=9, style='italic', color='0.35')
        ax.set_xticks([])
        ax.set_yticks([])
        return

    vmax = np.nanmax([np.nanmax(s['values']) for s in series])
    # Der Accept-Ring soll immer sichtbar bleiben, damit die Werte einen festen Bezug haben;
    # der Watch-Ring nur, wenn er die Datenskala nicht unnoetig stauchen wuerde.
    r_max = max(float(vmax) * 1.18, TOLERANCE_ACCEPT * 1.18)

    for level, color, name in ((TOLERANCE_ACCEPT, _COLOR_ACCEPT, 'accept'),
                               (TOLERANCE_WATCH, _COLOR_WATCH, 'watch')):
        if level <= r_max:
            ax.plot(np.linspace(0, 2 * np.pi, 181), np.full(181, level),
                    color=color, linewidth=1.0, linestyle=':', zorder=1.5)
            ax.text(np.deg2rad(62), level, f" {name} ({level:g} mm)",
                    fontsize=6.5, color=color, ha='left', va='bottom', zorder=6)

    for s in series:
        vals = np.asarray(s['values'], dtype=float)
        vals_closed = np.concatenate([vals, vals[:1]])
        ax.plot(closed, vals_closed, color=s['color'], linewidth=1.6,
                linestyle=s['linestyle'], marker=s['marker'], markersize=4.5,
                label=s['label'], zorder=3)
        ax.fill(closed, vals_closed, color=s['color'], alpha=0.07, zorder=2)

    ax.set_xticks(angles)
    ax.set_xticklabels(SPIDER_AXIS_LABELS, fontsize=9, fontweight='bold')
    ax.tick_params(axis='x', pad=18)

    ticks = _nice_ticks(r_max / 1.18)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks], fontsize=7, color='0.35')
    ax.set_ylim(0, r_max)
    ax.set_rlabel_position(90.0 / len(SPIDER_AXIS_LABELS))
    ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.6)
    ax.set_axisbelow(True)

    # Radiale Achslinien betonen (die drei Speichen).
    for ang in angles:
        ax.plot([ang, ang], [0, r_max], color='0.55', linewidth=0.8, zorder=1)


def _render_spider_values(ax, sd_vectors):
    """Kompakte Wertetabelle unter dem Spider-Plot (Betraege lassen sich im Radar nicht ablesen)."""
    ax.axis('off')
    if not sd_vectors:
        return

    rows = []
    for entry in sorted(sd_vectors, key=lambda e: e.get('deflection', 0)):
        d = entry.get('deflection')
        for key, desc in (('error_vs_phantom', 'SD - Phantom'),
                          ('error_vs_surface', 'SD - Surface Tracking')):
            vals = np.abs(np.asarray(entry[key], dtype=float))
            rows.append([f"D{d}", f"|{desc}|"] + ['-' if not np.isfinite(v) else f"{v:.2f}" for v in vals])

    table = ax.table(cellText=rows,
                     colLabels=['Defl.', 'Deviation'] + [f"{lbl} [mm]" for lbl in SPIDER_AXIS_LABELS],
                     cellLoc='center', loc='upper center')
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.4)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('0.75')
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor('0.92')
            cell.set_text_props(fontweight='bold')
        elif col <= 1:
            cell.set_text_props(fontweight='bold')


def plot_sphere_detection_spider(sd_vectors, title="Sphere detection - absolute deviations",
                                 subtitle=None):
    """Baut die Spider-Plot-Seite (A4) und gibt die Figure zurueck.

    sd_vectors: Liste aus sphere_detection.build_sphere_detection_vectors (oder None).
    Gezeigt werden je Deflection die Betraege von (SD - Phantom) und (SD - Surface Tracking)
    auf den drei Translationsachsen.
    """
    sd_vectors = sd_vectors or []
    series = _spider_series(sd_vectors)

    fig = plt.figure(figsize=A4_PORTRAIT_INCHES, dpi=150)
    fig.suptitle(title, fontsize=9, fontweight='bold', y=0.985)
    if subtitle:
        fig.text(0.5, 0.958, subtitle, ha='center', va='top', fontsize=8, color='0.3')

    # Die Polar-Achse ist quadratisch; die Hoehe ist hier der begrenzende Faktor, der Kreis
    # fuellt die Box also vertikal aus. Legende und Tabelle schliessen direkt darunter an.
    ax = fig.add_axes([0.10, 0.44, 0.80, 0.465], projection='polar')
    _render_spider(ax, series)

    if series:
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.425), ncol=2, fontsize=7.5,
                   frameon=True, edgecolor='black', handlelength=2.4,
                   borderpad=0.5, labelspacing=0.4, columnspacing=1.4)

    ax_tbl = fig.add_axes([0.08, 0.16, 0.84, 0.20])
    _render_spider_values(ax_tbl, sd_vectors)

    fig.text(0.5, 0.235,
             'Absolute deviations of the X-ray based sphere detection from the phantom axis '
             'setpoint and from the ETD surface tracking,\nevaluated at the deflected position '
             f'(end of measurement window). Tolerance rings: accept {TOLERANCE_ACCEPT:g} mm, '
             f'watch {TOLERANCE_WATCH:g} mm.',
             ha='center', va='top', fontsize=7, color='0.35')

    return fig


def _color_for(deflection):
    return DEFLECTION_COLORS.get(deflection, _FALLBACK_COLOR)


def _hatch_for(deflection):
    return DEFLECTION_HATCHES.get(deflection, _FALLBACK_HATCH)


def _marker_for(deflection):
    return DEFLECTION_MARKERS.get(deflection, _FALLBACK_MARKER)


def _marker_size_for(deflection):
    return DEFLECTION_MARKER_SIZES.get(deflection, _FALLBACK_MARKER_SIZE)


def _relative_time(values, t_zero):
    return np.asarray(values, dtype=float) - t_zero


def _draw_measurement_window(ax, t_win_start, t_win_end, color, schraff, deflection):
    """Grau schraffierter Hintergrund fuer den ausgewerteten Zeitraum (dezent).

    Kein label= hier: die Legende soll die Schraffur mit voller Deckkraft zeigen (siehe
    _window_legend_handle), waehrend die Flaeche im Plot selbst dezent bleibt.
    """
    ax.axvspan(t_win_start, t_win_end, facecolor='0.6', alpha=0.15,
               hatch=schraff, edgecolor=color, linewidth=0.0, zorder=0)


def _window_legend_handle(color, schraff, deflection):
    """Legend-Proxy fuer das Messfenster mit voller Deckkraft (bessere Erkennbarkeit),
    unabhaengig von der dezenten Transparenz der Flaeche im Plot."""
    return mpatches.Patch(facecolor='0.6', edgecolor=color, hatch=schraff, alpha=1.0,
                           label=f"Measurement window D{deflection}")


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


def _draw_sync_lines(ax, t_sync_first, t_sync_last, marker, marker_size, deflection):
    """Kleine schwarze Marker am oberen Rahmen der Box: Mitten der beiden Sync-Pulse.

    Symbol je Deflection (siehe DEFLECTION_MARKERS), damit sich beide Messungen auch bei
    Ueberlappung unterscheiden lassen; erscheint dank label= auch in der Legende.
    """
    trans = ax.get_xaxis_transform()  # x in Datenkoordinaten, y als Achsen-Fraktion
    hollow = marker == 'o'
    ax.plot([t_sync_first, t_sync_last], [1.0, 1.0],
            marker=marker, linestyle='none', markersize=marker_size, markeredgewidth=1.0,
            markerfacecolor='none' if hollow else 'black', markeredgecolor='black',
            transform=trans, clip_on=False, zorder=5,
            label=f"Sync Peak D{deflection}")


def _build_master_legend_handles(prepared):
    """Baut die Handles fuer die eine gemeinsame Legende am unteren Rand der Seite.

    Statt in jedem Feld eine eigene Legende zu wiederholen, sammelt diese Funktion je
    einen Proxy-Handle pro Element (ETD-Punkte, RMSE-Linien, Sync-Peaks, Messfenster,
    Toleranz-Schlauch) fuer alle Deflections.
    """
    handles = []
    for p in prepared:
        handles.append(Line2D([], [], color=p['color'], linestyle='none', marker='o',
                               markersize=3.5 * _LEGEND_SCALE,
                               label=f"ETDS Tracking D{p['deflection']}"))
    for p in prepared:
        handles.append(Line2D([], [], color=p['color'], linestyle='-', linewidth=1.2 * _LEGEND_SCALE,
                               label=f"RMSE 3D D{p['deflection']}"))
    for p in prepared:
        handles.append(Line2D([], [], color=p['color'], linestyle='--', linewidth=1.2 * _LEGEND_SCALE,
                               label=f"RMSE Temp D{p['deflection']}"))
    for p in prepared:
        hollow = p['marker'] == 'o'
        handles.append(Line2D([], [], color='black', linestyle='none', marker=p['marker'],
                               markersize=p['marker_size'] * _LEGEND_SCALE, markeredgewidth=1.0,
                               markerfacecolor='none' if hollow else 'black',
                               markeredgecolor='black', label=f"Sync Peak D{p['deflection']}"))
    for p in prepared:
        handles.append(_window_legend_handle(p['color'], p['schraff'], p['deflection']))
    handles.append(mpatches.Patch(facecolor=_COLOR_ACCEPT, alpha=0.7, label='Tolerance tube accept'))
    handles.append(mpatches.Patch(facecolor=_COLOR_WATCH, alpha=0.7, label='Tolerance tube watch'))
    return handles


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


def plot_qa_overview(records, sd_table=None, title="ETDS QA Overview", save_path=None,
                     sd_vectors=None):
    """Erzeugt die A4-QA-Uebersicht und speichert sie als zweiseitiges PDF.

    Seite 1: die acht Felder der Zeitreihen-Uebersicht (siehe Modul-Docstring).
    Seite 2: Spider-Plot der Sphere-Detection-Abweichungen (plot_sphere_detection_spider).

    records: Liste der Messungs-Records (je Deflection einer), jeweils mit
             'df_phantom', 'df_etd', 'sync_info', 'window' und 'meta'.
    sd_table: DataFrame aus sphere_detection.build_sphere_detection_table (oder None).
    sd_vectors: Liste aus sphere_detection.build_sphere_detection_vectors (oder None) -
             Datenbasis der zweiten Seite.
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
        fig.suptitle(title, fontsize=9, fontweight='bold', y=0.996)
        # Feste Raender statt tight_layout: die Tabelle im letzten Feld wuerde die
        # automatische Layout-Berechnung sonst verzerren. hspace klein, da die DoF-Felder
        # (1-6) keine eigene Zahlen-Zeitachse mehr tragen und dadurch enger zusammenruecken
        # (hspace=0.20 entspricht ca. 5.5mm Abstand). top laesst nur noch ein knappes
        # Polster zur gemeinsamen Legende direkt unter dem Titel (siehe fig.legend weiter
        # unten), statt einer Legende in jedem Feld.
        fig.subplots_adjust(left=0.115, right=0.975, top=0.925, bottom=0.03, hspace=0.20)

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
                'marker': _marker_for(rec['meta'].get('deflection')),
                'marker_size': _marker_size_for(rec['meta'].get('deflection')),
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
                _draw_measurement_window(ax, p['win_start_rel'], p['win_end_rel'], p['color'], p['schraff'], p['deflection'])
                _draw_act_bars(ax, p['ae_time'], dof.name, p['t_zero'])
                _draw_tolerance_tube(ax, p['t_phantom'], p['df_phantom'][dof.phantom_col].values)
                _draw_sync_lines(ax, p['sync_first_rel'], p['sync_last_rel'], p['marker'], p['marker_size'], p['deflection'])

                ax.plot(p['t_etd'], dof.etd_values(p['df_etd']),
                        color=p['color'], linestyle='None', marker='o', markersize=0.5,
                        label=f"ETDS Tracking D{p['deflection']}")

            ax.set_ylabel(f"{dof.label}\n[{dof.unit}]", fontsize=7.5)
            ax.grid(True, linestyle=':', alpha=0.5, linewidth=0.4)
            ax.tick_params(labelsize=7, labelbottom=False)
            _enforce_min_y_span(ax)

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
            _draw_measurement_window(ax_rmse, p['win_start_rel'], p['win_end_rel'], p['color'], p['schraff'], p['deflection'])
            _draw_sync_lines(ax_rmse, p['sync_first_rel'], p['sync_last_rel'], p['marker'], p['marker_size'], p['deflection'])

        ax_rmse.set_ylabel('RMSE', fontsize=7.5)
        ax_rmse.set_xlabel('Time relative to first sync pulse [s]', fontsize=8)
        ax_rmse.grid(True, linestyle=':', alpha=0.5, linewidth=0.6)
        ax_rmse.tick_params(labelsize=7)

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

        # Eine gemeinsame Legende direkt unter dem Titel, ganz oben auf der Seite -
        # statt einer Legende in jedem Feld.
        master_handles = _build_master_legend_handles(prepared)
        fig.legend(handles=master_handles, loc='upper center', ncol=6,
                   fontsize=6.3 * _LEGEND_SCALE,
                   frameon=True, edgecolor='black', bbox_to_anchor=(0.5, 0.975),
                   handlelength=1.6 * _LEGEND_SCALE, borderpad=0.4 * _LEGEND_SCALE,
                   labelspacing=0.35 * _LEGEND_SCALE, columnspacing=1.0 * _LEGEND_SCALE,
                   handletextpad=0.4 * _LEGEND_SCALE)


        # Seite 2: Spider-Plot der Sphere-Detection-Abweichungen. Wird auch ohne Daten
        # erzeugt (mit Hinweistext), damit der Seitenaufbau des Reports konstant bleibt.
        fig_spider = plot_sphere_detection_spider(
            sd_vectors,
            title='Sphere detection - absolute deviations',
            subtitle=title)

        if save_path is not None:
            with PdfPages(save_path) as pdf:
                pdf.savefig(fig)
                pdf.savefig(fig_spider)
        plt.close(fig)
        plt.close(fig_spider)
        return save_path
    finally:
        matplotlib.use(prev_backend, force=False)
