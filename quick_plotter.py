import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def plot_single_measurement(temp="RT", gruppe="Vertical", meas_id="7"):
    # 1. ID immer zweistellig formatieren und Gruppen-Leerzeichen abfangen
    meas_str = str(meas_id).strip().zfill(2)
    gruppe_folder = gruppe.replace(" ", "_")

    # Namens-Mapping abfangen (falls du aus Gewohnheit "Longitudinal" tippst)
    if gruppe_folder == "Longitudinal":
        gruppe_folder = "Horizontal"

    base_dir = Path(f"paper_data/process_export/{temp}/{gruppe_folder}/meas_{meas_str}")

    # Zustand 01 (Vor Alignment)
    etd_01_f = base_dir / "01_before_align_etd.csv"
    phan_01_f = base_dir / "01_before_align_phantom.csv"

    # Zustand 02 (Nach Alignment)
    etd_02_f = base_dir / "02_after_align_etd.csv"
    phan_02_f = base_dir / "02_after_align_phantom.csv"

    # Prüfen ob alles da ist
    files = [etd_01_f, phan_01_f, etd_02_f, phan_02_f]
    if not all(f.exists() for f in files):
        print(f"❌ Fehler: Nicht alle CSV-Dateien (Zustand 01 & 02) gefunden in {base_dir}")
        return

    # Daten laden
    df_etd_01 = pd.read_csv(etd_01_f, sep=';', decimal='.')
    df_phan_01 = pd.read_csv(phan_01_f, sep=';', decimal='.')
    df_etd_02 = pd.read_csv(etd_02_f, sep=';', decimal='.')
    df_phan_02 = pd.read_csv(phan_02_f, sep=';', decimal='.')

    # Mapping der Spaltenpaare (ETD, Phantom)
    dofs = [
        ('lateral', 'True_Lateral'),
        ('longitudinal', 'True_Longitudinal'),
        ('vertical', 'True_Vertical'),
        ('pitch', 'True_Pitch'),
        ('yaw', 'True_Yaw'),
        ('roll', 'True_Roll')
    ]

    # Erstelle 6 Subplots untereinander
    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[f"--- {d[0].upper()} ---" for d in dofs]
    )

    # Farbpalette für die Unterscheidung definieren
    # Zustand 1 (Blass/Transparent da "alt") | Zustand 2 (Kräftig da "aktuell")
    colors = {
        'etd_01': 'rgba(31, 119, 180, 0.4)',  # Blassblau
        'phan_01': 'rgba(255, 127, 14, 0.4)',  # Blassorange
        'etd_02': 'rgba(31, 119, 180, 1.0)',  # Voll-Blau
        'phan_02': 'rgba(255, 127, 14, 1.0)'  # Voll-Orange
    }

    for i, (etd_col, phan_col) in enumerate(dofs, start=1):
        # --- ZUSTAND 01: BEFORE ALIGN (Blasse Kurven) ---
        fig.add_trace(
            go.Scatter(x=df_etd_01['Time_Sec'], y=df_etd_01[etd_col],
                       name=f'ETD [Vor Align]', line=dict(color=colors['etd_01'], width=1.5),
                       legendgroup='Zustand 01', showlegend=(i == 1)),
            row=i, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_phan_01['Time_Sec'], y=df_phan_01[phan_col],
                       name=f'Phantom [Vor Align]', line=dict(color=colors['phan_01'], width=1.5, dash='dot'),
                       legendgroup='Zustand 01', showlegend=(i == 1)),
            row=i, col=1
        )

        # --- ZUSTAND 02: AFTER ALIGN (Kräftige Kurven) ---
        fig.add_trace(
            go.Scatter(x=df_etd_02['Time_Sec'], y=df_etd_02[etd_col],
                       name=f'ETD [Nach Align]', line=dict(color=colors['etd_02'], width=2.5),
                       legendgroup='Zustand 02', showlegend=(i == 1)),
            row=i, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_phan_02['Time_Sec'], y=df_phan_02[phan_col],
                       name=f'Phantom [Nach Align]', line=dict(color=colors['phan_02'], width=2.5, dash='dash'),
                       legendgroup='Zustand 02', showlegend=(i == 1)),
            row=i, col=1
        )

    # Layout-Feinschliff für maximale Übersicht im Browser
    fig.update_layout(
        height=1400,  # Schön hoch gezerrt für 6 Achsen
        title_text=f"Vergleich Vor/Nach Alignment: {temp} | {gruppe} | Messung {meas_str}",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Achsenbeschriftung fixieren
    fig.update_xaxes(title_text="Zeit (Sekunden)", row=6, col=1)

    fig.show()


if __name__ == "__main__":
    # Aufruf korrigiert über explizite Keywords
    plot_single_measurement(temp="RT", gruppe="Vertical", meas_id="7")