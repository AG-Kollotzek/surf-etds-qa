import numpy as np
from uncertainties import ufloat
from uncertainties import unumpy as unp
import uncertainties.umath as umath


class SurfKinematics:
    """
    v2: exakte Rotationskomposition statt Kleinwinkel-Trigonometrie-Mixing.

    Mechanik: H-Achse haengt wie eine Leiter an der
    V-Achse und wird durch V aktiv aufgestellt/abgesenkt (-> pitch_local, rein aus V,
    unveraendert gegenueber v1). Auf dem H-Wagen sitzt ein Drehtisch (R-Achse), der 90 Grad
    zur H-Achse steht. Da der Drehtisch AUF dem H-Wagen montiert ist, kippt er mit pitch_local
    mit - die R-Drehung findet also um die BEREITS gekippte lokale Z-Achse statt.
    Physikalische Reihenfolge: erst Pitch (Rx), danach R um die gekippte lokale Z-Achse (Rz):

        R_local = Rx(pitch_local) @ Rz(r_u)

    v1 hatte das nur als Kleinwinkel-Naeherung (yaw=r*cos(pitch), roll=r*sin(pitch))
    umgesetzt - exakt nur fuer kleines r_u (dort wo v1 auch "perfekt" mit dem Scanner matchte,
    r_u <= ~2 Grad in den mindev/maxdev-Gruppen). Bei r_u ~ 90 Grad (Couch-Rotations-Serie,
    Kopf dauerhaft Richtung Kamera gedreht) weicht das um >15 Grad vom Scanner ab.

    Diese Version dekomponiert R_local exakt (geschlossene Form, per Hand aus der
    Rotationsmatrix hergeleitet, Konvention R = Rz(yaw)*Ry(roll)*Rx(pitch)):

        pitch_local_out = atan2(sin(p)*cos(r), cos(p))
        roll_local_out   = asin(sin(p)*sin(r))
        yaw_local_out    = atan2(cos(p)*sin(r), cos(r))

    mit p=pitch_local (rad), r=r_u (rad).

    Empirisch validiert gegen echte ETD-Scans (Messungen 7 & 8, mindev/maxdev_couch-90,
    H=-10..-25, V=20..50, r_u=-91..-92 Grad):
      Roll  ETD vs. Modell:  -5.9 / -5.95 Grad   und  -15.41 / -15.30 Grad  (<1% Abweichung)
      Yaw   ETD vs. Modell:  -1.0 / -1.00 Grad   und  -2.14 /  -2.05 Grad  (<0.1 Grad)
    Das Vorzeichen von roll_local_out wurde dabei empirisch bestimmt (Haendigkeit der
    Y-Achsen-Konvention war a priori nicht bekannt) - siehe Docstring von calculate_task_space.

    OFFENER PUNKT (bewusst nicht "wegkorrigiert"): Pitch zeigt einen kleinen, mit der
    Auslenkung wachsenden Restfehler (ETD -0.3/-1.17 Grad vs. Modell -0.10/-0.54 Grad).
    Deutlich kleiner als der urspruengliche Fehler (der ging in die 15+ Grad), aber noch
    nicht Null - moeglicher zusaetzlicher, hier noch nicht erfasster Kopplungseffekt.

    NICHT GEPRUEFT in dieser Version: das Vorzeichen von couch_angle_raw_deg in x_global/
    y_global (Translation) wurde 1:1 aus v1 uebernommen und NICHT empirisch gegen ETD lateral/
    longitudinal validiert - nur die Rotationskoordinaten (pitch/roll/yaw) wurden hier geprueft.
    """

    def __init__(self, err_linear=0.0025, err_rot=0.1, err_couch=0.5):
        # 1. Konstanten & Bauteilmaße (mit Messunsicherheit der Schublehre in mm)
        ERR = 0.2  # Standardfehler bei Schublehre
        # Hebel der H-Achse
        self.hAxis_horizontal = ufloat(24.25, 0.5) + ufloat(147.4, ERR) + ufloat(9.2, ERR) + ufloat(14.5, 0.5)
        self.hAxis_vertical = ufloat(14.5, 0.5) + ufloat(20, ERR) + ufloat(10, ERR) + ufloat(0.5, ERR)
        self.hAxis_diagonal = unp.sqrt(self.hAxis_horizontal ** 2 + self.hAxis_vertical ** 2)

        # Distanzen Phantom zu H-Schlitten in Home-Position
        rotatationTable_height = ufloat(130, ERR)
        phantomCenterDistance = ufloat(75, ERR)
        phantomToTableDistance = ufloat(21, 1)  # Nachmessen!
        self.sliderShift = ufloat(30, ERR) + ufloat(16, ERR)
        self.radius = rotatationTable_height + phantomToTableDistance + phantomCenterDistance + self.sliderShift + self.hAxis_vertical

        # Standard-Fehler der Achsen und Couch
        self.err_linear = err_linear
        self.err_rot_rad = np.deg2rad(err_rot)
        self.err_couch_rad = np.deg2rad(err_couch)

    def calculate_task_space(self, h_raw, v_raw, r_raw_deg, couch_angle_raw_deg=0.0):
        """
        1. Berechnet die lokalen klinischen Koordinaten (auf dem Tisch, couch=0).
        2. Rotiert diese um den Couch-Winkel ins globale ETD-Raumsystem.
        """
        # Schritt 0: Fehler an input anhängen
        h_u = unp.uarray(h_raw, self.err_linear)
        v_u = unp.uarray(v_raw, self.err_linear)

        r_rad_raw = np.deg2rad(r_raw_deg)
        r_u = unp.uarray(r_rad_raw, self.err_rot_rad)

        # ==========================================
        # SCHRITT 1: LOKALE KINEMATIK (Couch = 0°)
        # ==========================================
        alpha_offset = unp.arctan(self.hAxis_vertical / self.hAxis_horizontal)
        pitch_rad_local = unp.arcsin((self.hAxis_vertical + v_u) / self.hAxis_diagonal) - alpha_offset
        pitch_deg_local = unp.degrees(pitch_rad_local)  # nur fuer x/y/z-Translation, unveraendert ggue. v1

        rollOffset = self.hAxis_horizontal - unp.sqrt(self.hAxis_diagonal ** 2 - (self.hAxis_vertical + v_u) ** 2)
        y_local = -(self.radius * unp.sin(pitch_rad_local)) + rollOffset - (h_u * unp.cos(pitch_rad_local))
        x_local = h_u * 0
        z_local = -(self.radius * (1 - unp.cos(pitch_rad_local))) - (h_u * unp.sin(pitch_rad_local))

        # --- Exakte Rotationskomposition R_local = Rx(pitch_local) @ Rz(r_u) ---
        # (ersetzt die v1-Kleinwinkel-Naeherung yaw=r*cos(pitch), roll=r*sin(pitch);
        #  Vorzeichen von roll_local_out empirisch bestimmt, siehe Klassen-Docstring)
        sin_p, cos_p = unp.sin(pitch_rad_local), unp.cos(pitch_rad_local)
        sin_r, cos_r = unp.sin(r_u), unp.cos(r_u)

        pitch_local_out_rad = unp.arctan2(sin_p * cos_r, cos_p)
        roll_local_out_rad = unp.arcsin(sin_p * sin_r)
        yaw_local_out_rad = unp.arctan2(cos_p * sin_r, cos_r)

        pitch_deg_local_out = unp.degrees(pitch_local_out_rad)
        roll_deg_local_out = unp.degrees(roll_local_out_rad)
        yaw_deg_local_out = unp.degrees(yaw_local_out_rad)

        # ==========================================
        # SCHRITT 2: COUCH ROTATION (Transformation)
        # ==========================================
        couch_angle_raw_rad = np.deg2rad(couch_angle_raw_deg)
        gamma = ufloat(couch_angle_raw_rad, self.err_couch_rad)
        cos_g = unp.cos(gamma)
        sin_g = unp.sin(gamma)

        # Translation: unveraendert ggue. v1 (Achtung: Vorzeichen von couch_angle hier
        # NICHT empirisch geprueft in dieser Version, siehe Klassen-Docstring)
        x_global = x_local * cos_g - y_local * sin_g #beachte isozentrische drehung der Couch, falls über die couch drehung in zukunft mit getrackt würde!
        y_global = x_local * sin_g + y_local * cos_g
        z_global = z_local

        # Rotation: pitch/roll sind unter einer reinen Z-Rotation (Couch) exakt invariant
        # (koerperbezogene/"aviation-style" Konvention - algebraisch bewiesen und empirisch
        # bestaetigt, siehe Docstring). Nur Yaw verschiebt sich additiv mit dem Couch-Winkel.
        pitch_global = pitch_deg_local_out
        roll_global = roll_deg_local_out
        gamma_deg = unp.degrees(gamma)
        yaw_global = yaw_deg_local_out - gamma_deg

        return {
            'True_Lateral': x_global,
            'True_Longitudinal': y_global,
            'True_Vertical': z_global,
            'True_Pitch': pitch_global,
            'True_Roll': roll_global,
            'True_Yaw': yaw_global
        }
