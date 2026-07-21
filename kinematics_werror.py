import numpy as np
from uncertainties import ufloat
from uncertainties import unumpy as unp
import uncertainties.umath as umath


class SurfKinematics:
    def __init__(self, err_linear=0.0025, err_rot=0.1, err_couch=0.5):
        # 1. Konstanten & Bauteilmaße (mit Messunsicherheit der Schublehre in mm)
        ERR = 0.2 #Standardfehler bei Schublehre
        # Hebel der H-Achse
        self.hAxis_horizontal = ufloat(24.25, 0.5) + ufloat(147.4, ERR) + ufloat(9.2, ERR) + ufloat(14.5, 0.5) #Lot über gelenk bis zwei schraub-nüsse, schraubnüsse bis schienenende, Dicke schwarzes Metallstück, Drehpunkt bis Montage Geleknk
        self.hAxis_vertical = ufloat(14.5, 0.5) + ufloat(20, ERR) + ufloat(10, ERR) + ufloat(0.5, ERR) #Gelenk + Profil+ halbes profil + Gap(!)
        self.hAxis_diagonal = unp.sqrt(self.hAxis_horizontal ** 2 + self.hAxis_vertical ** 2)

        # Distanzen Phantom zu H-Schlitten in Home-Position
        rotatationTable_height = ufloat(130, ERR)
        phantomCenterDistance = ufloat(75, ERR)
        phantomToTableDistance = ufloat(21, 1) #Nachmessen!
        self.sliderShift = ufloat(30, ERR) + ufloat(16, ERR) # Abstand Unterkante zu achse # Hälfte des Schlittens (Schlitten fährt auf mittigem loch)
        self.radius = rotatationTable_height + phantomToTableDistance + phantomCenterDistance + self.sliderShift + self.hAxis_vertical

        #Standard-Fehler der Achsen und Couch
        self.err_linear = err_linear
        self.err_rot_rad = np.deg2rad(err_rot)
        self.err_couch_rad = np.deg2rad(err_couch)
    def calculate_task_space(self, h_raw, v_raw, r_raw_deg, couch_angle_raw_deg=0.0):
        """
        1. Berechnet die lokalen klinischen Koordinaten (auf dem Tisch).
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
        alpha_offset = unp.arctan(self.hAxis_vertical/self.hAxis_horizontal)
        pitch_rad_local = unp.arcsin((self.hAxis_vertical + v_u) / self.hAxis_diagonal) - alpha_offset
        pitch_deg_local = unp.degrees(pitch_rad_local)

        rollOffset = self.hAxis_horizontal - unp.sqrt(self.hAxis_diagonal ** 2 - (self.hAxis_vertical + v_u) **2) #Wie verschiebt sich die rollunterlage vom schlitten, je nach kippung?
        y_local = - (self.radius * unp.sin(pitch_rad_local)) + rollOffset - (h_u * unp.cos(pitch_rad_local))
        x_local = h_u * 0
        z_local = - (self.radius * (1 - unp.cos(pitch_rad_local))) - (h_u * unp.sin(pitch_rad_local)) #Test: Radius + vertical hat gefehlt (grünes R in der Skizze)
        yaw_deg_local = unp.degrees(r_u * unp.cos(pitch_rad_local))
        roll_deg_local = unp.degrees(r_u * unp.sin(pitch_rad_local))

        # ==========================================
        # SCHRITT 2: COUCH ROTATION (Transformation)
        # ==========================================
        # Umwandlung des Winkels in Radiant und Fehlerübertragung

        couch_angle_raw_rad = np.deg2rad(couch_angle_raw_deg)
        gamma = ufloat(couch_angle_raw_rad, self.err_couch_rad)

        cos_g = unp.cos(gamma)
        sin_g = unp.sin(gamma)

        # 2D-Rotationsmatrix für die Translationsebene (X, Y)
        # (Z / Vertikal bleibt bei einer reinen Couch-Rotation um die Z-Achse identisch)
        x_global = x_local * cos_g - y_local * sin_g
        y_global = x_local * sin_g + y_local * cos_g
        z_global = z_local #kein Flip wegen ETDS werten (erst nach output)

        # Rotationsmatrix für die Winkel (Pitch, Roll), Vorzeichenfehler mit Test rausfinden
        # Bei +90° Couchrotation wird Pitch zu Roll und Roll zu Pitch
        pitch_global = pitch_deg_local * cos_g - roll_deg_local * sin_g ##wichtig wegen rot
        roll_global = pitch_deg_local * sin_g + roll_deg_local * cos_g
        yaw_global = yaw_deg_local - gamma # Der interne Yaw der Achse bleibt relativ zum Raum gleich additiv, jedoch hat die Test Unit bei gedrehtem tisch das phantom um 90 Grad wieder zum scanner gedreht

        return {
            'True_Lateral': x_global,
            'True_Longitudinal': y_global,
            'True_Vertical': z_global,
            'True_Pitch': pitch_global,
            'True_Roll': roll_global,
            'True_Yaw': yaw_global
        }
