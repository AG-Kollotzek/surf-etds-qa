import numpy as np
from scipy.optimize import minimize


class SurfKinematics:
    def __init__(self):
        # 1. Konstanten & Bauteilmaße (Idealisiert, ohne Unsicherheiten)
        self.hAxis_horizontal = 24.25 + 147.4 + 9.2 + 14.5
        self.hAxis_vertical = 14.5 + 20 + 10
        self.hAxis_diagonal = np.sqrt(self.hAxis_horizontal ** 2 + self.hAxis_vertical ** 2)

        rotatationTable_height = 130
        phantomCenterDistance = 75
        phantomToTableDistance = 22
        self.sliderShift = 30 + 16
        self.radius = rotatationTable_height + phantomToTableDistance + phantomCenterDistance + self.sliderShift

    def calculate_task_space(self, h_raw, v_raw, r_raw_deg, couch_angle_raw_deg=0.0):
        """ Vorwärts-Kinematik: Berechnet die globalen ETD-Koordinaten aus den physischen Achsenwerten. """
        h_u = np.asarray(h_raw)
        v_u = np.asarray(v_raw)
        r_u = np.deg2rad(np.asarray(r_raw_deg))

        # ==========================================
        # SCHRITT 1: LOKALE KINEMATIK (Couch = 0°)
        # ==========================================
        alpha_offset = np.arctan(self.hAxis_vertical / self.hAxis_horizontal)

        # Sicherheits-Clip: Verhindert, dass der Solver beim Ausprobieren den Sinus sprengt
        sin_val = np.clip((self.hAxis_vertical + v_u) / self.hAxis_diagonal, -1.0, 1.0)
        pitch_rad_local = np.arcsin(sin_val) - alpha_offset
        pitch_deg_local = np.degrees(pitch_rad_local)

        rollOffset = self.hAxis_horizontal - np.sqrt(self.hAxis_diagonal ** 2 - (self.hAxis_vertical + v_u) ** 2)

        y_local = -(self.radius + self.hAxis_vertical) * np.sin(pitch_rad_local) + rollOffset + (
                    h_u * np.cos(pitch_rad_local))
        x_local = h_u * 0
        z_local = -self.radius * (1 - np.cos(pitch_rad_local)) + (h_u * np.sin(pitch_rad_local))

        yaw_deg_local = np.degrees(r_u * np.cos(pitch_rad_local))
        roll_deg_local = np.degrees(r_u * np.sin(pitch_rad_local))

        # ==========================================
        # SCHRITT 2: COUCH ROTATION (Transformation)
        # ==========================================
        gamma = np.deg2rad(couch_angle_raw_deg)
        cos_g = np.cos(gamma)
        sin_g = np.sin(gamma)

        x_global = x_local * cos_g - y_local * sin_g
        y_global = x_local * sin_g + y_local * cos_g
        z_global = - z_local

        pitch_global = pitch_deg_local * cos_g - roll_deg_local * sin_g
        roll_global = pitch_deg_local * sin_g + roll_deg_local * cos_g
        yaw_global = yaw_deg_local - couch_angle_raw_deg

        # Die np.atleast_1d Rückgabe ist wichtig, falls Skalare übergeben werden
        return {
            'True_Lateral': x_global,
            'True_Longitudinal': y_global,
            'True_Vertical': z_global,
            'True_Pitch': pitch_global,
            'True_Roll': roll_global,
            'True_Yaw': yaw_global
        }

    def find_optimal_axes(self, target_params, couch_deg):
        """
        Rückwärts-Kinematik mit Penalty-Methode.
        Extrem robust gegen gekoppelte Achsen und "Sattelpunkte".
        """

        def objective(axes):
            # Berechne Ist-Position mit aktuellen Test-Achsen
            res = self.calculate_task_space(axes[0], axes[1], 0.0, couch_deg)
            error = 0.0

            # Harte Strafe: Abweichungen vom Zielwert tun weh (Faktor 10.000)
            if target_params.get('x') is not None:
                error += (res['True_Lateral'] - target_params['x']) ** 2 * 10000
            if target_params.get('y') is not None:
                error += (res['True_Longitudinal'] - target_params['y']) ** 2 * 10000
            if target_params.get('z') is not None:
                error += (res['True_Vertical'] - target_params['z']) ** 2 * 10000
            if target_params.get('pitch') is not None:
                error += (res['True_Pitch'] - target_params['pitch']) ** 2 * 10000

            # Leichte Strafe: Große Achsbewegungen vermeiden (Minimalprinzip)
            error += (axes[0] ** 2 + axes[1] ** 2) * 0.01

            return error

        # Start bei 0.1 statt 0.0 umarmt den Null-Gradienten bei Z!
        res = minimize(objective, x0=[0.1, 0.1], bounds=[(-150, 150), (-60, 60)], method='L-BFGS-B')

        # Letzter Check: Wurde das Ziel mit einer Toleranz von 50 Mikrometern / 0.05° erreicht?
        # Wenn nicht, waren die Vorgaben wirklich geometrisch unmöglich.
        final_res = self.calculate_task_space(res.x[0], res.x[1], 0.0, couch_deg)

        for key, res_key in [('x', 'True_Lateral'), ('y', 'True_Longitudinal'),
                             ('z', 'True_Vertical'), ('pitch', 'True_Pitch')]:
            if target_params.get(key) is not None:
                if abs(final_res[res_key] - target_params[key]) > 0.05:
                    raise ValueError(f"Position mechanisch unlösbar! Konflikt bei {key.upper()}.")

        return {'H': res.x[0], 'V': res.x[1]}