import numpy as np
import cv2
import math
import time
from collections import deque

class AnalizadorFisiologico:
    def __init__(self):
        self.buffer_anchos = deque(maxlen=300)
        self.buffer_tiempos = deque(maxlen=300)

        self.picos = []
        self.valles = []
        self.tiempos_picos = []
        self.bpms_historial = []

        self.ancho_max_promedio = 0.0
        self.ancho_min_promedio = 0.0
        self.bpm_media = 0.0
        self.variacion_bpm = 0.0

        self.puntos_grafico = 200
        self.sweep_y = np.zeros(self.puntos_grafico)
        self.cursor_x = 0

        self.memoria_rayos = {}
        self.inercia = 0.85

        # --- TEMPORAL BUFFER per slice ---
        self._temporal_buffers = {}
        self._temporal_buffer_size = 60
        self._frames_acumulados = 0

        # --- SAVITZKY-GOLAY ---
        self._sg_window = 7
        self._sg_order = 3
        self._sg_coeffs = self._calcular_coeffs_savgol(self._sg_window, self._sg_order)
        self._buffer_sg = deque(maxlen=self._sg_window)

        # --- GAUSSIAN WEIGHTS per slice ---
        self._pesos_gauss = None
        self._num_rebanadas_cache = 0

    @staticmethod
    def _calcular_coeffs_savgol(window_length, polyorder):
        half_window = window_length // 2
        x = np.arange(-half_window, half_window + 1, dtype=np.float64)
        A = np.vander(x, polyorder + 1, increasing=True)
        proj = A @ np.linalg.pinv(A)
        return proj[half_window]

    def _savgol_suavizar(self, valor):
        self._buffer_sg.append(valor)
        if len(self._buffer_sg) < self._sg_window:
            return valor
        ventana = np.array(self._buffer_sg, dtype=np.float64)
        return float(np.dot(self._sg_coeffs, ventana))

    @staticmethod
    def _subpixel(arr, pico):
        if 0 < pico < len(arr) - 1:
            y1, y2, y3 = arr[pico - 1], arr[pico], arr[pico + 1]
            denom = y1 - 2 * y2 + y3
            if abs(denom) > 1e-6:
                return pico + 0.5 * (y1 - y3) / denom
        return float(pico)

    def procesar_frame(self, img_tratada, mask_4bit, motion_map, p1, p2,
                       _angulo_ignorado, num_rebanadas=10):
        t_actual = time.time()

        if img_tratada is None or img_tratada.size == 0:
            return 0, None

        h, w = img_tratada.shape
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        largo_eje = math.hypot(dx, dy)
        if largo_eje == 0:
            return 0, None

        ux, uy = dx / largo_eje, dy / largo_eje
        px, py = -uy, ux

        dist_max = int(math.hypot(w, h) / 2)

        self._frames_acumulados += 1

        # Gaussian weights (lazy init)
        if self._num_rebanadas_cache != num_rebanadas:
            indices = np.arange(1, num_rebanadas, dtype=np.float64)
            centro = num_rebanadas / 2.0
            sigma = num_rebanadas / 6.0
            self._pesos_gauss = np.exp(-0.5 * ((indices - centro) / sigma) ** 2)
            self._pesos_gauss /= self._pesos_gauss.sum()
            self._num_rebanadas_cache = num_rebanadas

        # Adaptive weights: trust temporal more as buffer fills
        fill_ratio = min(1.0, self._frames_acumulados / self._temporal_buffer_size)
        w_temporal = 0.50 * fill_ratio
        w_motion = 0.30
        w_4bit = 0.20 + 0.20 * (1.0 - fill_ratio)

        anchos = []
        puntos_linea_debug = None
        mejor_calidad = 0.0

        for i in range(1, num_rebanadas):
            t = i / num_rebanadas
            cx = int(p1[0] + dx * t)
            cy = int(p1[1] + dy * t)

            r = np.arange(-dist_max, dist_max + 1)
            x_coords = np.clip(np.round(cx + r * px).astype(int), 0, w - 1)
            y_coords = np.clip(np.round(cy + r * py).astype(int), 0, h - 1)

            perfil_int = img_tratada[y_coords, x_coords].astype(np.float32)
            perfil_4bit = mask_4bit[y_coords, x_coords].astype(np.float32)
            perfil_motion = motion_map[y_coords, x_coords].astype(np.float32)

            # --- SOURCE 1: Temporal stddev ---
            if i not in self._temporal_buffers:
                self._temporal_buffers[i] = deque(maxlen=self._temporal_buffer_size)
            self._temporal_buffers[i].append(perfil_int.copy())

            temporal_std = np.zeros_like(perfil_int)
            if len(self._temporal_buffers[i]) >= 5:
                stack = np.array(self._temporal_buffers[i])
                temporal_std = np.std(stack, axis=0)

            # --- Elastic memory constraint ---
            centro_hist, ancho_hist = self.memoria_rayos.get(
                i, (dist_max, dist_max / 2.0))

            margen_latido = max(40, ancho_hist * 0.80)
            limite_izq = int(max(0, centro_hist - margen_latido))
            limite_der = int(min(len(perfil_int) - 1, centro_hist + margen_latido))

            if limite_der - limite_izq < 5:
                continue

            # --- Normalize each source inside search window ---
            def _normalize(arr):
                a = arr[limite_izq:limite_der + 1]
                mn, mx = np.min(a), np.max(a)
                if mx - mn < 1e-4:
                    return np.zeros_like(a)
                return (a - mn) / (mx - mn)

            std_win = _normalize(temporal_std)
            mot_win = _normalize(perfil_motion)
            # Invert 4-bit: high where it is NOT cavity = wall
            inv_4bit = 15.0 - perfil_4bit
            m4_win = _normalize(inv_4bit)

            # --- WALL SCORE: weighted sum ---
            wall_score = (w_temporal * std_win +
                          w_motion * mot_win +
                          w_4bit * m4_win)

            # --- Quality check: enough total energy? ---
            if np.max(wall_score) < 0.05:
                continue

            # --- Find two strongest peaks (left half, right half) ---
            half = len(wall_score) // 2
            segmento_izq = wall_score[:half]
            segmento_der = wall_score[half:]

            pico_izq_raw = np.argmax(segmento_izq)
            pico_der_raw = np.argmax(segmento_der)
            p_izq_local = pico_izq_raw
            p_der_local = half + pico_der_raw

            # Minimum separation
            if p_der_local - p_izq_local < 3:
                continue

            calidad = (segmento_izq[pico_izq_raw] + segmento_der[pico_der_raw]) / 2.0

            # Reject low quality slices
            if calidad < 0.10:
                continue

            # --- Sub-pixel refinement ---
            idx_izq = limite_izq + self._subpixel(wall_score, p_izq_local)
            idx_der = limite_izq + self._subpixel(wall_score, p_der_local)

            ancho_rebanada = float(idx_der - idx_izq)
            centro_actual = idx_izq + (ancho_rebanada / 2.0)

            if ancho_rebanada < 2:
                continue

            anchos.append((i, ancho_rebanada, calidad))

            # Update elastic memory
            self.memoria_rayos[i] = (
                (self.inercia * centro_hist) + ((1.0 - self.inercia) * centro_actual),
                (self.inercia * ancho_hist) + ((1.0 - self.inercia) * ancho_rebanada)
            )

            # Debug line from best quality slice
            if calidad > mejor_calidad:
                mejor_calidad = calidad
                idx_izq_int = int(round(idx_izq))
                idx_der_int = int(round(idx_der))
                idx_izq_int = min(idx_izq_int, len(r) - 1)
                idx_der_int = min(idx_der_int, len(r) - 1)
                pt_izq = (int(cx + r[idx_izq_int] * px),
                          int(cy + r[idx_izq_int] * py))
                pt_der = (int(cx + r[idx_der_int] * px),
                          int(cy + r[idx_der_int] * py))
                puntos_linea_debug = (pt_izq, pt_der)

        # --- Weighted mean: Gaussian x Quality ---
        if anchos:
            indices, valores, calidades = zip(*anchos)
            w_gauss = self._pesos_gauss[np.array(indices) - 1]
            w_qual = np.array(calidades)
            pesos = w_gauss * w_qual
            pesos = pesos / pesos.sum()
            ancho_raw = float(np.dot(valores, pesos))
        else:
            ancho_raw = 0.0

        # --- Savitzky-Golay ---
        ancho_final = self._savgol_suavizar(ancho_raw) if ancho_raw > 0 else 0.0

        if ancho_final > 0:
            self.buffer_anchos.append(ancho_final)
            self.buffer_tiempos.append(t_actual)
            self.sweep_y[self.cursor_x] = ancho_final
            self.cursor_x = (self.cursor_x + 1) % self.puntos_grafico
            self._detectar_eventos()
            self._actualizar_atributos_directos()

        return ancho_final, puntos_linea_debug

    def _detectar_eventos(self):
        if len(self.buffer_anchos) < 10:
            return
        vals = list(self.buffer_anchos)

        if vals[-3] > vals[-5] and vals[-3] > vals[-1] and vals[-3] > np.mean(vals) * 1.02:
            t_ahora = time.time()
            if not self.tiempos_picos or (t_ahora - self.tiempos_picos[-1] > 0.25):
                self.picos.append(vals[-3])
                self.tiempos_picos.append(t_ahora)
                if len(self.tiempos_picos) > 1:
                    self.bpms_historial.append(
                        60.0 / (self.tiempos_picos[-1] - self.tiempos_picos[-2]))

        if vals[-3] < vals[-5] and vals[-3] < vals[-1] and vals[-3] < np.mean(vals) * 0.98:
            self.valles.append(vals[-3])

    def _actualizar_atributos_directos(self):
        self.ancho_max_promedio = np.mean(self.picos[-10:]) if self.picos else 0.0
        self.ancho_min_promedio = np.mean(self.valles[-10:]) if self.valles else 0.0
        self.bpm_media = np.mean(self.bpms_historial[-10:]) if self.bpms_historial else 0.0
        self.variacion_bpm = np.std(self.bpms_historial[-15:]) if len(self.bpms_historial) > 1 else 0.0

    def obtener_estadisticas(self):
        avg = lambda lista, n: float(np.mean(lista[-n:])) if lista else 0.0
        return {
            "sys_1": avg(self.picos, 1),
            "sys_3": avg(self.picos, 3),
            "max_prom": self.ancho_max_promedio,
            "dia_1": avg(self.valles, 1),
            "dia_3": avg(self.valles, 3),
            "min_prom": self.ancho_min_promedio,
            "freq_2": avg(self.bpms_historial, 2),
            "freq_10": self.bpm_media,
            "var_15": self.variacion_bpm,
        }

    def generar_cardiograma(self, width, height):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        step_w, step_h = width // 10, height // 8
        for i in range(0, height, step_h):
            cv2.line(img, (0, i), (width, i), (20, 35, 20), 1)
        for i in range(0, width, step_w):
            cv2.line(img, (i, 0), (i, height), (20, 35, 20), 1)

        if len(self.buffer_anchos) < 2:
            return img

        vals_activos = self.sweep_y[self.sweep_y > 0]
        min_v = np.min(vals_activos) if len(vals_activos) > 0 else 0
        max_v = np.max(vals_activos) if len(vals_activos) > 0 else 1
        rango = max(max_v - min_v, 15.0)

        dx = width / self.puntos_grafico
        brecha = int(self.puntos_grafico * 0.08)

        for i in range(self.puntos_grafico - 1):
            if (self.cursor_x <= i <= self.cursor_x + brecha) or \
               (self.cursor_x + brecha >= self.puntos_grafico and
                i < (self.cursor_x + brecha) % self.puntos_grafico):
                continue

            if i == self.cursor_x - 1 or self.sweep_y[i] == 0 or self.sweep_y[i + 1] == 0:
                continue

            y1 = int(height - (((self.sweep_y[i] - min_v) / rango) *
                               (height * 0.8) + height * 0.1))
            y2 = int(height - (((self.sweep_y[i + 1] - min_v) / rango) *
                               (height * 0.8) + height * 0.1))

            cv2.line(img, (int(i * dx), y1), (int((i + 1) * dx), y2), (0, 255, 100), 2)

        return img
