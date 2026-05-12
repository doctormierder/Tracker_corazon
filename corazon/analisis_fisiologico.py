import numpy as np
import cv2
import math
import time
from collections import deque

class AnalizadorFisiologico:
    def __init__(self):
        # Buffers de señales fisiológicas
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
        
        # --- MEMORIA ELÁSTICA DEL ESCÁNER ---
        # Diccionario: { indice_rebanada : (centro_historico, ancho_historico) }
        self.memoria_rayos = {} 
        self.inercia = 0.85 # Fricción (0.0 = salta instantáneo, 0.99 = inamovible)

    def procesar_frame(self, mask, p1, p2, angulo_deg_ignorado, num_rebanadas=10):
        t_actual = time.time()
        
        if mask is None or np.sum(mask) == 0:
            return 0, None
            
        h, w = mask.shape
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        largo_eje = math.hypot(dx, dy)
        
        if largo_eje == 0:
            return 0, None
            
        # 1. Vector Perpendicular Estricto
        ux, uy = dx / largo_eje, dy / largo_eje
        px, py = -uy, ux 
        
        dist_max = int(math.hypot(w, h) / 2)
        anchos = []
        puntos_linea_debug = None
        
        # 2. Análisis por rebanadas con MEMORIA ELÁSTICA
        for i in range(1, num_rebanadas):
            t = i / num_rebanadas
            cx = int(p1[0] + dx * t)
            cy = int(p1[1] + dy * t)
            
            # Generar rayo de coordenadas transversales
            r = np.arange(-dist_max, dist_max + 1)
            x_coords = np.clip(np.round(cx + r * px).astype(int), 0, w - 1)
            y_coords = np.clip(np.round(cy + r * py).astype(int), 0, h - 1)
            
            perfil = mask[y_coords, x_coords]
            
            # Recuperar estado histórico de esta rebanada específica
            centro_hist, ancho_hist = self.memoria_rayos.get(i, (dist_max, dist_max / 2.0))
            
            # 3. VENTANA DE VISIÓN RESTRINGIDA (Evita ruidos lejanos)
            margen_latido = max(30, ancho_hist * 0.75) 
            limite_izq = int(max(0, centro_hist - margen_latido))
            limite_der = int(min(len(perfil) - 1, centro_hist + margen_latido))
            
            zona_permitida = np.zeros_like(perfil)
            zona_permitida[limite_izq:limite_der] = perfil[limite_izq:limite_der]
            
            blancos = np.where(zona_permitida > 0)[0]
            
            if len(blancos) < 2:
                continue # Rebanada vacía en esta iteración
                
            # 4. IGNORAMOS EL HUECO NEGRO CENTRAL (Sístole)
            # Tomamos exclusivamente la pared exterior izquierda y derecha
            idx_izq = blancos[0]
            idx_der = blancos[-1]
            
            ancho_rebanada = float(idx_der - idx_izq)
            centro_actual = idx_izq + (ancho_rebanada / 2.0)
            
            anchos.append(ancho_rebanada)
            
            # 5. ACTUALIZAR FRICCIÓN (El escáner es arrastrado suavemente)
            self.memoria_rayos[i] = (
                (self.inercia * centro_hist) + ((1.0 - self.inercia) * centro_actual),
                (self.inercia * ancho_hist) + ((1.0 - self.inercia) * ancho_rebanada)
            )
            
            # Guardar puntos para dibujar en pantalla
            if i == num_rebanadas // 2 or puntos_linea_debug is None:
                pt_izq = (int(cx + r[idx_izq] * px), int(cy + r[idx_izq] * py))
                pt_der = (int(cx + r[idx_der] * px), int(cy + r[idx_der] * py))
                puntos_linea_debug = (pt_izq, pt_der)
                
        # --- PROCESAMIENTO DE SEÑALES ---
        ancho_final = np.mean(anchos) if anchos else 0.0
        
        if ancho_final > 0:
            self.buffer_anchos.append(ancho_final)
            self.buffer_tiempos.append(t_actual)
            self.sweep_y[self.cursor_x] = ancho_final
            self.cursor_x = (self.cursor_x + 1) % self.puntos_grafico
            self._detectar_eventos()
            self._actualizar_atributos_directos()
            
        return ancho_final, puntos_linea_debug

    def _detectar_eventos(self):
        if len(self.buffer_anchos) < 10: return
        vals = list(self.buffer_anchos)
        
        # Pico (Sístole / Ancho Máximo)
        if vals[-3] > vals[-5] and vals[-3] > vals[-1] and vals[-3] > np.mean(vals) * 1.02:
            t_ahora = time.time()
            if not self.tiempos_picos or (t_ahora - self.tiempos_picos[-1] > 0.25):
                self.picos.append(vals[-3])
                self.tiempos_picos.append(t_ahora)
                if len(self.tiempos_picos) > 1:
                    self.bpms_historial.append(60.0 / (self.tiempos_picos[-1] - self.tiempos_picos[-2]))
        
        # Valle (Diástole / Ancho Mínimo)
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
            "sys_1": avg(self.picos, 1), "sys_3": avg(self.picos, 3), "max_prom": self.ancho_max_promedio,
            "dia_1": avg(self.valles, 1), "dia_3": avg(self.valles, 3), "min_prom": self.ancho_min_promedio,
            "freq_2": avg(self.bpms_historial, 2), "freq_10": self.bpm_media,
            "var_15": self.variacion_bpm
        }

    def generar_cardiograma(self, width, height):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        step_w, step_h = width // 10, height // 8
        for i in range(0, height, step_h): cv2.line(img, (0, i), (width, i), (20, 35, 20), 1)
        for i in range(0, width, step_w): cv2.line(img, (i, 0), (i, height), (20, 35, 20), 1)

        if len(self.buffer_anchos) < 2: return img
        
        vals_activos = self.sweep_y[self.sweep_y > 0]
        min_v = np.min(vals_activos) if len(vals_activos) > 0 else 0
        max_v = np.max(vals_activos) if len(vals_activos) > 0 else 1
        rango = max(max_v - min_v, 15.0) 
        
        dx = width / self.puntos_grafico
        brecha = int(self.puntos_grafico * 0.08)

        for i in range(self.puntos_grafico - 1):
            if (self.cursor_x <= i <= self.cursor_x + brecha) or \
               (self.cursor_x + brecha >= self.puntos_grafico and i < (self.cursor_x + brecha) % self.puntos_grafico):
                continue
            
            if i == self.cursor_x - 1 or self.sweep_y[i] == 0 or self.sweep_y[i+1] == 0: 
                continue

            y1 = int(height - (((self.sweep_y[i] - min_v) / rango) * (height * 0.8) + height * 0.1))
            y2 = int(height - (((self.sweep_y[i+1] - min_v) / rango) * (height * 0.8) + height * 0.1))
            
            cv2.line(img, (int(i*dx), y1), (int((i+1)*dx), y2), (0, 255, 100), 2)
            
        return img