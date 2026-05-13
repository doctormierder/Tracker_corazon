import numpy as np
import cv2
import math
import time
from collections import deque

class AnalizadorFisiologico:
    def __init__(self):
        # Buffers temporales para frecuencia cardiaca (los trataremos en la Fase 2)
        self.buffer_anchos = deque(maxlen=300)
        self.buffer_tiempos = deque(maxlen=300)
        self.picos = []; self.valles = []; self.tiempos_picos = []; self.bpms_historial = []
        self.ancho_max_promedio = 0.0; self.ancho_min_promedio = 0.0
        self.bpm_media = 0.0; self.variacion_bpm = 0.0
        self.puntos_grafico = 200
        self.sweep_y = np.zeros(self.puntos_grafico)
        self.cursor_x = 0
        
        # =======================================================
        # --- PARÁMETROS BALÍSTICOS (EL FRENTE 3) ---
        # =======================================================
        self.ancho_historico = None         # Memoria del grosor anterior
        self.offset_centro_historico = 0.0  # Desplazamiento del centro real vs el matemático
        
        self.num_rayos = 51               # Número impar para tener un rayo central perfecto
        self.margen_vision = 0.75         # 60% desde el centro a cada lado (Corta las 'alitas' exteriores)
        self.max_velocidad = 0.50         # Tolerancia de cambio brusco (Máx 10% de crecimiento por frame)
        self.inercia = 0.80               # Fricción de la línea (70% pasado, 30% presente)

    def procesar_frame(self, mask, p1, p2, angulo_deg_ignorado):
        t_actual = time.time()
        
        if mask is None or np.sum(mask) == 0:
            return 0.0, None
            
        h, w = mask.shape
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        largo_eje = math.hypot(dx, dy)
        
        if largo_eje == 0:
            return 0.0, None
            
        # Vector transversal estricto
        px, py = -dy / largo_eje, dx / largo_eje 
        dist_max = int(math.hypot(w, h) / 2)
        
        # 1. DISPARAR MULTIRRAYOS (Solo en el 20% central del corazón, la zona más confiable)
        # Ignoramos los polos (top y bottom) donde la geometría colapsa fácilmente.
        t_vals = np.linspace(0.30, 0.70, self.num_rayos) 
        
        anchos_candidatos = []
        centros_candidatos = []
        
        # Matriz temporal de coordenadas para el rayo
        r = np.arange(-dist_max, dist_max + 1)
        
        for t in t_vals:
            cx = p1[0] + dx * t
            cy = p1[1] + dy * t
            
            x_coords = np.clip(np.round(cx + r * px).astype(int), 0, w - 1)
            y_coords = np.clip(np.round(cy + r * py).astype(int), 0, h - 1)
            perfil = mask[y_coords, x_coords]
            
            # 2. CEGUERA SELECTIVA (Ignorar Ruido Externo y Alitas)
            if self.ancho_historico is not None:
                # Calculamos dónde debería estar el tejido basándonos en la historia
                lim_izq = int(dist_max + self.offset_centro_historico - (self.ancho_historico * self.margen_vision))
                lim_der = int(dist_max + self.offset_centro_historico + (self.ancho_historico * self.margen_vision))
                
                lim_izq = max(0, lim_izq)
                lim_der = min(len(perfil) - 1, lim_der)
                
                # Apagamos todo lo que esté fuera de esta ventana blindada
                perfil_blindado = np.zeros_like(perfil)
                perfil_blindado[lim_izq:lim_der] = perfil[lim_izq:lim_der]
                perfil = perfil_blindado
                
            blancos = np.where(perfil > 0)[0]
            
            # Si el rayo cayó en un hueco negro absoluto, lo descartamos
            if len(blancos) < 2:
                continue 
                
            idx_izq = blancos[0]
            idx_der = blancos[-1]
            ancho_rayo = float(idx_der - idx_izq)
            
            # Descartar micro-ruidos flotantes dentro de la ventana
            if ancho_rayo < 10: 
                continue
                
            offset_centro_rayo = (idx_izq + ancho_rayo / 2.0) - dist_max
            
            anchos_candidatos.append(ancho_rayo)
            centros_candidatos.append(offset_centro_rayo)

        # 3. MANEJO DE CRISIS (Sístole profunda / Apagón)
        if not anchos_candidatos:
            # Si TODOS los 31 rayos fallaron (el corazón se oscureció entero), aplicamos inercia total.
            # Congelamos la línea usando la última memoria conocida.
            if self.ancho_historico is not None:
                # Retornamos el estado congelado para que la interfaz no colapse a 0
                return self._generar_salida_visual(p1, dx, dy, px, py, 
                                                   self.ancho_historico, self.offset_centro_historico, t_actual)
            return 0.0, None

        # 4. VOTACIÓN ESTADÍSTICA (La Mediana destruye las anomalías)
        # Si 5 rayos cayeron en huecos y 20 tocaron tejido sólido, la mediana te da el tejido sólido.
        ancho_mediana = float(np.median(anchos_candidatos))
        centro_mediana = float(np.median(centros_candidatos))
        
        # 5. CONTROL BALÍSTICO Y ABRAZADERAS (Límites de Velocidad Física)
        if self.ancho_historico is not None:
            variacion_max = self.ancho_historico * self.max_velocidad
            
            # El grosor no puede saltar libremente, lo frenamos en el límite de velocidad
            ancho_mediana = np.clip(ancho_mediana, 
                                    self.ancho_historico - variacion_max, 
                                    self.ancho_historico + variacion_max)
                                    
            centro_mediana = np.clip(centro_mediana, 
                                     self.offset_centro_historico - variacion_max, 
                                     self.offset_centro_historico + variacion_max)
                                     
            # Aplicamos Inercia (fricción) para suavizar la vibración natural de los píxeles
            ancho_final = (self.inercia * self.ancho_historico) + ((1.0 - self.inercia) * ancho_mediana)
            centro_final = (self.inercia * self.offset_centro_historico) + ((1.0 - self.inercia) * centro_mediana)
        else:
            ancho_final = ancho_mediana
            centro_final = centro_mediana

        # Actualizamos la memoria
        self.ancho_historico = ancho_final
        self.offset_centro_historico = centro_final
        
        return self._generar_salida_visual(p1, dx, dy, px, py, ancho_final, centro_final, t_actual)

    def _generar_salida_visual(self, p1, dx, dy, px, py, ancho, offset, t_actual):
        """Genera los puntos (pt1, pt2) de la línea magenta central y guarda el registro."""
        # Calculamos la línea exactamente en el centro matemático del corazón (t = 0.5)
        cx = p1[0] + dx * 0.5 + offset * px
        cy = p1[1] + dy * 0.5 + offset * py
        
        pt_izq = (int(cx - (ancho / 2.0) * px), int(cy - (ancho / 2.0) * py))
        pt_der = (int(cx + (ancho / 2.0) * px), int(cy + (ancho / 2.0) * py))
        
        # Guardado en buffer para Frecuencia (Fase 2)
        if ancho > 0:
            self.buffer_anchos.append(ancho)
            self.buffer_tiempos.append(t_actual)
            self.sweep_y[self.cursor_x] = ancho
            self.cursor_x = (self.cursor_x + 1) % self.puntos_grafico
            # Placeholder de eventos de frecuencia (no tocar por ahora)
            # self._detectar_eventos()
            # self._actualizar_atributos_directos()
            
        return ancho, (pt_izq, pt_der)
        
    # [Mantén aquí intactas tus funciones: _detectar_eventos, _actualizar_atributos_directos, obtener_estadisticas, generar_cardiograma]

    def _detectar_eventos(self):
        if len(self.buffer_anchos) < 10: return
        vals = list(self.buffer_anchos)
        
        if vals[-3] > vals[-5] and vals[-3] > vals[-1] and vals[-3] > np.mean(vals) * 1.02:
            t_ahora = time.time()
            if not self.tiempos_picos or (t_ahora - self.tiempos_picos[-1] > 0.25):
                self.picos.append(vals[-3])
                self.tiempos_picos.append(t_ahora)
                if len(self.tiempos_picos) > 1:
                    self.bpms_historial.append(60.0 / (self.tiempos_picos[-1] - self.tiempos_picos[-2]))
        
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