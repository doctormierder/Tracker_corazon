import numpy as np
import cv2
import math
import time
from collections import deque

def calcular_ancho_instante(mask, p1, p2, angulo_deg_ignorado, num_rebanadas=10):
    """
    Mide el ancho del tejido utilizando Álgebra Lineal pura y Raycasting.
    Ignoramos el ángulo externo para evitar desfasajes trigonométricos.
    """
    if mask is None or np.sum(mask) == 0:
        return 0, None
    
    h, w = mask.shape
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    largo_eje = math.hypot(dx, dy)
    
    if largo_eje == 0:
        return 0, None
        
    # 1. Vector unitario longitudinal y Perpendicular Estricta
    ux, uy = dx / largo_eje, dy / largo_eje
    px, py = -uy, ux 
    
    # Rango máximo de búsqueda (mitad de la pantalla por seguridad)
    dist_max = int(math.hypot(w, h) / 2)
    
    anchos = []
    puntos_linea_debug = None
    
    # 2. Análisis por rebanadas
    for i in range(1, num_rebanadas):
        t = i / num_rebanadas
        cx = int(p1[0] + dx * t)
        cy = int(p1[1] + dy * t)
        
        # Generar rayo de coordenadas (desde -dist_max hasta +dist_max)
        r = np.arange(-dist_max, dist_max + 1)
        x_coords = np.clip(np.round(cx + r * px).astype(int), 0, w - 1)
        y_coords = np.clip(np.round(cy + r * py).astype(int), 0, h - 1)
        
        # Extraer los valores de los píxeles directamente de la matriz (Ultra rápido)
        perfil = mask[y_coords, x_coords]
        idx_centro_matematico = dist_max 
        
        # Buscar la masa blanca más cercana al centro matemático
        blancos = np.where(perfil > 0)[0]
        if len(blancos) == 0:
            continue # Rebanada vacía (fuera del tejido)
            
        idx_ancla = blancos[np.argmin(np.abs(blancos - idx_centro_matematico))]
        
        # 3. Caminar hacia la derecha hasta tocar fondo (negro)
        perfil_der = perfil[idx_ancla:]
        ceros_der = np.where(perfil_der == 0)[0]
        d_der = ceros_der[0] if len(ceros_der) > 0 else len(perfil_der) - 1
        
        # 4. Caminar hacia la izquierda hasta tocar fondo
        perfil_izq = perfil[:idx_ancla+1][::-1]
        ceros_izq = np.where(perfil_izq == 0)[0]
        d_izq = ceros_izq[0] if len(ceros_izq) > 0 else len(perfil_izq) - 1
        
        ancho_rebanada = d_der + d_izq
        anchos.append(ancho_rebanada)
        
        # 5. Guardar Puntos para Debuging (Asegura captura si la del medio falla)
        if i == num_rebanadas // 2 or puntos_linea_debug is None:
            r_der = r[idx_ancla + d_der]
            r_izq = r[idx_ancla - d_izq]
            pt_der = (int(cx + r_der * px), int(cy + r_der * py))
            pt_izq = (int(cx + r_izq * px), int(cy + r_izq * py))
            puntos_linea_debug = (pt_izq, pt_der)
            
    # Promedio de anchos limpios
    return np.mean(anchos) if anchos else 0.0, puntos_linea_debug

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

    def procesar_frame(self, mask, p1, p2, angulo):
        t_actual = time.time()
        ancho, pts_debug = calcular_ancho_instante(mask, p1, p2, angulo)
        
        if ancho > 0:
            self.buffer_anchos.append(ancho)
            self.buffer_tiempos.append(t_actual)
            self.sweep_y[self.cursor_x] = ancho
            self.cursor_x = (self.cursor_x + 1) % self.puntos_grafico
            self._detectar_eventos()
            self._actualizar_atributos_directos()
            
        return ancho, pts_debug

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