# corazon/analisis_fisiologico.py
import numpy as np
import cv2
import math
import time
from collections import deque

# =================================================================
# 🛠️ SECCIÓN EDITABLE: LÓGICA DE MEDICIÓN DE ANCHO
# =================================================================
def calcular_ancho_instante(mask, p1, p2, angulo_deg, num_rebanadas=10):
    """
    Esta es la parte que más editaremos. 
    Mide el ancho promedio del tejido a lo largo del eje.
    """
    if mask is None or np.sum(mask) == 0:
        return 0
    
    anchos = []
    # Generamos puntos a lo largo del eje longitudinal
    for i in range(1, num_rebanadas):
        # Punto de interpolación en el eje (de 10% a 90% del largo)
        t = i / num_rebanadas
        cx = int(p1[0] + (p2[0] - p1[0]) * t)
        cy = int(p1[1] + (p2[1] - p1[1]) * t)
        
        # Angulo perpendicular para medir el ancho
        rad_perp = math.radians(angulo_deg + 90)
        
        # Trazamos una línea perpendicular "infinita" (ancho de búsqueda)
        # Usamos 200px a cada lado como margen de seguridad
        dist_busqueda = 200 
        x1 = int(cx - dist_busqueda * math.cos(rad_perp))
        y1 = int(cy - dist_busqueda * math.sin(rad_perp))
        x2 = int(cx + dist_busqueda * math.cos(rad_perp))
        y2 = int(cy + dist_busqueda * math.sin(rad_perp))
        
        # Creamos una micro-máscara para esta línea
        line_mask = np.zeros_like(mask)
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)
        
        # Intersección de la línea con el tejido real
        interseccion = cv2.bitwise_and(mask, line_mask)
        puntos_blancos = np.where(interseccion > 0)
        
        if len(puntos_blancos[0]) > 1:
            # Distancia entre el primer y último píxel blanco en esa línea
            d = math.sqrt((puntos_blancos[1][0] - puntos_blancos[1][-1])**2 + 
                          (puntos_blancos[0][0] - puntos_blancos[0][-1])**2)
            anchos.append(d)
            
    return np.mean(anchos) if anchos else 0

# =================================================================
# 🧠 CLASE PROCESADORA (GESTIÓN DE SEÑALES)
# =================================================================
class AnalizadorFisiologico:
    def __init__(self):
        # Buffers de datos
        self.buffer_anchos = deque(maxlen=200) # Para el gráfico
        self.buffer_tiempos = deque(maxlen=200)
        
        # Memoria de picos (Sístole/Diástole)
        self.picos = []
        self.valles = []
        self.tiempos_picos = []
        
        # Resultados finales
        self.bpm_3 = 0
        self.bpm_5 = 0
        self.bpm_10 = 0
        self.variacion_bpm = 0
        self.ancho_max_promedio = 0
        self.ancho_min_promedio = 0

    def procesar_frame(self, mask, p1, p2, angulo):
        t_actual = time.time()
        ancho = calcular_ancho_instante(mask, p1, p2, angulo)
        
        self.buffer_anchos.append(ancho)
        self.buffer_tiempos.append(t_actual)
        
        self._detectar_eventos()
        self._calcular_metricas()
        
        return ancho

    def _detectar_eventos(self):
        """Detecta máximos y mínimos locales (latidos)"""
        if len(self.buffer_anchos) < 5: return
        
        vals = list(self.buffer_anchos)
        # Un pico es mayor que sus vecinos (ventana de 5)
        if vals[-3] > vals[-4] and vals[-3] > vals[-2] and vals[-3] > np.mean(vals) * 1.05:
            if not self.picos or (time.time() - self.tiempos_picos[-1] > 0.3): # Debounce 0.3s
                self.picos.append(vals[-3])
                self.tiempos_picos.append(self.buffer_tiempos[-3])
        
        # Un valle es menor que sus vecinos
        if vals[-3] < vals[-4] and vals[-3] < vals[-2] and vals[-3] < np.mean(vals) * 0.95:
            self.valles.append(vals[-3])

    def _calcular_metricas(self):
        # 1. Promedios de Sístole/Diástole
        if self.picos: self.ancho_max_promedio = np.mean(self.picos[-10:])
        if self.valles: self.ancho_min_promedio = np.mean(self.valles[-10:])
        
        # 2. Frecuencias (BPM)
        if len(self.tiempos_picos) >= 2:
            deltas = np.diff(self.tiempos_picos) # Diferencia en segundos entre latidos
            bpms = 60.0 / deltas
            
            if len(bpms) >= 3: self.bpm_3 = np.mean(bpms[-3:])
            if len(bpms) >= 5: self.bpm_5 = np.mean(bpms[-5:])
            if len(bpms) >= 10: 
                self.bpm_10 = np.mean(bpms[-10:])
                # Variación (Desviación estándar de los últimos 10)
                self.variacion_bpm = np.std(bpms[-10:])

    def generar_cardiograma(self, width=400, height=300):
        """Crea el dibujo del osciloscopio para la Pantalla 2"""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        if len(self.buffer_anchos) < 2: return img
        
        # Dibujar rejilla de fondo
        for i in range(0, height, 50): cv2.line(img, (0, i), (width, i), (30, 30, 30), 1)
        
        # Escalar datos al alto del widget
        max_v = max(self.buffer_anchos) if max(self.buffer_anchos) > 0 else 1
        min_v = min(self.buffer_anchos)
        rango = (max_v - min_v) if (max_v - min_v) > 0 else 1
        
        puntos = []
        for i, val in enumerate(self.buffer_anchos):
            x = int(i * (width / 200))
            y = int(height - ((val - min_v) / rango * (height * 0.8) + height * 0.1))
            puntos.append((x, y))
        
        # Dibujar la línea del latido (Verde neón)
        for i in range(len(puntos)-1):
            cv2.line(img, puntos[i], puntos[i+1], (0, 255, 100), 2)
            
        return img