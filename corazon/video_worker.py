# corazon/video_worker.py
import cv2
import time
import math
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage
from collections import deque

from .buscando_eje import viaje_en_el_tiempo_definitivo
from .core.config import TrackerConfig
from .core.kit_procesamiento import SelectorKits

class VideoWorker(QObject):
    """
    Trabajador principal encargado de:
    1. Geometría Hexagonal anclada a la pantalla (+20% largo, ancho 50%).
    2. Detección de Extremos en la franja central (1%).
    3. Tolerancia Dinámica (Mediana = 50).
    4. Tanteo IoU Altamente Optimizado (ROI).
    """
    
    frame_ready = pyqtSignal(QImage, QImage, QImage, QImage) 
    stats_ready = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, video_path, config: TrackerConfig):
        super().__init__()
        self.video_path = video_path
        self.config = config
        self._running = True
    
        # Sistema de Fricción (Jerarquía de Ejes)
        self.historial_cx = deque(maxlen=10)
        self.historial_cy = deque(maxlen=10)
        self.historial_rojo_x = deque(maxlen=30)
        self.historial_rojo_y = deque(maxlen=30)

        self.hex_solido_maestro = None
        self.roi_hex_coords = None
        self.extremos_corazon = [] 
        self._last_blur_val = -1
        self._hex_suave_cache = None

    def stop(self):
        self._running = False

    @pyqtSlot()
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished.emit()
            return

        # =================================================================
        # FASE 0: GEOMETRÍA ANCLADA A LOS BORDES DE LA PANTALLA
        # =================================================================
        # Busca donde llamas a la función:
        res = viaje_en_el_tiempo_definitivo(cap)
        # Desempaquetamos los 8 valores (ahora incluimos p1 y p2)
        (angulo_cyan, mascara_tiempo, fijo_x, fijo_y, 
         self.eje_mayor, self.eje_menor, p1_calib, p2_calib) = res

        if mascara_tiempo is None:
            self.finished.emit()
            return

        h_f, w_f = mascara_tiempo.shape
        rad_cyan = math.radians(angulo_cyan)
        
        v_long_x, v_long_y = math.sin(rad_cyan), -math.cos(rad_cyan)
        v_trans_x, v_trans_y = math.cos(rad_cyan), math.sin(rad_cyan)
        
        # 1. Trazar el eje y ver dónde choca exactamente con los bordes de la cámara
        pt1 = (int(fijo_x + 5000 * v_long_x), int(fijo_y + 5000 * v_long_y))
        pt2 = (int(fijo_x - 5000 * v_long_x), int(fijo_y - 5000 * v_long_y))
        
        rect_img = (0, 0, w_f, h_f)
        ret_clip, p_a, p_b = cv2.clipLine(rect_img, pt1, pt2)
        
        if ret_clip:
            # El centro del segmento es el centro de nuestro hexágono
            centro_vis_x = (p_a[0] + p_b[0]) / 2.0
            centro_vis_y = (p_a[1] + p_b[1]) / 2.0
            
            # El largo visible es la distancia de borde a borde
            largo_pantalla = math.hypot(p_a[0] - p_b[0], p_a[1] - p_b[1])
            
            # --- GEOMETRÍA: +20% de largo, 50% de ancho ---
            L_h = largo_pantalla * 1.40 
            W_h = L_h * 0.50
            
            pts_hex = np.array([
                [centro_vis_x + v_long_x * (L_h/2), centro_vis_y + v_long_y * (L_h/2)],
                [centro_vis_x + v_long_x * (L_h/4) + v_trans_x * (W_h/2), centro_vis_y + v_long_y * (L_h/4) + v_trans_y * (W_h/2)],
                [centro_vis_x - v_long_x * (L_h/4) + v_trans_x * (W_h/2), centro_vis_y - v_long_y * (L_h/4) + v_trans_y * (W_h/2)],
                [centro_vis_x - v_long_x * (L_h/2), centro_vis_y - v_long_y * (L_h/2)],
                [centro_vis_x - v_long_x * (L_h/4) - v_trans_x * (W_h/2), centro_vis_y - v_long_y * (L_h/4) - v_trans_y * (W_h/2)],
                [centro_vis_x + v_long_x * (L_h/4) - v_trans_x * (W_h/2), centro_vis_y + v_long_y * (L_h/4) - v_trans_y * (W_h/2)]
            ], dtype=np.int32)
            
            self.hex_solido_maestro = np.zeros((h_f, w_f), dtype=np.uint8)
            cv2.fillPoly(self.hex_solido_maestro, [pts_hex], 255)

            # Precalculamos el Bounding Box del Hexágono para cálculos matemáticos veloces después
            y_idx, x_idx = np.where(self.hex_solido_maestro > 0)
            if len(y_idx) > 0:
                self.roi_hex_coords = (np.min(y_idx), np.max(y_idx)+1, np.min(x_idx), np.max(x_idx)+1)
            else:
                self.roi_hex_coords = (0, h_f, 0, w_f)

            # --- DETECCIÓN DE EXTREMOS (FRANJA DEL 1%) ---
            grosor_1_porciento = max(1, int(L_h * 0.01))
            franja_mask = np.zeros((h_f, w_f), dtype=np.uint8)
            cv2.line(franja_mask, p_a, p_b, 255, grosor_1_porciento)
            
            # Intersección entre la Exposición (el corazón real) y el eje del 1%
            inter_eje_corazon = cv2.bitwise_and(mascara_tiempo, franja_mask)
            cnts_eje, _ = cv2.findContours(inter_eje_corazon, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Limpiamos basura de 1 píxel
            cnts_validos = [c for c in cnts_eje if cv2.contourArea(c) > 1.0]
            
            if len(cnts_validos) >= 2:
                centroides = []
                for c in cnts_validos:
                    M_c = cv2.moments(c)
                    if M_c["m00"] != 0:
                        centroides.append((int(M_c["m10"]/M_c["m00"]), int(M_c["m01"]/M_c["m00"])))
                    else:
                        centroides.append((c[0][0][0], c[0][0][1]))
                
                # Buscamos la distancia máxima entre 2 grupos para hallar las "puntas"
                max_d = -1
                mejor_par = []
                for i in range(len(centroides)):
                    for j in range(i+1, len(centroides)):
                        d = math.hypot(centroides[i][0]-centroides[j][0], centroides[i][1]-centroides[j][1])
                        if d > max_d:
                            max_d = d
                            mejor_par = [centroides[i], centroides[j]]
                
                self.extremos_corazon = mejor_par

        else:
            self.hex_solido_maestro = np.zeros((h_f, w_f), dtype=np.uint8)
            self.roi_hex_coords = (0, h_f, 0, w_f)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        procesador = SelectorKits.obtener_kit(self.config.kit_procesamiento)
        lx, ly = 3000 * v_long_x, 3000 * v_long_y

        # =================================================================
        # BUCLE DE TRACKING EN TIEMPO REAL
        # =================================================================
        while self._running:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret: 
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # --- CACHÉ DE DESENFOQUE (Alto rendimiento visual) ---
            if self._last_blur_val != self.config.desenfoque:
                # El slider ahora rige el % sobre el largo del hexágono total
                blur_ratio = max(1, int(self.config.desenfoque)) / 100.0
                k_size = int(L_h * blur_ratio)
                if k_size % 2 == 0: k_size += 1
                k_size = max(3, k_size)
                
                self._hex_suave_cache = cv2.GaussianBlur(self.hex_solido_maestro, (k_size, k_size), 0)
                self._last_blur_val = self.config.desenfoque

            hex_suave = self._hex_suave_cache

            # --- PROCESAMIENTO ---
            mask_rastreo, img_tratada, _ = procesador.procesar(
                frame, self.config, self.hex_solido_maestro, hex_suave
            )

            # --- UMBRALIZACIÓN ADAPTATIVA POR FRANJAS PERPENDICULARES ---
            # 1. Extracción de la zona de interés
            y1_h, y2_h, x1_h, x2_h = self.roi_hex_coords
            roi_tratada = img_tratada[y1_h:y2_h, x1_h:x2_h]
            roi_mask = self.hex_solido_maestro[y1_h:y2_h, x1_h:x2_h]
            
            # 2. Crear un sistema de coordenadas espaciales local
            Y_roi, X_roi = np.indices(roi_tratada.shape)
            X_global = X_roi + x1_h
            Y_global = Y_roi + y1_h
            
            # 3. Proyectar cada píxel sobre el eje longitudinal del corazón
            # Esto asigna un valor numérico a cada píxel indicando a qué "altura" del tubo está
            proyeccion_long = (X_global * v_long_x) + (Y_global * v_long_y)
            
            pixeles_validos = roi_mask > 0
            mask_medicion = np.zeros_like(img_tratada)

            if np.any(pixeles_validos):
                proj_validas = proyeccion_long[pixeles_validos]
                min_p, max_p = np.min(proj_validas), np.max(proj_validas)
                
                # Vamos a cortar el corazón en 8 "rodajas" (franjas)
                num_franjas = 16
                rango_franja = (max_p - min_p) / num_franjas
                
                # Creamos un "Mapa Topográfico de Umbrales"
                mapa_umbrales = np.zeros_like(roi_tratada, dtype=np.float32)
                val_slider = self.config.tolerancia

                for i in range(num_franjas):
                    lim_inf = min_p + i * rango_franja
                    lim_sup = lim_inf + rango_franja if i < num_franjas - 1 else max_p + 1
                    
                    # Aislamos solo los píxeles de ESTA rodaja
                    mask_franja = pixeles_validos & (proyeccion_long >= lim_inf) & (proyeccion_long < lim_sup)
                    pixeles_franja = roi_tratada[mask_franja]
                    
                    if len(pixeles_franja) > 10:
                        luz_fondo = np.percentile(pixeles_franja, 50)
                        max_absoluto = np.percentile(pixeles_franja, 98) # Ignoramos ruido quemado
                    else:
                        luz_fondo, max_absoluto = 127, 255
                        
                    # Calculamos el umbral exclusivo para esta rodaja
                    if val_slider >= 50:
                        porcentaje = (val_slider - 50) / 50.0
                        umbral_franja = luz_fondo + ((max_absoluto - luz_fondo) * porcentaje)
                    else:
                        porcentaje = (50 - val_slider) / 49.0
                        distancia = max_absoluto - luz_fondo
                        umbral_franja = luz_fondo - (distancia * porcentaje)
                        
                    # Asignamos este umbral al mapa topográfico
                    mapa_umbrales[mask_franja] = np.clip(umbral_franja, 0, 255)

                # 4. Difuminado del Mapa (Crucial)
                # Difuminamos el mapa de umbrales para que no haya cortes duros o "escalones" entre las franjas
                mapa_suave = cv2.GaussianBlur(mapa_umbrales, (19, 19), 0)
                
                # 5. Binarización Final (Cada píxel se compara con su propio umbral personalizado)
                mask_medicion_roi = (roi_tratada >= mapa_suave).astype(np.uint8) * 255
                mask_medicion_roi = cv2.bitwise_and(mask_medicion_roi, roi_mask)
                
                mask_medicion[y1_h:y2_h, x1_h:x2_h] = mask_medicion_roi

            # --- TANTEO DE CONFIANZA (OPTIMIZADO CON BOUNDING BOX) ---
            contornos, _ = cv2.findContours(mask_rastreo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mejor_confianza = 0.0

            if contornos:
                mayor = max(contornos, key=cv2.contourArea)
                M = cv2.moments(mayor)
                if M["m00"] != 0:
                    base_cx, base_cy = M["m10"]/M["m00"], M["m01"]/M["m00"]

                    tejido_puro = np.zeros_like(mask_rastreo)
                    cv2.drawContours(tejido_puro, [mayor], -1, 255, thickness=cv2.FILLED)

                    ganador_cx, ganador_cy = base_cx, base_cy
                    desplazamientos = [-15, -5, 0, 5, 15]
                    
                    nx, ny = -v_long_y, v_long_x 

                    # Aceleración Extrema: Calculamos ROI basado en el corazón para saltarnos el fondo 4K
                    bx, by, bw, bh = cv2.boundingRect(mayor)
                    pad = max(30, int(w_f * 0.05)) # Margen para permitir el movimiento lateral
                    y1 = max(0, by - pad)
                    y2 = min(h_f, by + bh + pad)
                    x1 = max(0, bx - pad)
                    x2 = min(w_f, bx + bw + pad)
                    
                    tejido_puro_roi = tejido_puro[y1:y2, x1:x2].astype(np.float32) / 255.0
                    X_roi, Y_roi = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))

                    for offset in desplazamientos:
                        test_cx = base_cx + v_trans_x * offset
                        test_cy = base_cy + v_trans_y * offset

                        cos2a, sin2a = math.cos(2 * rad_cyan), math.sin(2 * rad_cyan)
                        tx = test_cx - test_cx * cos2a - test_cy * sin2a
                        ty = test_cy - test_cx * sin2a + test_cy * cos2a
                        
                        matriz_espejo = np.float32([[cos2a, sin2a, tx], [sin2a, -cos2a, ty]])
                        tejido_reflejado = cv2.warpAffine(tejido_puro, matriz_espejo, (w_f, h_f))

                        # Solo comparamos en el recuadro del corazón (ROI)
                        tejido_ref_roi = tejido_reflejado[y1:y2, x1:x2].astype(np.float32) / 255.0

                        # AND y OR matemáticos ultra rápidos
                        roi_inter = tejido_puro_roi * tejido_ref_roi
                        roi_union = np.maximum(tejido_puro_roi, tejido_ref_roi)
                        
                        # Distancia de Gauss al vuelo
                        dist_map_roi = np.abs((X_roi - test_cx)*nx + (Y_roi - test_cy)*ny)
                        mapa_pesos_roi = np.exp(-0.5 * (dist_map_roi / 10.0)**4)
                        
                        peso_inter = np.sum(roi_inter * mapa_pesos_roi)
                        peso_union = np.sum(roi_union * mapa_pesos_roi)
                        
                        if peso_union > 0:
                            factor_castigo = math.exp(-0.5 * (offset / 15.0)**2)
                            confianza_actual = ((peso_inter / peso_union) * 100.0) * factor_castigo
                            if confianza_actual > mejor_confianza:
                                mejor_confianza = confianza_actual
                                ganador_cx, ganador_cy = test_cx, test_cy

                    # --- JERARQUÍA DE EJES ---
                    self.historial_cx.append(ganador_cx); self.historial_cy.append(ganador_cy)
                    cx_verde = np.median(self.historial_cx)
                    cy_verde = np.median(self.historial_cy)

                    media_x = (2.0 * fijo_x + cx_verde) / 3.0
                    media_y = (2.0 * fijo_y + cy_verde) / 3.0
                    self.historial_rojo_x.append(media_x); self.historial_rojo_y.append(media_y)
                    cx_rojo = int(np.median(self.historial_rojo_x))
                    cy_rojo = int(np.median(self.historial_rojo_y))

                    if self.config.mostrar_rosa:
                        cv2.line(frame, (int(cx_verde - lx), int(cy_verde - ly)), 
                                        (int(cx_verde + lx), int(cy_verde + ly)), (0, 255, 0), 1)

                    if self.config.mostrar_elipse: 
                        cv2.line(frame, (int(cx_rojo - lx), int(cy_rojo - ly)), 
                                        (int(cx_rojo + lx), int(cy_rojo + ly)), (0, 0, 255), 3)

            # Dibujo de Extremos Superiores/Inferiores
            cv2.circle(frame, p1_calib, max(6, int(w_f*0.007)), (0, 255, 255), -1)
            cv2.circle(frame, p2_calib, max(6, int(w_f*0.007)), (0, 255, 255), -1)
            
            # Dibujamos una línea fina entre ellos para ver la distancia evaluada
            cv2.line(frame, p1_calib, p2_calib, (0, 255, 255), 1)

            self.stats_ready.emit({"confianza": round(float(mejor_confianza), 1)})

            # ========================================================
            # EMISIÓN A PANTALLAS
            # ========================================================
            rgb_f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_f = QImage(rgb_f.data, w_f, h_f, 3 * w_f, QImage.Format.Format_RGB888).copy()

            rgb_m1 = cv2.cvtColor(mask_rastreo, cv2.COLOR_GRAY2RGB)
            qt_m1 = QImage(rgb_m1.data, w_f, h_f, 3 * w_f, QImage.Format.Format_RGB888).copy()

            rgb_tr = cv2.cvtColor(img_tratada, cv2.COLOR_GRAY2RGB)
            qt_tr = QImage(rgb_tr.data, w_f, h_f, 3 * w_f, QImage.Format.Format_RGB888).copy()

            rgb_m2 = cv2.cvtColor(mask_medicion, cv2.COLOR_GRAY2RGB)
            qt_m2 = QImage(rgb_m2.data, w_f, h_f, 3 * w_f, QImage.Format.Format_RGB888).copy()

            self.frame_ready.emit(qt_f, qt_m1, qt_tr, qt_m2)

            elapsed = time.time() - start_time
            time.sleep(max(0, (1.0/self.config.fps_objetivo) - elapsed))

        cap.release()
        self.finished.emit()