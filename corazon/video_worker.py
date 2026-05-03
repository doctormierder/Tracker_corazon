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
    # Emitimos 3 imágenes: Original, Máscara 1 (Rastreo), Máscara 2 (Medición)
    frame_ready = pyqtSignal(QImage, QImage, QImage, QImage) 
    stats_ready = pyqtSignal(dict)
    exposicion_lista = pyqtSignal(np.ndarray)
    finished = pyqtSignal()

    def __init__(self, video_path, config: TrackerConfig):
        super().__init__()
        self.video_path = video_path
        self.config = config
        self._running = True
    
        # SISTEMA DE FRICCIÓN
        self.historial_cx = deque(maxlen=10)
        self.historial_cy = deque(maxlen=10)
        self.historial_rojo_x = deque(maxlen=30)
        self.historial_rojo_y = deque(maxlen=30)

    @pyqtSlot()
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished.emit()
            return

        # --- FASE 1: MÁQUINA DEL TIEMPO ---
        datos_eje = viaje_en_el_tiempo_definitivo(cap)
        ang_cyan, img_exposicion, fix_x, fix_y, eje_mayor, eje_menor_base = datos_eje
        
        if img_exposicion is not None:
            self.exposicion_lista.emit(img_exposicion)
        
        rad_cyan = math.radians(ang_cyan)
        procesador = SelectorKits.obtener_kit(self.config.kit_procesamiento)

        fps = cap.get(cv2.CAP_PROP_FPS)
        intervalo = 1 / (fps if fps > 0 else 30)

        # --- FASE 2: BUCLE DE TRACKING ---
        while self._running:
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # 1. GROSORES RELATIVOS
            h_f, w_f = frame.shape[:2]
            t_fino = max(1, int(w_f * 0.002))
            t_medio = max(2, int(w_f * 0.004))
            t_grueso = max(3, int(w_f * 0.006))

            # ========================================================
            # 2. PLANOS GEOMÉTRICOS (El Worker solo dibuja dónde buscar)
            # ========================================================
            v_long_x, v_long_y = math.sin(rad_cyan), -math.cos(rad_cyan)
            v_trans_x, v_trans_y = math.cos(rad_cyan), math.sin(rad_cyan)

            L = eje_mayor
            
            # --- MODIFICACIÓN AQUÍ ---
            # Antes: W = min(eje_menor_base, L / 2.0)
            # Ahora: Hacemos el ancho base un 50% más grande, 
            # y subimos el límite para que pueda ser hasta el 80% del largo.
            W = 2.8*min(eje_menor_base * 1.3, L * 0.8)

            # Polígono Sólido (Límite exterior)
            hex_pts = np.array([
                [fix_x + L * v_long_x, fix_y + L * v_long_y],
                [fix_x + (L/2) * v_long_x + W * v_trans_x, fix_y + (L/2) * v_long_y + W * v_trans_y],
                [fix_x - (L/2) * v_long_x + W * v_trans_x, fix_y - (L/2) * v_long_y + W * v_trans_y],
                [fix_x - L * v_long_x, fix_y - L * v_long_y],
                [fix_x - (L/2) * v_long_x - W * v_trans_x, fix_y - (L/2) * v_long_y - W * v_trans_y],
                [fix_x + (L/2) * v_long_x - W * v_trans_x, fix_y + (L/2) * v_long_y - W * v_trans_y]
            ], np.int32)
            hexagon_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(hexagon_mask, [hex_pts], 255)
            
            # Polígono Estrecho (Zona de medición)
            W_estrecho = W * 0.60
            hex_pts_suave = np.array([
                [fix_x + L * v_long_x, fix_y + L * v_long_y],
                [fix_x + (L/2) * v_long_x + W_estrecho * v_trans_x, fix_y + (L/2) * v_long_y + W_estrecho * v_trans_y],
                [fix_x - (L/2) * v_long_x + W_estrecho * v_trans_x, fix_y - (L/2) * v_long_y + W_estrecho * v_trans_y],
                [fix_x - L * v_long_x, fix_y - L * v_long_y],
                [fix_x - (L/2) * v_long_x - W_estrecho * v_trans_x, fix_y - (L/2) * v_long_y - W_estrecho * v_trans_y],
                [fix_x + (L/2) * v_long_x - W_estrecho * v_trans_x, fix_y + (L/2) * v_long_y - W_estrecho * v_trans_y]
            ], np.int32)
            hex_mask_2 = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(hex_mask_2, [hex_pts_suave], 255)

            # ========================================================
            # 3. EL KIT HACE LA MAGIA (Delega todo el tratamiento)
            # ========================================================
            mask_filtrada, img_tratada, mascara_naranja = procesador.procesar(
                frame, self.config, hexagon_mask, hex_mask_2
            )

            # --- DIBUJOS Y ANÁLISIS ---

            if self.config.mostrar_elipse:
                cv2.polylines(frame, [hex_pts], True, (100, 100, 100), t_fino)

            dxc, dyc = 1000 * math.sin(rad_cyan), 1000 * math.cos(rad_cyan)
            if self.config.mostrar_cyan:
                cv2.line(frame, (int(fix_x - dxc), int(fix_y + dyc)), 
                                (int(fix_x + dxc), int(fix_y - dyc)), (255, 255, 0), t_medio)

            contornos, _ = cv2.findContours(mask_filtrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            mejor_confianza = 0.0

            if contornos and self.config.mostrar_rosa:
                mayor = max(contornos, key=cv2.contourArea)
                M = cv2.moments(mayor)
                if M["m00"] != 0:
                    base_cx = M["m10"]/M["m00"]
                    base_cy = M["m01"]/M["m00"]

                    # --- CORRECCIÓN AQUÍ: Usamos mask_filtrada como lienzo ---
                    tejido_puro = np.zeros_like(mask_filtrada)
                    cv2.drawContours(tejido_puro, [mayor], -1, 255, thickness=cv2.FILLED)
                    h_m, w_m = tejido_puro.shape

                    # Tanteo Multi-Eje (Rastreo)
                    perp_dx, perp_dy = math.cos(rad_cyan), -math.sin(rad_cyan)
                    ganador_cx, ganador_cy = base_cx, base_cy
                    desplazamientos = [-15, -5, 0, 5, 15]

                    for offset in desplazamientos:
                        test_cx = base_cx + perp_dx * offset
                        test_cy = base_cy + perp_dy * offset

                        line_mask = np.ones((h_m, w_m), dtype=np.uint8) * 255
                        cv2.line(line_mask, (int(test_cx - dxc), int(test_cy + dyc)), 
                                            (int(test_cx + dxc), int(test_cy - dyc)), 0, 1)
                        dist_map = cv2.distanceTransform(line_mask, cv2.DIST_L2, 3)
                        mapa_pesos = np.exp(-0.5 * (dist_map / 10.0)**4).astype(np.float32)

                        cos2a, sin2a = math.cos(2 * rad_cyan), math.sin(2 * rad_cyan)
                        tx = test_cx - test_cx * cos2a - test_cy * sin2a
                        ty = test_cy - test_cx * sin2a + test_cy * cos2a
                        
                        matriz_espejo = np.float32([[cos2a, sin2a, tx], [sin2a, -cos2a, ty]])
                        tejido_reflejado = cv2.warpAffine(tejido_puro, matriz_espejo, (w_m, h_m))

                        peso_inter = np.sum((cv2.bitwise_and(tejido_puro, tejido_reflejado).astype(np.float32) / 255.0) * mapa_pesos)
                        peso_union = np.sum((cv2.bitwise_or(tejido_puro, tejido_reflejado).astype(np.float32) / 255.0) * mapa_pesos)
                        
                        if peso_union > 0:
                            factor_castigo = math.exp(-0.5 * (offset / 15.0)**2)
                            confianza_actual = ((peso_inter / peso_union) * 100.0) * factor_castigo
                            if confianza_actual > mejor_confianza:
                                mejor_confianza = confianza_actual
                                ganador_cx, ganador_cy = test_cx, test_cy

                    self.historial_cx.append(ganador_cx)
                    self.historial_cy.append(ganador_cy)

                    media_ponderada_x = (2 * fix_x + ganador_cx) / 3.0
                    media_ponderada_y = (2 * fix_y + ganador_cy) / 3.0
                    
                    self.historial_rojo_x.append(media_ponderada_x)
                    self.historial_rojo_y.append(media_ponderada_y)
                    
                    cx_rojo = int(np.median(self.historial_rojo_x))
                    cy_rojo = int(np.median(self.historial_rojo_y))

                    # Eje Rojo Definitivo
                    cv2.line(frame, (int(cx_rojo - dxc), int(cy_rojo + dyc)), 
                                    (int(cx_rojo + dxc), int(cy_rojo - dyc)), (0, 0, 255), t_grueso)

                    # ESCÁNER TRANSVERSAL (Usando la Máscara 2 Suave)
                    deg_rot = math.degrees(rad_cyan)
                    M_rot = cv2.getRotationMatrix2D((cx_rojo, cy_rojo), deg_rot, 1.0)
                    tejido_upright = cv2.warpAffine(mascara_naranja, M_rot, (w_m, h_m))
                    
                    pixeles_por_fila = np.sum(tejido_upright == 255, axis=1)
                    
                    if np.max(pixeles_por_fila) > 0:
                        y_max_upright = np.argmax(pixeles_por_fila) 
                        ancho_maximo = pixeles_por_fila[y_max_upright]

                        M_inv = cv2.getRotationMatrix2D((cx_rojo, cy_rojo), -deg_rot, 1.0)
                        pto_centro_upright = np.array([[[cx_rojo, y_max_upright]]], dtype=np.float32)
                        pto_original = cv2.transform(pto_centro_upright, M_inv)[0][0]
                        sec_x, sec_y = pto_original

                        # Cruz Naranja Viva
                        p1_sec = (int(sec_x - (ancho_maximo/2) * v_trans_x), int(sec_y - (ancho_maximo/2) * v_trans_y))
                        p2_sec = (int(sec_x + (ancho_maximo/2) * v_trans_x), int(sec_y + (ancho_maximo/2) * v_trans_y))
                        cv2.line(frame, p1_sec, p2_sec, (0, 165, 255), t_medio)

                    # Enviamos estadísticas
                    self.stats_ready.emit({"confianza": round(float(mejor_confianza), 1)})
            # ========================================================
            # 4. CONVERSIÓN SEGURA A PYQT PARA 4 PANTALLAS
            # ========================================================
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h_f, w_f, ch_f = rgb_frame.shape
            qt_frame = QImage(rgb_frame.data, w_f, h_f, ch_f * w_f, QImage.Format.Format_RGB888).copy()

            rgb_mask1 = cv2.cvtColor(mask_filtrada, cv2.COLOR_GRAY2RGB)
            qt_mask1 = QImage(rgb_mask1.data, w_f, h_f, 3 * w_f, QImage.Format.Format_RGB888).copy()

            # PANTALLA 3: Imagen Tratada Pesada (El Kit le aplicó el Desenfoque y el Hexágono)
            rgb_tratada = cv2.cvtColor(img_tratada, cv2.COLOR_GRAY2RGB)
            qt_tratada = QImage(rgb_tratada.data, w_f, h_f, 3 * w_f, QImage.Format.Format_RGB888).copy()

            # PANTALLA 4: Máscara Simple (Solo Binarización)
            rgb_mask2 = cv2.cvtColor(mascara_naranja, cv2.COLOR_GRAY2RGB)
            qt_mask2 = QImage(rgb_mask2.data, w_f, h_f, 3 * w_f, QImage.Format.Format_RGB888).copy()

            self.frame_ready.emit(qt_frame, qt_mask1, qt_tratada, qt_mask2)

            elapsed = time.time() - start_time
            time.sleep(max(0, intervalo - elapsed))

        cap.release()
        self.finished.emit()

    def stop(self):
        self._running = False