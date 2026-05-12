# corazon/video_worker.py
import cv2
import time
import math
import sys
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage
from collections import deque

from .buscando_eje import viaje_en_el_tiempo_definitivo
from .core.config import TrackerConfig
from .core.kit_procesamiento import SelectorKits
from .analisis_fisiologico import AnalizadorFisiologico

class VideoWorker(QObject):
    """
    Trabajador principal del motor de rastreo.
    Responsabilidades:
    1. Geometría Hexagonal absoluta y dinámica.
    2. Umbralización topográfica por franjas.
    3. Simetría Instantánea (IoU) sin suavizado temporal.
    4. Envío de datos al Analizador Fisiológico.
    """
    
    # Señales de comunicación con la UI (Pantalla 1, Pantalla 2, Pantalla 3, Pantalla 4)
    frame_ready = pyqtSignal(QImage, QImage, QImage, QImage) 
    stats_ready = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, video_path, config: TrackerConfig):
        super().__init__()
        self.video_path = video_path
        self.config = config
        self._running = True
        
        # Módulo de análisis estadístico y generación de ECG
        self.analizador = AnalizadorFisiologico()
    
        # --- ATRIBUTOS DE CLASE PARA EVITAR ERRORES DE SCOPE ---
        # Referencia de resolución base (1080p) para que los dibujos mantengan su proporción
        self.REF_WIDTH = 1920.0 
        
        # Variables de calibración (Fase 0) que deben perdurar en el tiempo
        self.p1_calib = (0, 0)
        self.p2_calib = (0, 0)
        self.angulo_base = 0.0
        self.eje_mayor = 0
        self.eje_menor = 0
    
        # Sistema de Fricción visual (Solo para estabilizar el dibujo de los ejes rojo/verde, no afecta los cálculos puros)
        self.historial_cx = deque(maxlen=10)
        self.historial_cy = deque(maxlen=10)
        self.historial_rojo_x = deque(maxlen=30)
        self.historial_rojo_y = deque(maxlen=30)

        # Variables en Caché
        self.hex_solido_maestro = None
        self.roi_hex_coords = None
        self.extremos_corazon = [] 
        self._last_blur_val = -1
        self._hex_suave_cache = None

        # Frame anterior para mapa de movimiento
        self._prev_gray = None

    def escalar_pixel(self, valor_base, width_actual):
        """
        Calcula el tamaño en píxeles que debe tener un dibujo (línea, círculo) 
        en función de la resolución actual del video, evitando grosores invisibles en 4K 
        o invasivos en 480p.
        """
        factor = width_actual / self.REF_WIDTH
        return max(1, int(valor_base * factor))

    def stop(self):
        """Detiene el bucle principal de forma segura."""
        self._running = False

    @pyqtSlot()
    def run(self):
        # Selección dinámica de motor según el Sistema Operativo
        cap = cv2.VideoCapture(self.video_path)
        
        # Detectamos si estamos en Linux lidiando con el formato problemático
        es_mpg_linux = sys.platform.startswith("linux") and self.video_path.lower().endswith(('.mpg', '.mpeg'))
        
        if es_mpg_linux:
            # APAGAMOS EL SWSCLALER: Le pedimos a OpenCV la matriz YUV cruda (sin convertir)
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        if not cap.isOpened():
            print(f"Error: OpenCV no pudo abrir {self.video_path}.")
            self.finished.emit()
            return

        # =================================================================
        # FASE 0: VIAJE EN EL TIEMPO Y CALIBRACIÓN DE EJES
        # =================================================================
        res = viaje_en_el_tiempo_definitivo(cap)
        # Desempaquetamos los 8 valores devueltos por la nueva versión de la Fase 0
        (self.angulo_base, mascara_tiempo, fijo_x, fijo_y, 
         self.eje_mayor, self.eje_menor, self.p1_calib, self.p2_calib) = res

        if mascara_tiempo is None:
            self.finished.emit()
            return

        h_f, w_f = mascara_tiempo.shape
        rad_cyan = math.radians(self.angulo_base)
        
        v_long_x, v_long_y = math.sin(rad_cyan), -math.cos(rad_cyan)
        v_trans_x, v_trans_y = math.cos(rad_cyan), math.sin(rad_cyan)
        
        # 1. Trazar el eje y calcular intersección con bordes de cámara
        pt1 = (int(fijo_x + 5000 * v_long_x), int(fijo_y + 5000 * v_long_y))
        pt2 = (int(fijo_x - 5000 * v_long_x), int(fijo_y - 5000 * v_long_y))
        
        rect_img = (0, 0, w_f, h_f)
        ret_clip, p_a, p_b = cv2.clipLine(rect_img, pt1, pt2)
        
        if ret_clip:
            centro_vis_x = (p_a[0] + p_b[0]) / 2.0
            centro_vis_y = (p_a[1] + p_b[1]) / 2.0
            largo_pantalla = math.hypot(p_a[0] - p_b[0], p_a[1] - p_b[1])
            
            # Geometría: +40% de largo (1.40), 50% de ancho del largo total
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

            # Precalculamos el ROI del Hexágono para cálculos matemáticos veloces
            y_idx, x_idx = np.where(self.hex_solido_maestro > 0)
            if len(y_idx) > 0:
                self.roi_hex_coords = (np.min(y_idx), np.max(y_idx)+1, np.min(x_idx), np.max(x_idx)+1)
            else:
                self.roi_hex_coords = (0, h_f, 0, w_f)
        else:
            self.hex_solido_maestro = np.zeros((h_f, w_f), dtype=np.uint8)
            self.roi_hex_coords = (0, h_f, 0, w_f)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        procesador = SelectorKits.obtener_kit(self.config.kit_procesamiento)
        
        # Longitudes para el dibujo de los ejes de debug
        lx, ly = 3000 * v_long_x, 3000 * v_long_y
        
        # Recuperamos los FPS objetivo si existen en config, si no, usamos 30 por defecto
        fps_target = getattr(self.config, 'fps_objetivo', 30.0)

        # =================================================================
        # FASE 1: BUCLE DE TRACKING EN TIEMPO REAL
        # =================================================================
        while self._running:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret: 
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            h_f_actual, w_f_actual = frame.shape[:2]

                
            # --- BYPASS DE DECODIFICACIÓN MANUAL PARA LINUX ---
            if es_mpg_linux:
                # Si OpenCV nos tiró la toalla y nos dio un 8UC1 (Gris puro)
                if len(frame.shape) == 2 or frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    try:
                        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
                    except cv2.error:
                        try:
                            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
                        except cv2.error:
                            break

                
            # =======================================================
            # DE AQUÍ EN ADELANTE, ES TU CÓDIGO DE SIEMPRE
            # img_cv = frame.copy() ...
            # =======================================================

            # --- 1.1 CACHÉ DE DESENFOQUE ---
            if self._last_blur_val != self.config.desenfoque:
                blur_ratio = max(1, int(self.config.desenfoque)) / 100.0
                k_size = int(L_h * blur_ratio) if 'L_h' in locals() else 5
                if k_size % 2 == 0: k_size += 1
                k_size = max(3, k_size)
                self._hex_suave_cache = cv2.GaussianBlur(self.hex_solido_maestro, (k_size, k_size), 0)
                self._last_blur_val = self.config.desenfoque

            hex_suave = self._hex_suave_cache

            # --- 1.2 PROCESAMIENTO PIXEL A PIXEL (Kit Biológico) ---
            resultado_kit = procesador.procesar(frame, self.config, self.hex_solido_maestro, hex_suave)
            if len(resultado_kit) == 3:
                mask_rastreo, img_tratada, _ = resultado_kit
            else:
                mask_rastreo, img_tratada = resultado_kit

            # --- MAPA DE MOVIMIENTO (frame differencing) ---
            gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_curr = cv2.GaussianBlur(gray_curr, (5, 5), 0)
            if self._prev_gray is not None:
                motion_map = cv2.absdiff(gray_curr, self._prev_gray)
                motion_map = cv2.GaussianBlur(motion_map, (5, 5), 0)
            else:
                motion_map = np.zeros_like(gray_curr)
            self._prev_gray = gray_curr.copy()

            # --- 1.3 UMBRALIZACIÓN TOPOGRÁFICA ADAPTATIVA (4-bit mask) ---
            y1_h, y2_h, x1_h, x2_h = self.roi_hex_coords
            roi_tratada = img_tratada[y1_h:y2_h, x1_h:x2_h]
            roi_mask = self.hex_solido_maestro[y1_h:y2_h, x1_h:x2_h]

            Y_roi, X_roi = np.indices(roi_tratada.shape)
            X_global = X_roi + x1_h
            Y_global = Y_roi + y1_h

            proyeccion_long = (X_global * v_long_x) + (Y_global * v_long_y)
            pixeles_validos = roi_mask > 0

            mask_4bit = np.zeros_like(img_tratada, dtype=np.uint8)
            mask_binaria = np.zeros_like(img_tratada, dtype=np.uint8)

            if np.any(pixeles_validos):
                proj_validas = proyeccion_long[pixeles_validos]
                min_p, max_p = np.min(proj_validas), np.max(proj_validas)

                num_franjas = 16
                rango_franja = (max_p - min_p) / num_franjas
                mapa_umbrales = np.zeros_like(roi_tratada, dtype=np.float32)
                val_slider = self.config.tolerancia

                for i in range(num_franjas):
                    lim_inf = min_p + i * rango_franja
                    lim_sup = lim_inf + rango_franja if i < num_franjas - 1 else max_p + 1

                    mask_franja = pixeles_validos & (proyeccion_long >= lim_inf) & (proyeccion_long < lim_sup)
                    pixeles_franja = roi_tratada[mask_franja]

                    if len(pixeles_franja) > 10:
                        luz_fondo = np.percentile(pixeles_franja, 50)
                        max_absoluto = np.percentile(pixeles_franja, 98)
                    else:
                        luz_fondo, max_absoluto = 127, 255

                    if val_slider >= 50:
                        porcentaje = (val_slider - 50) / 50.0
                        umbral_franja = luz_fondo + ((max_absoluto - luz_fondo) * porcentaje)
                    else:
                        porcentaje = (50 - val_slider) / 49.0
                        umbral_franja = luz_fondo - ((max_absoluto - luz_fondo) * porcentaje)

                    mapa_umbrales[mask_franja] = np.clip(umbral_franja, 0, 255)

                mapa_suave = cv2.GaussianBlur(mapa_umbrales, (19, 19), 0)

                # Binary mask (for visualization only)
                mask_binaria_roi = (roi_tratada >= mapa_suave).astype(np.uint8) * 255
                mask_binaria_roi = cv2.bitwise_and(mask_binaria_roi, roi_mask)
                mask_binaria[y1_h:y2_h, x1_h:x2_h] = mask_binaria_roi

                # 4-bit mask: how far above threshold, mapped to 0-15
                diff = roi_tratada.astype(np.float32) - mapa_suave
                diff = np.clip(diff, 0, None)
                diff = diff * (roi_mask > 0).astype(np.float32)
                if np.any(diff > 0):
                    p95 = np.percentile(diff[diff > 0], 95)
                    max_diff = max(p95, 1.0)
                    mask_4bit_roi = np.clip((diff / max_diff * 15).astype(np.uint8), 0, 15)
                else:
                    mask_4bit_roi = np.zeros(roi_tratada.shape, dtype=np.uint8)
                mask_4bit[y1_h:y2_h, x1_h:x2_h] = mask_4bit_roi

            # --- 1.4 ANÁLISIS FISIOLÓGICO (fusión temporal multi-señal) ---
            ancho_f, pts_debug = self.analizador.procesar_frame(
                img_tratada, mask_4bit, motion_map,
                self.p1_calib, self.p2_calib, self.angulo_base
            )

            # Dibujo de línea de medición (Debug) usando proporciones de pantalla
            grosor_debug = self.escalar_pixel(2, w_f_actual)
            radio_debug = self.escalar_pixel(4, w_f_actual)
            
            if pts_debug:
                cv2.line(frame, pts_debug[0], pts_debug[1], (255, 0, 255), grosor_debug)
                cv2.circle(frame, pts_debug[0], radio_debug, (255, 255, 255), -1)
                cv2.circle(frame, pts_debug[1], radio_debug, (255, 255, 255), -1)

            # --- 1.5 CÁLCULO DE POSICIÓN ÓPTIMA (Puntuación por Línea Gruesa) ---
            contornos, _ = cv2.findContours(mask_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contornos:
                mayor = max(contornos, key=cv2.contourArea)
                M = cv2.moments(mayor)
                if M["m00"] != 0:
                    base_cx, base_cy = M["m10"]/M["m00"], M["m01"]/M["m00"]

                    ganador_cx, ganador_cy = base_cx, base_cy
                    mejor_score = -float('inf')
                    
                    desplazamientos = [-15, -5, 0, 5, 15] 
                    
                    bx, by, bw, bh = cv2.boundingRect(mayor)
                    pad = max(30, int(w_f_actual * 0.05)) 
                    y1 = max(0, by - pad); y2 = min(h_f_actual, by + bh + pad)
                    x1 = max(0, bx - pad); x2 = min(w_f_actual, bx + bw + pad)
                    
                    roi_rastreo = mask_binaria[y1:y2, x1:x2]
                    grosor_test = self.escalar_pixel(15, w_f_actual) 

                    for offset in desplazamientos:
                        test_cx = base_cx + v_trans_x * offset
                        test_cy = base_cy + v_trans_y * offset

                        temp_roi = np.zeros_like(roi_rastreo)
                        pt1_roi = (int(test_cx - lx) - x1, int(test_cy - ly) - y1)
                        pt2_roi = (int(test_cx + lx) - x1, int(test_cy + ly) - y1)
                        
                        cv2.line(temp_roi, pt1_roi, pt2_roi, 255, thickness=grosor_test)

                        inter = cv2.bitwise_and(roi_rastreo, temp_roi)
                        blancos = cv2.countNonZero(inter)
                        
                        total_area_linea = cv2.countNonZero(temp_roi)
                        negros = total_area_linea - blancos
                        
                        score = (blancos * 9) - (negros * 40) - (abs(offset) * 9)
                        
                        if score > mejor_score:
                            mejor_score = score
                            ganador_cx, ganador_cy = test_cx, test_cy

                    # --- Ejes visuales (Mantienen la inercia visual para el dibujo limpio) ---
                    self.historial_cx.append(ganador_cx); self.historial_cy.append(ganador_cy)
                    cx_verde = np.median(self.historial_cx)
                    cy_verde = np.median(self.historial_cy)

                    media_x = (2.0 * fijo_x + cx_verde) / 3.0
                    media_y = (2.0 * fijo_y + cy_verde) / 3.0
                    self.historial_rojo_x.append(media_x); self.historial_rojo_y.append(media_y)
                    cx_rojo = int(np.median(self.historial_rojo_x))
                    cy_rojo = int(np.median(self.historial_rojo_y))

                    if getattr(self.config, 'mostrar_rosa', True):
                        cv2.line(frame, (int(cx_verde - lx), int(cy_verde - ly)), 
                                        (int(cx_verde + lx), int(cy_verde + ly)), (0, 255, 0), 1)

                    if getattr(self.config, 'mostrar_elipse', True): 
                        cv2.line(frame, (int(cx_rojo - lx), int(cy_rojo - ly)), 
                                        (int(cx_rojo + lx), int(cy_rojo + ly)), (0, 0, 255), self.escalar_pixel(2, w_f_actual))

            # --- 1.6 DIBUJO DE PUNTOS CALIBRADORES ---
            radio_calib = self.escalar_pixel(8, w_f_actual)
            grosor_calib = self.escalar_pixel(1, w_f_actual)
            cv2.circle(frame, self.p1_calib, radio_calib, (0, 255, 255), -1)
            cv2.circle(frame, self.p2_calib, radio_calib, (0, 255, 255), -1)
            cv2.line(frame, self.p1_calib, self.p2_calib, (0, 255, 255), grosor_calib)

            # --- 1.7 RECOPILACIÓN Y EMISIÓN DE MÉTRICAS ---
            stats = self.analizador.obtener_estadisticas()
            stats["confianza"] = round(float(0), 1)
            stats["ancho_actual"] = round(float(ancho_f), 2)
            # Inyectamos de forma explícita para asegurar compatibilidad con la UI actual
            stats["max_prom"] = round(float(self.analizador.ancho_max_promedio), 2)
            stats["min_prom"] = round(float(self.analizador.ancho_min_promedio), 2)
            
            self.stats_ready.emit(stats)

            # --- 1.8 EMISIÓN A PANTALLAS (CONVERSIONES SEGURAS QImage) ---
            def to_qt_image(img_cv, is_gray=False):
                if is_gray:
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
                else:
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                h, w, ch = img_cv.shape
                return QImage(img_cv.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

            qt_f = to_qt_image(frame)
            qt_tr = to_qt_image(img_tratada, is_gray=True)
            qt_m2 = to_qt_image(mask_binaria, is_gray=True)
            
            # Generamos el cardiograma de tamaño idéntico a la cámara
            img_cardio = self.analizador.generar_cardiograma(w_f_actual, h_f_actual)
            qt_cardio = to_qt_image(img_cardio)

            self.frame_ready.emit(qt_f, qt_cardio, qt_tr, qt_m2)

            # --- 1.9 CONTROL DE LATENCIA (FPS) ---
            elapsed = time.time() - start_time
            if fps_target > 0:
                time.sleep(max(0, (1.0 / fps_target) - elapsed))

        cap.release()
        self.finished.emit()