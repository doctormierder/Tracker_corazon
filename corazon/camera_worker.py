# corazon/camera_worker.py
import cv2
import time
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage
from .core.config import TrackerConfig
from .core.kit_procesamiento import SelectorKits

class CameraWorker(QObject):
    frame_ready = pyqtSignal(QImage, QImage) 
    log_msg = pyqtSignal(str) # Para enviar mensajes al chat
    finished = pyqtSignal()

    def __init__(self, camera_index, config: TrackerConfig):
        super().__init__()
        self.camera_index = camera_index
        self.config = config
        self._running = True
        
        # --- MEMORIA A CORTO PLAZO (Tu idea del buffer circular) ---
        self.historial_angulos = []
        self.historial_ejes = []
        self.exposicion_primaria = None
        self.exposicion_secundaria = None
        self.frame_count = 0

    @pyqtSlot()
    def run(self):
        # En OpenCV, 0 suele ser la webcam, 1 o 2 la cámara del microscopio
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.log_msg.emit("❌ Error: No se pudo conectar a la cámara.")
            self.finished.emit()
            return

        self.log_msg.emit("✅ Cámara conectada. Iniciando calibración en vivo...")
        procesador = SelectorKits.obtener_kit(self.config.kit_procesamiento)

        while self._running:
            ret, frame = cap.read()
            if not ret: continue

            # 1. Procesamiento Básico
            mask = procesador.procesar(frame, self.config)
            
            # --- AQUÍ IRA TU LÓGICA DE EXPOSICIÓN PROGRESIVA ---
            # Por ahora, solo inicializamos la exposición para que no crashee
            if self.exposicion_primaria is None:
                self.exposicion_primaria = np.zeros_like(mask, dtype=np.float32)
            
            # (El código del doble buffer y medianas que programaremos en el futuro irá aquí)
            
            # 2. Conversión segura para PyQt
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h_f, w_f, ch_f = rgb_frame.shape
            qt_frame = QImage(rgb_frame.data, w_f, h_f, ch_f * w_f, QImage.Format.Format_RGB888).copy()

            rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            h_m, w_m, ch_m = rgb_mask.shape
            qt_mask = QImage(rgb_mask.data, w_m, h_m, ch_m * w_m, QImage.Format.Format_RGB888).copy()

            self.frame_ready.emit(qt_frame, qt_mask)
            time.sleep(0.03) # Freno de 30fps aprox

        cap.release()
        self.finished.emit()

    def stop(self):
        self._running = False