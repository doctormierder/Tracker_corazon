# corazon/core/kit_procesamiento.py
import cv2
import numpy as np
from abc import ABC, abstractmethod
from .config import TrackerConfig

class ProcesadorFrame(ABC):
    @abstractmethod
    def procesar(self, frame: np.ndarray, config: TrackerConfig, hex_solido: np.ndarray, hex_suave: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

# --- EL NUEVO "LABORATORIO AVANZADO" ---
class KitBiologico(ProcesadorFrame):
    def procesar(self, frame: np.ndarray, config: TrackerConfig, hex_solido: np.ndarray, hex_suave: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. CLAHE (Ecualización Adaptativa) - Arregla luces/sombras
        # 
        if config.bio_clahe > 0:
            clahe = cv2.createCLAHE(clipLimit=float(config.bio_clahe/2.0), tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
        # 2. FILTRO BILATERAL - Suaviza fondo sin perder bordes
        if config.bio_bilateral > 0:
            d = int(config.bio_bilateral / 10) + 3
            gray = cv2.bilateralFilter(gray, d, config.bio_bilateral, config.bio_bilateral)

        # 3. TOP-HAT - Resalta estructuras pequeñas/brillantes (el latido)
        if config.bio_tophat > 0:
            k_size = config.bio_tophat if config.bio_tophat % 2 != 0 else config.bio_tophat + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # 4. SOBEL (Bordes) - Si quieres ver solo el "dibujo" de los bordes
        if config.bio_sobel > 0:
            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            alpha = config.bio_sobel / 50.0
            gray = cv2.addWeighted(gray, 1 - alpha, edges, alpha, 0)

        # --- TRATAMIENTO PARA PANTALLA 3 ---
        # Aplicamos el desenfoque del hexágono para crear la "niebla"
        val_blur = config.desenfoque if config.desenfoque % 2 != 0 else config.desenfoque + 1
        hex_difuminado = cv2.GaussianBlur(hex_suave, (val_blur, val_blur), 0)
        
        # Imagen Tratada: Píxeles procesados * Máscara de enfoque
        img_tratada = (gray.astype(np.float32) * (hex_difuminado.astype(np.float32) / 255.0)).astype(np.uint8)

        # --- MÁSCARA SIMPLE PARA PANTALLA 4 ---
        _, mask_medicion = cv2.threshold(img_tratada, config.tolerancia, 255, cv2.THRESH_BINARY)
        
        # Máscara de rastreo (usando la sensibilidad base)
        corte = max(0, np.max(gray) - config.sensibilidad)
        _, mask_rastreo = cv2.threshold(gray, corte, 255, cv2.THRESH_BINARY)
        mask_rastreo = cv2.bitwise_and(mask_rastreo, hex_solido)

        return mask_rastreo, img_tratada, mask_medicion

# Kit estándar (mantenlo para comparar)
class KitEstandar(ProcesadorFrame):
    def procesar(self, frame: np.ndarray, config: TrackerConfig, hex_solido: np.ndarray, hex_suave: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        val_blur_img = config.blur_imagen if config.blur_imagen % 2 != 0 else config.blur_imagen + 1
        blur_base = cv2.GaussianBlur(gray, (val_blur_img, val_blur_img), 0)
        
        val_blur_hex = config.desenfoque if config.desenfoque % 2 != 0 else config.desenfoque + 1
        hex_difuminado = cv2.GaussianBlur(hex_suave, (val_blur_hex, val_blur_hex), 0)
        
        img_tratada = (blur_base.astype(np.float32) * (hex_difuminado.astype(np.float32) / 255.0)).astype(np.uint8)
        _, mask_medicion = cv2.threshold(img_tratada, config.tolerancia, 255, cv2.THRESH_BINARY)
        
        corte = max(0, np.max(blur_base) - config.sensibilidad)
        _, mask_rastreo = cv2.threshold(blur_base, corte, 255, cv2.THRESH_BINARY)
        mask_rastreo = cv2.bitwise_and(mask_rastreo, hex_solido)

        return mask_rastreo, img_tratada, mask_medicion

class SelectorKits:
    @staticmethod
    def obtener_kit(nombre: str) -> ProcesadorFrame:
        kits = {
            "estandar": KitEstandar(),
            "biologico": KitBiologico()
        }
        return kits.get(nombre, KitEstandar())