# corazon/buscando_eje.py
import cv2
import numpy as np
import math

def viaje_en_el_tiempo_definitivo(cap):
    """
    Analiza todo el video rápidamente para crear la Máscara del Tiempo.
    Devuelve: angulo_cyan, mascara_tiempo, fijo_x, fijo_y, eje_mayor, eje_menor
    """
    print("Calculando Eje Absoluto (Cyan) y Dimensiones... Por favor espera.")
    angulos_antigos = []
    
    ret, frame_prueba = cap.read()
    if not ret: return 90.0, None, 0, 0, 100, 50
    acumulador = np.zeros(frame_prueba.shape[:2], dtype=np.float32)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // 30)
    
    # --- 1. VIAJE EN EL TIEMPO ---
    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        blanco_max = np.max(blurred)
        
        for sensibilidad in range(10, 90, 5):
            corte = max(0, blanco_max - sensibilidad)
            _, thresh = cv2.threshold(blurred, corte, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            acumulador += (closing / 255.0)
            
            contornos, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contornos:
                mayor = max(contornos, key=cv2.contourArea)
                casco = cv2.convexHull(mayor)
                if len(casco) >= 5:
                    _, eixos, ang = cv2.fitEllipse(casco)
                    e_min, e_max = min(eixos[0], eixos[1]), max(eixos[0], eixos[1])
                    if e_min > 0.1: 
                        peso = int(min(e_max / e_min, 50.0) * 5) 
                        angulos_antigos.extend([float(ang)] * peso)
                            
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    angulo_base = np.median(angulos_antigos) if angulos_antigos else 90.0

    cv2.normalize(acumulador, acumulador, 0, 255, cv2.NORM_MINMAX)
    _, mascara_tiempo = cv2.threshold(np.uint8(acumulador), 150, 255, cv2.THRESH_BINARY)
    
    # --- 2. ESCÁNER CYAN ---
    mejor_score = -float('inf')
    angulo_cyan = angulo_base
    fijo_x, fijo_y = 0, 0 
    
    M_tiempo = cv2.moments(mascara_tiempo)
    if M_tiempo["m00"] != 0:
        cx_t, cy_t = int(M_tiempo["m10"] / M_tiempo["m00"]), int(M_tiempo["m01"] / M_tiempo["m00"])
        fijo_x, fijo_y = cx_t, cy_t
        
        for ang in range(int(angulo_base) - 10, int(angulo_base) + 11):
            rad = math.radians(ang)
            for offset in range(-20, 21, 4): 
                px = cx_t + (offset * math.cos(rad))
                py = cy_t + (offset * math.sin(rad))
                
                pt1 = (int(px - 1000 * math.sin(rad)), int(py + 1000 * math.cos(rad)))
                pt2 = (int(px + 1000 * math.sin(rad)), int(py - 1000 * math.cos(rad)))
                
                line_mask = np.zeros_like(mascara_tiempo)
                cv2.line(line_mask, pt1, pt2, 255, 1)
                
                interseccion = cv2.bitwise_and(line_mask, mascara_tiempo)
                pts_blancos = np.where(interseccion > 0)
                
                if len(pts_blancos[0]) > 0:
                    distancia = math.sqrt((pts_blancos[0][0] - pts_blancos[0][-1])**2 + 
                                          (pts_blancos[1][0] - pts_blancos[1][-1])**2)
                    pretos = cv2.countNonZero(line_mask) - len(pts_blancos[0])
                    score = (len(pts_blancos[0]) * 2) - (pretos * 10) + (distancia * 20)
                    
                    if score > mejor_score:
                        mejor_score = score
                        angulo_cyan = ang
                        fijo_x, fijo_y = px, py
    
    # --- 3. DIMENSIONES ELIPSE ---
    puntos_y, puntos_x = np.where(mascara_tiempo > 0)
    eje_mayor, eje_menor = 100, 50
    if len(puntos_x) > 0:
        rad_cyan = math.radians(angulo_cyan)
        dx = puntos_x - fijo_x
        dy = puntos_y - fijo_y
        
        dist_largo = np.abs(dx * (-math.sin(rad_cyan)) + dy * math.cos(rad_cyan))
        dist_ancho = np.abs(dx * math.cos(rad_cyan) + dy * math.sin(rad_cyan))
        
        eje_mayor = int(np.max(dist_largo)) + 20
        eje_menor = int(np.max(dist_ancho)) + 20

    return angulo_cyan, mascara_tiempo, fijo_x, fijo_y, eje_mayor, eje_menor