# corazon/buscando_eje.py
import cv2
import numpy as np
import math

def viaje_en_el_tiempo_definitivo(cap):
    """
    Analiza todo el video mediante acumulación de intensidad luminosa.
    Incluye limpieza topológica avanzada para una Máscara del Tiempo perfecta.
    Devuelve: angulo_cyan, mascara_tiempo, fijo_x, fijo_y, eje_mayor, eje_menor
    """
    print("Calculando Eje Absoluto (Cyan) y Dimensiones... Por favor espera.")
    angulos_antigos = []
    
    ret, frame_prueba = cap.read()
    if not ret: return 90.0, None, 0, 0, 100, 50
    acumulador = np.zeros(frame_prueba.shape[:2], dtype=np.float32)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // 30)
    
    # --- 1. VIAJE EN EL TIEMPO (Tu acumulador original) ---
    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue
            
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            gray = frame  # Ya nos llegó en escala de grises
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        blanco_max = np.max(blurred)
        
        # Reducimos el límite superior de 90 a 55. 
        # Al tener limpieza morfológica después, no necesitamos recolectar "grises oscuros".
        for sensibilidad in range(45, 86, 5): 
            corte = max(0, blanco_max - sensibilidad)
            _, thresh = cv2.threshold(blurred, corte, 255, cv2.THRESH_BINARY)
            kernel = np.ones((9, 9), np.uint8)
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
    
    # =================================================================
    # 🌟 NUEVA FASE: LIMPIEZA TOPOLÓGICA DE LA MÁSCARA DEL TIEMPO
    # =================================================================
    # 1. Apertura: Elimina islas de ruido (brillos estáticos lejos del corazón)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mascara_tiempo = cv2.morphologyEx(mascara_tiempo, cv2.MORPH_OPEN, kernel_open)
    
    # 2. Cierre: Sella el interior del corazón para que el Escáner no cuente píxeles negros por error
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mascara_tiempo = cv2.morphologyEx(mascara_tiempo, cv2.MORPH_CLOSE, kernel_close)
    # =================================================================

    # =================================================================
    # 🐛 DEBUG: VER LA MÁSCARA DEL TIEMPO (Descomentar para probar)
    # =================================================================
    # cv2.imshow("DEBUG - Mascara del Tiempo", mascara_tiempo)
    # print("👁️ Viendo máscara... Presiona CUALQUIER TECLA en la ventana de la imagen para continuar.")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # =================================================================
    
    # --- 2. ESCÁNER CYAN (AHORA 100% RELATIVO A LA RESOLUCIÓN) ---
    mejor_score = -float('inf')
    angulo_cyan = angulo_base
    fijo_x, fijo_y = 0, 0 
    p1_ganador = (0, 0) # Inicializamos
    p2_ganador = (0, 0)
    
    M_tiempo = cv2.moments(mascara_tiempo)
    if M_tiempo["m00"] != 0:
        cx_t, cy_t = int(M_tiempo["m10"] / M_tiempo["m00"]), int(M_tiempo["m01"] / M_tiempo["m00"])
        fijo_x, fijo_y = cx_t, cy_t
        
        # 1. Parámetros Relativos
        alto_img, ancho_img = mascara_tiempo.shape
        grosor_rectangulo = max(1, int(ancho_img * 0.05))  # 1% del ancho
        
        # 2. Desplazamiento Relativo (+/- 1% del ancho total de la imagen)
        rango_offset = max(5, int(ancho_img * 0.1)) 
        # Queremos hacer siempre ~20 saltos de evaluación, sin importar la resolución
        salto_offset = max(1, int(rango_offset / 10)) 
        
        for ang in range(int(angulo_base) - 30, int(angulo_base) + 31, 2):
            rad = math.radians(ang)
            
            # EL BUCLE DINÁMICO
            for offset in range(-rango_offset, rango_offset + 1, salto_offset): 
                px = cx_t + (offset * math.cos(rad))
                py = cy_t + (offset * math.sin(rad))
                
                pt1 = (int(px - 1000 * math.sin(rad)), int(py + 1000 * math.cos(rad)))
                pt2 = (int(px + 1000 * math.sin(rad)), int(py - 1000 * math.cos(rad)))
                
                line_mask = np.zeros_like(mascara_tiempo)
                cv2.line(line_mask, pt1, pt2, 255, thickness=grosor_rectangulo)
                
                # ... (El resto del código de la intersección y el score sigue igual)
                interseccion = cv2.bitwise_and(line_mask, mascara_tiempo)
                interseccion = cv2.bitwise_and(line_mask, mascara_tiempo)
                pts_blancos = np.where(interseccion > 0)
                
                if len(pts_blancos[0]) > 0:
                    # Distancia (largo) aproximada
                    punto_inicio = (int(pts_blancos[1][0]), int(pts_blancos[0][0]))
                    punto_fin = (int(pts_blancos[1][-1]), int(pts_blancos[0][-1]))
                    
                    distancia = math.sqrt((punto_inicio[0] - punto_fin[0])**2 + 
                                          (punto_inicio[1] - punto_fin[1])**2)
                    
                    
                    area_total_rectangulo = cv2.countNonZero(line_mask)
                    area_blanca_real = cv2.countNonZero(interseccion)
                    pretos = area_total_rectangulo - area_blanca_real
                    
                    # NORMALIZAMOS LOS PÍXELES dividiendo por el grosor para no romper tu fórmula
                    blancos_norm = area_blanca_real / grosor_rectangulo
                    pretos_norm = pretos / grosor_rectangulo
                    
                    score = (blancos_norm * 3) - (pretos_norm * 50) + (distancia * 80)
                    
                    if score > mejor_score:
                        mejor_score = score
                        angulo_cyan = ang
                        fijo_x, fijo_y = px, py
                        p1_ganador = punto_inicio
                        p2_ganador = punto_fin
    
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

    return angulo_cyan, mascara_tiempo, fijo_x, fijo_y, eje_mayor, eje_menor, p1_ganador, p2_ganador