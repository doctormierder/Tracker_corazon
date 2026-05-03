import cv2
import numpy as np
import math

def nada(x):
    pass

def viagem_no_tempo_otimizada(cap):
    print("Calculando Eixos e Mascara do Tempo... Aguarde.")
    angulos_antigos = []
    
    ret, frame_teste = cap.read()
    if not ret: return 90.0, 90.0, None, 0, 0
    acumulador = np.zeros(frame_teste.shape[:2], dtype=np.float32)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // 30)
    
    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        branco_maximo = np.max(blurred)
        
        for sensibilidade in range(10, 90, 5):
            ponto_de_corte = max(0, branco_maximo - sensibilidade)
            _, thresh = cv2.threshold(blurred, ponto_de_corte, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            acumulador += (closing / 255.0)
            
            contornos, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contornos:
                maior_contorno = max(contornos, key=cv2.contourArea)
                casco_convexo = cv2.convexHull(maior_contorno)
                if len(casco_convexo) >= 5:
                    _, eixos, angulo = cv2.fitEllipse(casco_convexo)
                    eixo_min, eixo_max = min(eixos[0], eixos[1]), max(eixos[0], eixos[1])
                    
                    if eixo_min > 0.1: 
                        relacao = eixo_max / eixo_min
                        peso = int(min(relacao, 50.0) * 5) 
                        angulos_antigos.extend([float(angulo)] * peso)
                            
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    angulo_base = np.median(angulos_antigos) if angulos_antigos else 90.0

    cv2.normalize(acumulador, acumulador, 0, 255, cv2.NORM_MINMAX)
    _, mascara_tempo = cv2.threshold(np.uint8(acumulador), 150, 255, cv2.THRESH_BINARY)
    
    # Parâmetros de Recompensa/Punição
    PESO_BRANCO = 2
    PENALIDADE_PRETO = 3
    BONUS_DISTANCIA = 5
    
    melhor_score = -float('inf')
    angulo_rosa = angulo_base
    melhor_px, melhor_py = 0, 0 # Armazena o ponto fixo absoluto
    
    M_tempo = cv2.moments(mascara_tempo)
    if M_tempo["m00"] != 0:
        cx_t, cy_t = int(M_tempo["m10"] / M_tempo["m00"]), int(M_tempo["m01"] / M_tempo["m00"])
        melhor_px, melhor_py = cx_t, cy_t # Fallback inicial
        
        for ang in range(int(angulo_base) - 10, int(angulo_base) + 11):
            ang_rad = math.radians(ang)
            
            for offset in range(-20, 21, 4): 
                px = cx_t + (offset * math.cos(ang_rad))
                py = cy_t + (offset * math.sin(ang_rad))
                
                pt1 = (int(px - 1000 * math.sin(ang_rad)), int(py + 1000 * math.cos(ang_rad)))
                pt2 = (int(px + 1000 * math.sin(ang_rad)), int(py - 1000 * math.cos(ang_rad)))
                
                line_mask = np.zeros_like(mascara_tempo)
                cv2.line(line_mask, pt1, pt2, 255, 1)
                
                intersecao = cv2.bitwise_and(line_mask, mascara_tempo)
                pontos_brancos = np.where(intersecao > 0)
                
                num_brancos = len(pontos_brancos[0])
                if num_brancos > 0:
                    distancia_pontas = math.sqrt((pontos_brancos[0][0] - pontos_brancos[0][-1])**2 + 
                                               (pontos_brancos[1][0] - pontos_brancos[1][-1])**2)
                    
                    total_linha = cv2.countNonZero(line_mask)
                    num_pretos = total_linha - num_brancos
                    
                    score = (num_brancos * PESO_BRANCO) - (num_pretos * PENALIDADE_PRETO) + (distancia_pontas * BONUS_DISTANCIA)
                    
                    if score > melhor_score:
                        melhor_score = score
                        angulo_rosa = ang
                        melhor_px = px # Salvamos a coordenada X exata da vitória
                        melhor_py = py # Salvamos a coordenada Y exata da vitória
    
    # Retornamos as coordenadas absolutas também
    return angulo_base, angulo_rosa, mascara_tempo, melhor_px, melhor_py

# --- CONFIGURAÇÃO ---
janela = 'Analise Multi-Eixo'
cv2.namedWindow(janela)
cv2.createTrackbar('Sensibilidade', janela, 30, 100, nada)

videos = ['coracao_video.mpg', 'coracao_video1.mpg', 'coracao_video2.avi', 'coracao_video3.avi', 'coracao_video4.avi', 'coracao_video5.mpg', 'coracao_video6.mpg']
video_idx = 0

while True:
    cap = cv2.VideoCapture(videos[video_idx])
    
    # Recebe os ângulos e as coordenadas fixas absolutas
    ang_amarelo, ang_rosa, img_tempo, fixo_x, fixo_y = viagem_no_tempo_otimizada(cap)
    
    rad_amarelo = math.radians(ang_amarelo)
    rad_rosa = math.radians(ang_rosa)

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        sens = cv2.getTrackbarPos('Sensibilidade', janela)
        _, thresh = cv2.threshold(blurred, np.max(blurred) - sens, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # O CYAN NÃO PRECISA DO CONTORNO! Ele é desenhado sempre, no mesmo lugar.
        dxc = 1000 * math.sin(rad_rosa)
        dyc = 1000 * math.cos(rad_rosa)
        cv2.line(frame, (int(fixo_x - dxc), int(fixo_y + dyc)), (int(fixo_x + dxc), int(fixo_y - dyc)), (255, 255, 0), 2) # Cyan, espessura 2
        
        if contornos:
            maior = max(contornos, key=cv2.contourArea)
            M = cv2.moments(maior)
            if M["m00"] != 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                
                # 1. EIXO DO TEMPO (VERDE) - Angulo deste frame, movendo com cx, cy
                if len(maior) >= 5:
                    _, _, ang_verde_frame = cv2.fitEllipse(maior)
                    rad_verde = math.radians(ang_verde_frame)
                    dxv, dyv = 1000 * math.sin(rad_verde), 1000 * math.cos(rad_verde)
                    cv2.line(frame, (int(cx-dxv), int(cy+dyv)), (int(cx+dxv), int(cy-dyv)), (0, 255, 0), 1)

                # 2. EIXO ANTIGO (AMARELO) - Angulo Base, movendo com cx, cy
                dxa, dya = 1000 * math.sin(rad_amarelo), 1000 * math.cos(rad_amarelo)
                cv2.line(frame, (int(cx-dxa), int(cy+dya)), (int(cx+dxa), int(cy-dya)), (0, 255, 255), 1)

                # 3. EIXO DEFINITIVO (ROSA) - Angulo Perfeito, movendo com cx, cy
                dxr, dyr = 1000 * math.sin(rad_rosa), 1000 * math.cos(rad_rosa)
                cv2.line(frame, (int(cx-dxr), int(cy+dyr)), (int(cx+dxr), int(cy-dyr)), (255, 0, 255), 1)

        cv2.putText(frame, "AMARELO: Base | VERDE: Frame | ROSA: Segue Massa | CYAN: Absoluto", (10, 30), 1, 1, (255,255,255), 2)
        cv2.imshow(janela, frame)
        if img_tempo is not None: cv2.imshow('Raio-X (Mascara Tempo)', img_tempo)
        
        tecla = cv2.waitKey(30) & 0xFF
        if tecla == ord('q'): exit()
        if tecla == ord('n'): break

    cap.release()
    video_idx = (video_idx + 1) % len(videos)