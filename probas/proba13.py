import cv2
import numpy as np
import math

def nada(x):
    pass

def viagem_no_tempo_otimizada(cap):
    """
    Calcula o Eixo Base e passa um "Scanner" restrito (+- 10 graus) na Máscara do Tempo,
    recompensando pixels brancos e punindo duramente pixels pretos.
    """
    print("Iniciando Viagem no Tempo... Calculando Eixos. Aguarde.")
    angulos_antigos = []
    
    ret, frame_teste = cap.read()
    if not ret: return 90.0, 90.0, None
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
            
            # 1. Alimenta a Máscara do Tempo
            acumulador += (closing / 255.0)
            
            # 2. Calcula o Eixo Base (Elipse)
            contornos, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contornos:
                maior_contorno = max(contornos, key=cv2.contourArea)
                casco_convexo = cv2.convexHull(maior_contorno)
                if len(casco_convexo) >= 5:
                    _, eixos, angulo = cv2.fitEllipse(casco_convexo)
                    eixo_min, eixo_max = min(eixos[0], eixos[1]), max(eixos[0], eixos[1])
                    if eixo_min > 0 and (eixo_max / eixo_min) > 1.3:
                        angulos_antigos.extend([angulo] * int((eixo_max / eixo_min) * 10))
                            
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # --- RESULTADO DO EIXO BASE ---
    angulo_base = np.median(angulos_antigos) if angulos_antigos else 90.0

    # --- PROCESSANDO A MÁSCARA DO TEMPO ---
    cv2.normalize(acumulador, acumulador, 0, 255, cv2.NORM_MINMAX)
    _, mascara_tempo = cv2.threshold(np.uint8(acumulador), 150, 255, cv2.THRESH_BINARY)
    
    # --- O SEU SCANNER DE RECOMPENSAS/PUNIÇÕES ---
    melhor_score = -float('inf')
    angulo_refinado = angulo_base
    
    # Acha o centro aproximado do músculo na Máscara do Tempo
    M_tempo = cv2.moments(mascara_tempo)
    if M_tempo["m00"] != 0:
        cx_t = int(M_tempo["m10"] / M_tempo["m00"])
        cy_t = int(M_tempo["m01"] / M_tempo["m00"])
        
        # Regra 1: Atado ao eixo antigo (divergência máxima de 10 graus)
        for ang in range(int(angulo_base) - 10, int(angulo_base) + 11):
            ang_rad = math.radians(ang)
            
            # Testa a linha no centro e com pequenos desvios paralelos para achar a "veia" mais branca
            for offset in range(-30, 31, 3): 
                px = cx_t + (offset * math.cos(ang_rad))
                py = cy_t + (offset * math.sin(ang_rad))
                
                # Cria uma linha imaginária muito longa
                pt1 = (int(px - 2000 * math.sin(ang_rad)), int(py + 2000 * math.cos(ang_rad)))
                pt2 = (int(px + 2000 * math.sin(ang_rad)), int(py - 2000 * math.cos(ang_rad)))
                
                line_mask = np.zeros_like(mascara_tempo)
                cv2.line(line_mask, pt1, pt2, 255, 1) # Linha de 1 pixel de espessura
                
                # Regra 2 e 3: Contar brancos e pretos
                intersecao = cv2.bitwise_and(line_mask, mascara_tempo)
                brancos = cv2.countNonZero(intersecao)
                total_pixels_na_linha = cv2.countNonZero(line_mask)
                pretos = total_pixels_na_linha - brancos
                
                # A Mágica: Dobra o valor do branco, e subtrai o preto
                score = (brancos * 2) - pretos
                
                if score > melhor_score:
                    melhor_score = score
                    angulo_refinado = ang
    
    print(f"-> Eixo Base (Amarelo) : {angulo_base:.2f} graus")
    print(f"-> Eixo Scanner (Verde): {angulo_refinado:.2f} graus (Puniu horizontais!)")
    
    return angulo_base, angulo_refinado, mascara_tempo

# ==========================================
# CONFIGURAÇÃO DA INTERFACE
# ==========================================
janela_principal = 'Duelo de Eixos: Scanner Restrito'
cv2.namedWindow(janela_principal)
cv2.createTrackbar('Sensibilidade', janela_principal, 30, 100, nada)

videos = ['coracao_video.mpg', 'coracao_video1.mpg', 'coracao_video2.avi', 'coracao_video3.avi', 'coracao_video4.avi', 'coracao_video5.mpg', 'coracao_video6.mpg']
video_atual_idx = 0

while True:
    nome_video = videos[video_atual_idx]
    cap = cv2.VideoCapture(nome_video)
    print(f"\n--- Iniciando: {nome_video} ---")
    
    angulo_antigo, angulo_novo, imagem_tempo = viagem_no_tempo_otimizada(cap)
    
    ang_rad_antigo = math.radians(angulo_antigo)
    ang_rad_novo = math.radians(angulo_novo)

    sair_de_tudo = False

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        branco_maximo = np.max(blurred)
        sensibilidade = cv2.getTrackbarPos('Sensibilidade', janela_principal)
        ponto_de_corte = max(0, branco_maximo - sensibilidade)
        _, thresh = cv2.threshold(blurred, ponto_de_corte, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contornos:
            maior_contorno = max(contornos, key=cv2.contourArea)
            casco_convexo = cv2.convexHull(maior_contorno)
            
            M = cv2.moments(casco_convexo)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                tamanho_infinito = 2000
                
                # --- DESENHA O EIXO BASE (AMARELO) ---
                dx_a = tamanho_infinito * math.sin(ang_rad_antigo)
                dy_a = tamanho_infinito * math.cos(ang_rad_antigo)
                cv2.line(frame, (int(cx - dx_a), int(cy + dy_a)), (int(cx + dx_a), int(cy - dy_a)), (0, 255, 255), 2)
                
                # --- DESENHA O EIXO SCANNER (VERDE) ---
                dx_n = tamanho_infinito * math.sin(ang_rad_novo)
                dy_n = tamanho_infinito * math.cos(ang_rad_novo)
                cv2.line(frame, (int(cx - dx_n), int(cy + dy_n)), (int(cx + dx_n), int(cy - dy_n)), (0, 255, 0), 2)

        cv2.putText(frame, f"Amarelo (Base): {angulo_antigo:.1f} graus", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Verde (Restrito): {angulo_novo:.1f} graus", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "'n' Prox | 'q' Sair", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(janela_principal, frame)
        
        if imagem_tempo is not None:
            cv2.imshow('Mascara do Tempo (Raio-X)', imagem_tempo)

        tecla = cv2.waitKey(30) & 0xFF
        if tecla == ord('q'):
            sair_de_tudo = True
            break
        elif tecla == ord('n'):
            break

    cap.release()
    if sair_de_tudo:
        break

    video_atual_idx = (video_atual_idx + 1) % len(videos)

cv2.destroyAllWindows()