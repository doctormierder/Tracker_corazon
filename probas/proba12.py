import cv2
import numpy as np
import math

def nada(x):
    pass

def viagem_no_tempo_dupla(cap):
    """
    Calcula o Eixo Antigo (Elipse) e o Eixo Novo (Máscara do Tempo) simultaneamente.
    """
    print("Iniciando Viagem no Tempo... Calculando Eixos. Aguarde.")
    angulos_antigos = []
    
    # Prepara uma tela preta vazia para acumular as máscaras (formato float para não estourar o limite de cor)
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
            
            # --- 1. A MÁSCARA DO TEMPO (Soma de todos os frames e sensibilidades) ---
            # Adiciona os pixels brancos atuais no acumulador geral
            acumulador += (closing / 255.0)
            
            # --- 2. O CÁLCULO ANTIGO (Elipse) ---
            contornos, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contornos:
                maior_contorno = max(contornos, key=cv2.contourArea)
                casco_convexo = cv2.convexHull(maior_contorno)
                if len(casco_convexo) >= 5:
                    _, eixos, angulo = cv2.fitEllipse(casco_convexo)
                    eixo_a, eixo_b = eixos
                    eixo_min, eixo_max = min(eixo_a, eixo_b), max(eixo_a, eixo_b)
                    
                    if eixo_min > 0:
                        relacao = eixo_max / eixo_min
                        if relacao > 1.3:
                            peso = int(relacao * 10) 
                            angulos_antigos.extend([angulo] * peso)
                            
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # --- FECHAMENTO DO EIXO ANTIGO ---
    angulo_antigo_final = np.median(angulos_antigos) if angulos_antigos else 90.0

    # --- FECHAMENTO DO EIXO NOVO (MÁSCARA DO TEMPO) ---
    # Normaliza o acumulador para virar uma imagem visível (0 a 255)
    cv2.normalize(acumulador, acumulador, 0, 255, cv2.NORM_MINMAX)
    acumulador_img = np.uint8(acumulador)
    
    # Corta apenas o "Núcleo Duro": pixels que apareceram em pelo menos 60% do tempo (150 de 255)
    _, mascara_tempo_bin = cv2.threshold(acumulador_img, 150, 255, cv2.THRESH_BINARY)
    
    # Aplica a matemática (PCA/fitLine) em cima do Núcleo Duro
    pontos_y, pontos_x = np.nonzero(mascara_tempo_bin)
    if len(pontos_x) > 50: # Se sobrou massa suficiente
        pontos = np.column_stack((pontos_x, pontos_y)).astype(np.float32)
        # fitLine acha a reta central perfeita ignorando as pontas soltas
        [vx, vy, x0, y0] = cv2.fitLine(pontos, cv2.DIST_L2, 0, 0.01, 0.01)
        angulo_novo_rad = np.atan2(vx, vy)
        angulo_novo_final = np.degrees(angulo_novo_rad)
        if angulo_novo_final < 0: angulo_novo_final += 180
    else:
        angulo_novo_final = 90.0
        
    print(f"-> Eixo Antigo (Amarelo): {angulo_antigo_final.item():.2f} graus")
    print(f"-> Eixo Novo   (Verde)  : {angulo_novo_final.item():.2f} graus")
    
    return angulo_antigo_final, angulo_novo_final, mascara_tempo_bin

# ==========================================
# CONFIGURAÇÃO DA INTERFACE
# ==========================================
janela_principal = 'Duelo de Eixos'
cv2.namedWindow(janela_principal)
cv2.createTrackbar('Sensibilidade', janela_principal, 30, 100, nada)

videos = ['coracao_video.mpg', 'coracao_video1.mpg', 'coracao_video2.avi', 'coracao_video3.avi', 'coracao_video4.avi', 'coracao_video5.mpg', 'coracao_video6.mpg']
video_atual_idx = 0

while True:
    nome_video = videos[video_atual_idx]
    cap = cv2.VideoCapture(nome_video)
    print(f"\n--- Iniciando: {nome_video} ---")
    
    # Roda a nossa função dupla
    angulo_antigo, angulo_novo, imagem_tempo = viagem_no_tempo_dupla(cap)
    
    ang_rad_antigo = np.radians(angulo_antigo)
    ang_rad_novo = np.radians(angulo_novo)

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
                
                # --- DESENHA O EIXO ANTIGO (AMARELO) ---
                dx_a = tamanho_infinito * np.sin(ang_rad_antigo)
                dy_a = tamanho_infinito * np.cos(ang_rad_antigo)
                cv2.line(frame, (int(cx - dx_a), int(cy + dy_a)), (int(cx + dx_a), int(cy - dy_a)), (0, 255, 255), 2)
                
                # --- DESENHA O EIXO NOVO (VERDE) ---
                dx_n = tamanho_infinito * np.sin(ang_rad_novo)
                dy_n = tamanho_infinito * np.cos(ang_rad_novo)
                cv2.line(frame, (int((cx - dx_n).item()), int((cy + dy_n).item())), (int((cx + dx_n).item()), int((cy - dy_n).item())), (0, 255, 0), 2)

        # Textos na tela para você saber quem é quem
        cv2.putText(frame, f"Amarelo (Antigo): {angulo_antigo:.1f} graus", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Verde (Novo): {angulo_novo.item():.1f} graus", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "'n' Prox | 'q' Sair", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(janela_principal, frame)
        
        # Mostra o Raio-X que gerou a linha Verde!
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