import cv2
import numpy as np
import math

def nada(x):
    pass

# 1. Configuração da Janela e Barra
janela_principal = 'Ajuste Fino do Coracao'
cv2.namedWindow(janela_principal)
cv2.createTrackbar('Sensibilidade', janela_principal, 30, 100, nada)

# 2. Array com os seus vídeos
videos = ['coracao_video.mpg', 'coracao_video1.mpg', 'coracao_video2.avi', 'coracao_video3.avi', 'coracao_video4.avi', 'coracao_video5.mpg', 'coracao_video6.mpg']
video_atual_idx = 0

while True:
    nome_video = videos[video_atual_idx]
    cap = cv2.VideoCapture(nome_video)
    
    print(f"Reproduzindo: {nome_video} | Pressione 'n' para o próximo ou 'q' para sair.")

    sair_de_tudo = False

    while True:
        ret, frame = cap.read()
        
        # Loop infinito no MESMO vídeo até apertar 'n'
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        altura_frame, largura_frame = frame.shape[:2]

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
            
            cv2.drawContours(frame, [casco_convexo], -1, (0, 0, 255), 2)
            
            # --- VERIFICAÇÃO DE TOQUE NAS BORDAS ---
            bx, by, bw, bh = cv2.boundingRect(casco_convexo)
            toca_topo = (by <= 5) # Margem de 5 pixels
            toca_chao = (by + bh >= altura_frame - 5)
            
            status_borda = "Livre"
            cor_status = (0, 255, 0)
            if toca_topo and toca_chao:
                status_borda = "TOCA TOPO E CHAO!"
                cor_status = (0, 0, 255)
            elif toca_topo:
                status_borda = "Toca Topo!"
                cor_status = (0, 165, 255)
            elif toca_chao:
                status_borda = "Toca Chao!"
                cor_status = (0, 165, 255)

            cv2.putText(frame, status_borda, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_status, 2)

            # --- DESENHO DOS EIXOS ---
            if len(casco_convexo) >= 5:
                (x, y), (eixo_menor, eixo_maior), angulo = cv2.fitEllipse(casco_convexo)
                angulo_rad = math.radians(angulo)
                
                # 1. Linha Amarela (INFINITA)
                tamanho_infinito = 2000
                dx_maior = tamanho_infinito * math.sin(angulo_rad)
                dy_maior = tamanho_infinito * math.cos(angulo_rad)
                pt1_maior = (int(x - dx_maior), int(y + dy_maior))
                pt2_maior = (int(x + dx_maior), int(y - dy_maior))
                cv2.line(frame, pt1_maior, pt2_maior, (0, 255, 255), 2)
                
                # 2. Linha Magenta (Largura)
                dx_menor = (eixo_menor / 2) * math.cos(angulo_rad)
                dy_menor = (eixo_menor / 2) * math.sin(angulo_rad)
                pt1_menor = (int(x - dx_menor), int(y - dy_menor))
                pt2_menor = (int(x + dx_menor), int(y + dy_menor))
                cv2.line(frame, pt1_menor, pt2_menor, (255, 0, 255), 2)
                
                cv2.putText(frame, f"Largura: {int(eixo_menor)} px", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Info do vídeo atual
        cv2.putText(frame, f"Video: {nome_video}", (10, altura_frame - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(janela_principal, frame)
        cv2.imshow('Como o PC ve (Mascara)', closing)

        tecla = cv2.waitKey(30) & 0xFF
        if tecla == ord('q'):
            sair_de_tudo = True
            break
        elif tecla == ord('n'):
            break # Quebra o loop interno para carregar o próximo vídeo

    cap.release()
    
    if sair_de_tudo:
        break

    # Avança no array e volta ao zero se chegar no final
    video_atual_idx += 1
    if video_atual_idx >= len(videos):
        video_atual_idx = 0

cv2.destroyAllWindows()