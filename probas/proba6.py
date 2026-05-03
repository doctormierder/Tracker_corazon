import cv2
import numpy as np
import math

def nada(x):
    pass

def calcular_angulo_rigido(cap):
    """
    Lê os frames manualmente (para evitar o bug do frame_count do OpenCV)
    e encontra o eixo dominante com base nas formas mais compridas.
    """
    print("Calculando Fator de Rigidez (Mediana Ponderada)... Aguarde.")
    angulos_pesados = []
    angulos_ligeros = []
    
    contador_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break # Chegou ao fim do vídeo real
            
        contador_frames += 1
        
        # Pula frames para processar mais rápido (analisa 1 a cada 10 frames)
        if contador_frames % 10 != 0:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        branco_maximo = np.max(blurred)
        
        for sensibilidade in range(10, 90, 5):
            ponto_de_corte = max(0, branco_maximo - sensibilidade)
            _, thresh = cv2.threshold(blurred, ponto_de_corte, 255, cv2.THRESH_BINARY)
            
            kernel = np.ones((5, 5), np.uint8)
            closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contornos, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contornos:
                maior_contorno = max(contornos, key=cv2.contourArea)
                casco_convexo = cv2.convexHull(maior_contorno)
                
                if len(casco_convexo) >= 5:
                    _, (eixo_a, eixo_b), angulo = cv2.fitEllipse(casco_convexo)
                    
                    eixo_min = min(eixo_a, eixo_b)
                    eixo_max = max(eixo_a, eixo_b)
                    
                    if eixo_min > 0:
                        angulos_ligeros.extend([angulo] * 1)
                        relacao = eixo_max / eixo_min
                        if relacao > 1.7:
                            peso1 = int((relacao**4) * 10) 
                            angulos_pesados.extend([angulo] * peso1)
                            
                            
    # IMPORTANTE: Rebobina o vídeo de volta para o frame 0 depois da análise
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    
    if angulos_pesados:
        angulo_final = np.median(angulos_pesados)
        angulo_final_ligero = np.median(angulos_ligeros)
        print(f"Sucesso! Eixo Rígido travado em: {angulo_final:.2f} graus.")
        return angulo_final, angulo_final_ligero
    else:
        print("Aviso: Formas muito irregulares. Usando 90 graus por padrão.")
        return 90.0

# 1. Configuração da Janela e Barra
janela_principal = 'Ajuste Fino do Coracao'
cv2.namedWindow(janela_principal)
cv2.createTrackbar('Sensibilidade', janela_principal, 30, 100, nada)

# 2. Playlist
videos = ['coracao_video.mpg', 'coracao_video1.mpg', 'coracao_video2.avi', 'coracao_video3.avi', 'coracao_video4.avi', 'coracao_video5.mpg', 'coracao_video6.mpg']
video_atual_idx = 0

while True:
    nome_video = videos[video_atual_idx]
    cap = cv2.VideoCapture(nome_video)
    
    # Trava de Segurança 1: Checa se o vídeo realmente abriu
    if not cap.isOpened():
        print(f"\nERRO: Não encontrei o vídeo '{nome_video}'. Verifique o nome ou pasta.")
        video_atual_idx = (video_atual_idx + 1) % len(videos)
        cv2.waitKey(1500) # Espera 1.5s antes de tentar o próximo
        continue

    print(f"\n--- Iniciando: {nome_video} ---")
    
    angulo_rigido, angulo_rigido1 = calcular_angulo_rigido(cap)
    angulo_rad = math.radians(angulo_rigido)
    angulo_rad1 = math.radians(angulo_rigido1)

    sair_de_tudo = False

    while True:
        ret, frame = cap.read()
        
        # Trava de Segurança 2: Impede o loop infinito invisível
        if not ret:
            # Rebobina
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # Lê de novo para garantir que não estamos num loop travado no frame 0
            ret, frame = cap.read()
            if not ret:
                print("Erro crítico ao tentar rebobinar o vídeo. Pulando para o próximo.")
                break
            
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
            
            M = cv2.moments(casco_convexo)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                if len(casco_convexo) >= 5:
                    _, (eixo_a, eixo_b), _ = cv2.fitEllipse(casco_convexo)
                    largura = min(eixo_a, eixo_b)
                else:
                    largura = 50 
                
                #linha amarela
                tamanho_infinito = 2000
                dx_maior = tamanho_infinito * math.sin(angulo_rad)
                dy_maior = tamanho_infinito * math.cos(angulo_rad)
                pt1_maior = (int(cx - dx_maior), int(cy + dy_maior))
                pt2_maior = (int(cx + dx_maior), int(cy - dy_maior))
                cv2.line(frame, pt1_maior, pt2_maior, (0, 255, 255), 2)

                #linha verde
                tamanho_infinito = 2000
                dx_maior = tamanho_infinito * math.sin(angulo_rad1)
                dy_maior = tamanho_infinito * math.cos(angulo_rad1)
                pt1_maior = (int(cx - dx_maior), int(cy + dy_maior))
                pt2_maior = (int(cx + dx_maior), int(cy - dy_maior))
                cv2.line(frame, pt1_maior, pt2_maior, (0, 255, 0), 2)
                
                #linha roxo
                dx_menor = (largura / 2) * math.cos(angulo_rad)
                dy_menor = (largura / 2) * math.sin(angulo_rad)
                pt1_menor = (int(cx - dx_menor), int(cy - dy_menor))
                pt2_menor = (int(cx + dx_menor), int(cy + dy_menor))
                cv2.line(frame, pt1_menor, pt2_menor, (255, 0, 255), 2)
                
                cv2.putText(frame, f"Largura: {int(largura)} px", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        cv2.putText(frame, f"Eixo: {angulo_rigido:.1f} | 'n' Prox | 'q' Sair", 
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(janela_principal, frame)
        cv2.imshow('Como o PC ve (Mascara)', closing)

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