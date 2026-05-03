import cv2
import numpy as np
import math

def nada(x):
    pass

# Nova playlist de vídeos
videos = [
    'coracao_video.mpg', 'coracao_video1.mpg', 'coracao_video2.avi', 
    'coracao_video3.avi', 'coracao_video4.avi', 'coracao_video5.mpg', 'coracao_video6.mpg'
]
video_atual_idx = 0

# Configuração da Interface
janela_principal = 'Analise do Coracao (Brilho + Movimento)'
cv2.namedWindow(janela_principal)
# O limiar de brilho para a Fase 2
cv2.createTrackbar('Sensibilidade Brilho', janela_principal, 15, 100, nada)

# Inicializa o Subtrator de Fundo (A "Câmera de Segurança" da Fase 1)
# history=500 frames na memória, detectShadows=False (não queremos tons de cinza)
subtrator = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

fator_atrito = 0.15 # Amortecedor do eixo (quanto menor, mais suave/pesado)

while True:
    nome_video = videos[video_atual_idx]
    cap = cv2.VideoCapture(nome_video)
    
    if not cap.isOpened():
        print(f"Erro: Não encontrei {nome_video}. Pulando...")
        video_atual_idx = (video_atual_idx + 1) % len(videos)
        continue

    print(f"\n--- Iniciando Segmentação Cinética: {nome_video} ---")
    
    # Variáveis de memória para a Inércia (Fase 5)
    cx_suave, cy_suave, angulo_suave, largura_suave = None, None, None, None
    sair_de_tudo = False

    while True:
        ret, frame = cap.read()
        if not ret:
            # Rebobina se acabar
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret: break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # =================================================================
        # FASE 1: A MÁSCARA DE MOVIMENTO (Ignora sujeira parada)
        # =================================================================
        # Aprende o que é fundo estático e destaca apenas o que está pulsando.
        # LearningRate=-1 faz o algoritmo adaptar a velocidade de aprendizado sozinho.
        mask_movimento = subtrator.apply(blurred, learningRate=-1)
        
        # Garante que seja estritamente binária (preto ou branco)
        _, mask_movimento = cv2.threshold(mask_movimento, 200, 255, cv2.THRESH_BINARY)

        # =================================================================
        # FASE 2: A MÁSCARA DE BRILHO (Ignora sombras em movimento)
        # =================================================================
        sensibilidade = cv2.getTrackbarPos('Sensibilidade Brilho', janela_principal)
        branco_referencia = np.percentile(blurred, 95)
        ponto_de_corte = max(0, branco_referencia - sensibilidade)
        
        _, mask_brilho = cv2.threshold(blurred, ponto_de_corte, 255, cv2.THRESH_BINARY)

        # =================================================================
        # FASE 3: O CASAMENTO (A Guilhotina Lógica "AND")
        # =================================================================
        # Só sobrevive quem for BRILHANTE E ESTIVER SE MOVENDO
        mask_casamento = cv2.bitwise_and(mask_brilho, mask_movimento)

        # =================================================================
        # FASE 4: REBOCO E LIXA (Morfologia)
        # =================================================================
        kernel_dilatar = np.ones((7, 7), np.uint8)
        kernel_fechar = np.ones((11, 11), np.uint8)
        
        # Infla os pontos brancos sobreviventes para que se toquem
        dilated = cv2.dilate(mask_casamento, kernel_dilatar, iterations=1)
        # Preenche os buracos pretos no meio da massa branca
        mask_limpa = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_fechar)

        # =================================================================
        # FASE 5: EIXO DE SIMETRIA (O Retorno)
        # =================================================================
        contornos, _ = cv2.findContours(mask_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contornos:
            maior_contorno = max(contornos, key=cv2.contourArea)
            
            # Filtro para ignorar ruídos mínimos que sobraram
            if cv2.contourArea(maior_contorno) > 500:
                casco = cv2.convexHull(maior_contorno)
                cv2.drawContours(frame, [casco], -1, (0, 0, 255), 2)
                
                M = cv2.moments(casco)
                if M["m00"] != 0:
                    cx_bruto = M["m10"] / M["m00"]
                    cy_bruto = M["m01"] / M["m00"]
                    
                    _, _, w_box, h_box = cv2.boundingRect(casco)
                    largura_bruta = w_box * 0.8 
                    
                    # Tenta ajustar a elipse para pegar a inclinação do músculo
                    if len(casco) >= 5:
                        _, _, angulo_bruto = cv2.fitEllipse(casco)
                    else:
                        angulo_bruto = 90.0

                    # --- AMORTECEDOR (Inércia) ---
                    if cx_suave is None:
                        cx_suave, cy_suave = cx_bruto, cy_bruto
                        angulo_suave = angulo_bruto
                        largura_suave = largura_bruta
                    else:
                        cx_suave = (fator_atrito * cx_bruto) + ((1 - fator_atrito) * cx_suave)
                        cy_suave = (fator_atrito * cy_bruto) + ((1 - fator_atrito) * cy_suave)
                        largura_suave = (fator_atrito * largura_bruta) + ((1 - fator_atrito) * largura_suave)
                        
                        # Evita que o ângulo dê um pulo de 180 graus de cabeça para baixo
                        diff_angulo = angulo_bruto - angulo_suave
                        if diff_angulo > 90: diff_angulo -= 180
                        elif diff_angulo < -90: diff_angulo += 180
                        angulo_suave += fator_atrito * diff_angulo

                    # DESENHO DA CRUZ
                    angulo_rad = math.radians(angulo_suave)
                    tamanho_infinito = 2000
                    
                    # Eixo Longitudinal (Linha Amarela)
                    dx_maior = tamanho_infinito * math.sin(angulo_rad)
                    dy_maior = tamanho_infinito * math.cos(angulo_rad)
                    pt1_maior = (int(cx_suave - dx_maior), int(cy_suave + dy_maior))
                    pt2_maior = (int(cx_suave + dx_maior), int(cy_suave - dy_maior))
                    cv2.line(frame, pt1_maior, pt2_maior, (0, 255, 255), 2)
                    
                    # Eixo Transversal (Linha Magenta)
                    dx_menor = (largura_suave / 2) * math.cos(angulo_rad)
                    dy_menor = (largura_suave / 2) * math.sin(angulo_rad)
                    pt1_menor = (int(cx_suave - dx_menor), int(cy_suave - dy_menor))
                    pt2_menor = (int(cx_suave + dx_menor), int(cy_suave + dy_menor))
                    cv2.line(frame, pt1_menor, pt2_menor, (255, 0, 255), 2)

        # Interface com as duas janelas cruciais
        cv2.imshow(janela_principal, frame)
        # Mostra o cérebro da operação: a máscara já cruzada com movimento e curada
        cv2.imshow('Mascara (Movimento + Brilho)', mask_limpa)

        tecla = cv2.waitKey(30) & 0xFF
        if tecla == ord('q'):
            sair_de_tudo = True
            break
        elif tecla == ord('n'):
            break

    cap.release()
    if sair_de_tudo: break

    video_atual_idx = (video_atual_idx + 1) % len(videos)

cv2.destroyAllWindows()