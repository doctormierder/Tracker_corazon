import cv2
import numpy as np
import math

def nada(x):
    pass

def achar_eixo_pelas_pontas(cap):
    print("Calculando eixo pelas âncoras (Topo e Base)... Aguarde.")
    angulos = []
    
    contador_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        contador_frames += 1
        if contador_frames % 5 != 0: continue # Processa 1 a cada 5 frames para ser rápido
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 1. MÁSCARA ESTRITA (Sua ideia de baixar a tolerância)
        branco_maximo = np.max(blurred)
        ponto_de_corte = max(0, branco_maximo - 100) # Apenas os pixels MUITO brancos passam
        _, thresh = cv2.threshold(blurred, ponto_de_corte, 255, cv2.THRESH_BINARY)
        
        # 2. LIMPEZA AGRESSIVA (Mata conexões diagonais com sujeira)
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(thresh, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=2)
        
        contornos, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contornos:
            maior_contorno = max(contornos, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(maior_contorno)
            
            # Só analisa se o objeto tiver uma altura decente
            if h > 50:
                # 3. FATIAMOS AS PONTAS DO CORAÇÃO (15% do topo e 15% da base)
                fatia_topo = dilated[y : y + int(h * 0.15), x : x + w]
                fatia_base = dilated[y + h - int(h * 0.15) : y + h, x : x + w]
                
                # Acha o centro de massa da ponta de CIMA
                M_topo = cv2.moments(fatia_topo)
                if M_topo["m00"] != 0:
                    cx_topo = x + int(M_topo["m10"] / M_topo["m00"])
                    cy_topo = y + int((h * 0.15) / 2)
                    
                    # Acha o centro de massa da ponta de BAIXO
                    M_base = cv2.moments(fatia_base)
                    if M_base["m00"] != 0:
                        cx_base = x + int(M_base["m10"] / M_base["m00"])
                        cy_base = y + h - int((h * 0.15) / 2)
                        
                        # 4. CALCULA O ÂNGULO DA LINHA QUE UNE AS PONTAS
                        dx = cx_base - cx_topo
                        dy = cy_base - cy_topo
                        
                        # atan2(dx, dy) garante que 0 graus seja uma linha perfeitamente vertical
                        angulo_rad = math.atan2(dx, dy)
                        angulos.append(math.degrees(angulo_rad))
                        
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    if angulos:
        angulo_final = np.median(angulos)
        print(f"Sucesso! Eixo travado nas pontas em: {angulo_final:.2f} graus.")
        return angulo_final
    else:
        print("Aviso: Falha ao ancorar as pontas. Usando vertical padrão.")
        return 0.0

# --- INÍCIO DA INTERFACE ---
janela_principal = 'Ajuste Fino do Coracao'
cv2.namedWindow(janela_principal)
cv2.createTrackbar('Sensibilidade', janela_principal, 30, 100, nada)

videos = ['coracao_video.mpg', 'coracao_video1.mpg', 'coracao_video2.avi', 'coracao_video3.avi', 'coracao_video4.avi', 'coracao_video5.mpg', 'coracao_video6.mpg']
video_atual_idx = 0

while True:
    nome_video = videos[video_atual_idx]
    cap = cv2.VideoCapture(nome_video)
    
    if not cap.isOpened():
        video_atual_idx = (video_atual_idx + 1) % len(videos)
        continue

    print(f"\n--- Iniciando: {nome_video} ---")
    
    # === ACHA O EIXO FIXO PELAS PONTAS ===
    angulo_rigido = achar_eixo_pelas_pontas(cap)
    angulo_rad = math.radians(angulo_rigido)

    sair_de_tudo = False

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret: break
            
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
            
            # O Centro de Massa atual do coração inteiro para posicionar o cruzamento
            M = cv2.moments(casco_convexo)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Largura estimada baseada na área e na altura da caixa delimitadora
                _, _, _, h_box = cv2.boundingRect(casco_convexo)
                area = cv2.contourArea(casco_convexo)
                largura_estimada = (area / h_box) * 1.5 if h_box > 0 else 50
                
                tamanho_infinito = 2000
                
                # 1. Linha Amarela (Comprimento Infinito, imune a "achatamentos")
                dx_maior = tamanho_infinito * math.sin(angulo_rad)
                dy_maior = tamanho_infinito * math.cos(angulo_rad)
                pt1_maior = (int(cx - dx_maior), int(cy - dy_maior)) # Ajuste de sinal
                pt2_maior = (int(cx + dx_maior), int(cy + dy_maior)) # Ajuste de sinal
                cv2.line(frame, pt1_maior, pt2_maior, (0, 255, 255), 2)
                
                # 2. Linha Magenta (Largura Transversal)
                dx_menor = (largura_estimada / 2) * math.cos(angulo_rad)
                dy_menor = (largura_estimada / 2) * math.sin(angulo_rad)
                pt1_menor = (int(cx - dx_menor), int(cy + dy_menor)) # Ajuste de sinal
                pt2_menor = (int(cx + dx_menor), int(cy - dy_menor)) # Ajuste de sinal
                cv2.line(frame, pt1_menor, pt2_menor, (255, 0, 255), 2)

        cv2.imshow(janela_principal, frame)
        cv2.imshow('Como o PC ve (Mascara)', closing)

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