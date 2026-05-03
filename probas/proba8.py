import cv2
import numpy as np
import math

def nada(x):
    pass

janela_principal = 'Ajuste Fino do Coracao'
cv2.namedWindow(janela_principal)
cv2.createTrackbar('Sensibilidade', janela_principal, 30, 100, nada)

# ATRITO (EMA): Quanto menor o valor (ex: 0.1), mais "pesado/lento" e suave é o movimento.
# Um valor de 1.0 significa sem atrito nenhum (tremedeira total).
fator_atrito = 0.15 

videos = ['coracao_video.mpg', 'coracao_video1.mpg', 'coracao_video2.avi', 'coracao_video3.avi', 'coracao_video4.avi', 'coracao_video5.mpg', 'coracao_video6.mpg']
video_atual_idx = 0

while True:
    nome_video = videos[video_atual_idx]
    cap = cv2.VideoCapture(nome_video)
    
    if not cap.isOpened():
        video_atual_idx = (video_atual_idx + 1) % len(videos)
        continue

    print(f"\n--- Iniciando: {nome_video} ---")
    sair_de_tudo = False

    # Variáveis de Memória para o Amortecedor
    cx_suave, cy_suave = None, None
    angulo_suave = None
    largura_suave = None

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret: break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 1. MÁSCARA MELHORADA: Ignorando sujeiras extremas com Percentil 95
        # Em vez do pixel mais brilhante absoluto, pega o "brilho da massa principal"
        branco_referencia = np.percentile(blurred, 95) 
        
        sensibilidade = cv2.getTrackbarPos('Sensibilidade', janela_principal)
        ponto_de_corte = max(0, branco_referencia - sensibilidade)

        _, thresh = cv2.threshold(blurred, ponto_de_corte, 255, cv2.THRESH_BINARY)
        
        # Limpeza morfológica pesada para grudar o coração
        kernel = np.ones((7, 7), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contornos:
            # Pega sempre a maior massa branca
            maior_contorno = max(contornos, key=cv2.contourArea)
            casco_convexo = cv2.convexHull(maior_contorno)
            
            cv2.drawContours(frame, [casco_convexo], -1, (0, 0, 255), 2)
            
            M = cv2.moments(casco_convexo)
            if M["m00"] != 0:
                # Centro e Largura brutos (que tremem muito)
                cx_bruto = M["m10"] / M["m00"]
                cy_bruto = M["m01"] / M["m00"]
                
                _, _, w_box, h_box = cv2.boundingRect(casco_convexo)
                area = cv2.contourArea(casco_convexo)
                largura_bruta = (area / h_box) * 1.5 if h_box > 0 else 50
                
                # Para achar o eixo sem tremer, tentamos ajustar uma elipse, 
                # mas vamos amortecer o resultado violentamente
                if len(casco_convexo) >= 5:
                    _, _, angulo_bruto = cv2.fitEllipse(casco_convexo)
                else:
                    angulo_bruto = 90.0

                # 2. O AMORTECEDOR (Filtro EMA) Entra em Ação
                if cx_suave is None:
                    # No primeiro frame, ele simplesmente assume o valor bruto
                    cx_suave, cy_suave = cx_bruto, cy_bruto
                    angulo_suave = angulo_bruto
                    largura_suave = largura_bruta
                else:
                    # Nos frames seguintes, ele mistura com atrito
                    cx_suave = (fator_atrito * cx_bruto) + ((1 - fator_atrito) * cx_suave)
                    cy_suave = (fator_atrito * cy_bruto) + ((1 - fator_atrito) * cy_suave)
                    largura_suave = (fator_atrito * largura_bruta) + ((1 - fator_atrito) * largura_suave)
                    
                    # Cuidado extra com o ângulo por causa do limite de 0 a 180 graus
                    diff_angulo = angulo_bruto - angulo_suave
                    if diff_angulo > 90: diff_angulo -= 180
                    elif diff_angulo < -90: diff_angulo += 180
                    angulo_suave += fator_atrito * diff_angulo

                # DESENHA AS LINHAS USANDO OS VALORES SUAVIZADOS
                angulo_rad = math.radians(angulo_suave)
                tamanho_infinito = 2000
                
                # Linha Amarela (Desliza suavemente)
                dx_maior = tamanho_infinito * math.sin(angulo_rad)
                dy_maior = tamanho_infinito * math.cos(angulo_rad)
                pt1_maior = (int(cx_suave - dx_maior), int(cy_suave + dy_maior))
                pt2_maior = (int(cx_suave + dx_maior), int(cy_suave - dy_maior))
                cv2.line(frame, pt1_maior, pt2_maior, (0, 255, 255), 2)
                
                # Linha Magenta (Bate suavemente)
                dx_menor = (largura_suave / 2) * math.cos(angulo_rad)
                dy_menor = (largura_suave / 2) * math.sin(angulo_rad)
                pt1_menor = (int(cx_suave - dx_menor), int(cy_suave - dy_menor))
                pt2_menor = (int(cx_suave + dx_menor), int(cy_suave + dy_menor))
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