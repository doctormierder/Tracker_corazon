import cv2
import numpy as np
import math

def nada(x):
    pass

def calcular_angulo_rigido(cap, bbox):
    """
    Viaja pelos frames e sensibilidades para encontrar o ângulo real do coração,
    restringindo a busca apenas à Região de Interesse (ROI).
    """
    print("Calculando Fator de Rigidez (Mediana Ponderada)... Aguarde.")
    angulos_pesados = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // 30)
    
    x_roi, y_roi, w_roi, h_roi = bbox
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue
            
        # 1. Aplicar o recorte da ROI
        if w_roi > 0 and h_roi > 0:
            roi = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        else:
            roi = frame
            
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 2. Aplicar CLAHE para equilibrar o brilho (centro queimado)
        gray_clahe = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray_clahe, (5, 5), 0)
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
                    _, eixos, angulo = cv2.fitEllipse(casco_convexo)
                    eixo_a, eixo_b = eixos
                    
                    eixo_min = min(eixo_a, eixo_b)
                    eixo_max = max(eixo_a, eixo_b)
                    
                    if eixo_min > 0:
                        relacao = eixo_max / eixo_min
                        if relacao > 1.3:
                            peso = int(relacao * 10) 
                            angulos_pesados.extend([angulo] * peso)
                            
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    if angulos_pesados:
        angulo_final = np.median(angulos_pesados)
        print(f"Eixo Rígido travado em: {angulo_final:.2f} graus.")
        return angulo_final
    else:
        print("Aviso: Não foi possível calcular um eixo claro. Usando 90 graus por padrão.")
        return 90.0

# --- Configuração Principal ---
janela_principal = 'Ajuste Fino do Coracao'
cv2.namedWindow(janela_principal)
cv2.createTrackbar('Sensibilidade', janela_principal, 30, 100, nada)

videos = ['coracao_video.mpg', 'coracao_video1.mpg', 'coracao_video2.avi', 'coracao_video3.avi', 'coracao_video4.avi', 'coracao_video5.mpg', 'coracao_video6.mpg']
video_atual_idx = 0

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

while True:
    nome_video = videos[video_atual_idx]
    cap = cv2.VideoCapture(nome_video)
    print(f"\n--- Iniciando: {nome_video} ---")
    
    # Ler o primeiro frame para o usuário desenhar a ROI
    ret, frame_inicial = cap.read()
    if not ret:
        print(f"Erro ao abrir {nome_video}. Pulando...")
        video_atual_idx = (video_atual_idx + 1) % len(videos)
        continue

    print("=> SELECIONE A ÁREA DO CORAÇÃO desenhando um retângulo.")
    print("=> Pressione ESPAÇO ou ENTER para confirmar a seleção.")
    bbox = cv2.selectROI('Selecione a ROI (Espaco para confirmar)', frame_inicial, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow('Selecione a ROI (Espaco para confirmar)')
    x_roi, y_roi, w_roi, h_roi = bbox
    
    angulo_rigido = calcular_angulo_rigido(cap, bbox)
    angulo_rad = math.radians(angulo_rigido)

    sair_de_tudo = False

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Isolar a ROI para processamento (Ignora lixo externo)
        if w_roi > 0 and h_roi > 0:
            roi_frame = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        else:
            roi_frame = frame.copy()
            x_roi, y_roi = 0, 0

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray_clahe = clahe.apply(gray) # Corrige a luz
        blurred = cv2.GaussianBlur(gray_clahe, (5, 5), 0)

        branco_maximo = np.max(blurred)
        sensibilidade = cv2.getTrackbarPos('Sensibilidade', janela_principal)
        ponto_de_corte = max(0, branco_maximo - sensibilidade)

        _, thresh = cv2.threshold(blurred, ponto_de_corte, 255, cv2.THRESH_BINARY)
        
        # Kernel ligeiramente maior para garantir o fechamento de buracos de luz
        kernel = np.ones((7, 7), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contornos:
            maior_contorno = max(contornos, key=cv2.contourArea)
            casco_convexo = cv2.convexHull(maior_contorno)
            
            M = cv2.moments(casco_convexo)
            if M["m00"] != 0:
                # O centro é calculado na ROI, então somamos x_roi e y_roi para mapear para o frame original
                cx = int(M["m10"] / M["m00"]) + x_roi
                cy = int(M["m01"] / M["m00"]) + y_roi
                
                if len(casco_convexo) >= 5:
                    _, (eixo_a, eixo_b), _ = cv2.fitEllipse(casco_convexo)
                    largura = min(eixo_a, eixo_b)
                else:
                    largura = 50 
                
                # Desenhando o contorno ajustado no frame principal
                # Precisamos deslocar os pontos do contorno para a posição correta
                casco_deslocado = casco_convexo + [x_roi, y_roi]
                cv2.drawContours(frame, [casco_deslocado], -1, (0, 0, 255), 2)
                
                # 1. Linha de Simetria (Amarela)
                tamanho_infinito = 2000
                dx_maior = tamanho_infinito * math.sin(angulo_rad)
                dy_maior = tamanho_infinito * math.cos(angulo_rad)
                pt1_maior = (int(cx - dx_maior), int(cy + dy_maior))
                pt2_maior = (int(cx + dx_maior), int(cy - dy_maior))
                cv2.line(frame, pt1_maior, pt2_maior, (0, 255, 255), 2)
                
                # 2. Linha da Largura (Magenta)
                dx_menor = (largura / 2) * math.cos(angulo_rad)
                dy_menor = (largura / 2) * math.sin(angulo_rad)
                pt1_menor = (int(cx - dx_menor), int(cy - dy_menor))
                pt2_menor = (int(cx + dx_menor), int(cy + dy_menor))
                cv2.line(frame, pt1_menor, pt2_menor, (255, 0, 255), 2)
                
                cv2.putText(frame, f"Largura: {int(largura)} px", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Desenhar a caixa da ROI para referência visual
        if w_roi > 0 and h_roi > 0:
            cv2.rectangle(frame, (x_roi, y_roi), (x_roi+w_roi, y_roi+h_roi), (255, 255, 0), 1)

        cv2.putText(frame, f"Angulo Travado: {angulo_rigido:.1f} | 'n' Prox | 'q' Sair", 
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(janela_principal, frame)
        cv2.imshow('Como o PC ve (Mascara na ROI)', closing)

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