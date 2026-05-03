import cv2
import numpy as np
import math

def nada(x):
    pass # Função vazia exigida pelo OpenCV para criar o Trackbar

# 1. Configuração da Janela e Barra de Ajuste
janela_principal = 'Ajuste Fino do Coracao'
cv2.namedWindow(janela_principal)
cv2.createTrackbar('Sensibilidade', janela_principal, 30, 100, nada)

# Carrega o vídeo (lembre-se de checar o nome exato do arquivo)
cap = cv2.VideoCapture('coracao_video2.avi')

while True:
    ret, frame = cap.read()
    
    # 2. Loop Infinito
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Lógica do "Branco Mais Claro"
    branco_maximo = np.max(blurred)
    sensibilidade = cv2.getTrackbarPos('Sensibilidade', janela_principal)
    ponto_de_corte = max(0, branco_maximo - sensibilidade)

    _, thresh = cv2.threshold(blurred, ponto_de_corte, 255, cv2.THRESH_BINARY)

    # 4. Limpeza
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 5. Desenhar o Contorno
    contornos, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contornos:
        maior_contorno = max(contornos, key=cv2.contourArea)
        
        # --- A MÁGICA ENTRA AQUI ---
        
        # A. Forçar a convexidade (elimina as verrugas e apêndices)
        casco_convexo = cv2.convexHull(maior_contorno)
        
        # Desenha a borda do coração limpo em vermelho
        cv2.drawContours(frame, [casco_convexo], -1, (0, 0, 255), 2)
        
        # B. Encontrar os Eixos (precisa de pelo menos 5 pontos para a matemática da elipse funcionar)
        if len(casco_convexo) >= 5:
            # Pega o centro, os tamanhos dos eixos e o grau de inclinação
            (x, y), (eixo_menor, eixo_maior), angulo = cv2.fitEllipse(casco_convexo)
            
            # CÁLCULO TRIGONOMÉTRICO PARA DESENHAR AS LINHAS
            angulo_rad = math.radians(angulo)
            
            # 1. Linha de Simetria (Eixo Maior - Amarela)
            dx_maior = (eixo_maior / 2) * math.sin(angulo_rad)
            dy_maior = (eixo_maior / 2) * math.cos(angulo_rad)
            pt1_maior = (int(x - dx_maior), int(y + dy_maior))
            pt2_maior = (int(x + dx_maior), int(y - dy_maior))
            cv2.line(frame, pt1_maior, pt2_maior, (0, 255, 255), 2)
            
            # 2. Linha da Largura (Eixo Menor perpendicular - Magenta)
            dx_menor = (eixo_menor / 2) * math.cos(angulo_rad)
            dy_menor = (eixo_menor / 2) * math.sin(angulo_rad)
            pt1_menor = (int(x - dx_menor), int(y - dy_menor))
            pt2_menor = (int(x + dx_menor), int(y + dy_menor))
            cv2.line(frame, pt1_menor, pt2_menor, (255, 0, 255), 2)
            
            # Mostra a largura do coração no frame
            cv2.putText(frame, f"Largura: {int(eixo_menor)} px", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # Exibe as janelas
    cv2.imshow(janela_principal, frame)
    cv2.imshow('Como o PC ve (Mascara)', closing)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()