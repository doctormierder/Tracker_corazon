import cv2
import numpy as np

def nada(x):
    pass # Função vazia exigida pelo OpenCV para criar o Trackbar

# 1. Configuração da Janela e Barra de Ajuste
janela_principal = 'Ajuste Fino do Coracao'
cv2.namedWindow(janela_principal)

# Cria a barra 'Sensibilidade'. Vai de 0 a 100. 
# Valor inicial sugerido: 30.
cv2.createTrackbar('Sensibilidade', janela_principal, 30, 100, nada)

# Carrega o vídeo
cap = cv2.VideoCapture('coracao_video.mpg')

while True:
    ret, frame = cap.read()
    
    # 2. Loop Infinito: Se o vídeo acabar, volta para o primeiro frame
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Lógica do "Branco Mais Claro"
    # Descobre o valor máximo de brilho neste frame específico (0 a 255)
    branco_maximo = np.max(blurred)

    # Lê o valor atual da barra deslizante
    sensibilidade = cv2.getTrackbarPos('Sensibilidade', janela_principal)

    # Calcula o ponto de corte dinâmico. 
    # Ex: Se o brilho máximo for 250 e a sensibilidade 30, o corte é 220.
    # Apenas pixels entre 220 e 250 serão considerados "coração".
    ponto_de_corte = max(0, branco_maximo - sensibilidade)

    # Aplica o threshold com o nosso valor calculado
    _, thresh = cv2.threshold(blurred, ponto_de_corte, 255, cv2.THRESH_BINARY)

    # 4. Limpeza para remover pontinhos isolados
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 5. Desenhar o Contorno
    contornos, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contornos:
        maior_contorno = max(contornos, key=cv2.contourArea)
        area = cv2.contourArea(maior_contorno)
        
        cv2.drawContours(frame, [maior_contorno], -1, (0, 0, 255), 2)
        cv2.putText(frame, f"Area: {int(area)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Exibe as janelas
    cv2.imshow(janela_principal, frame)
    cv2.imshow('Como o PC ve (Mascara)', closing)

    # Aperte 'q' no teclado para sair
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()