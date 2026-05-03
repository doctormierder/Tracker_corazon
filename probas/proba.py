import cv2
import numpy as np

cap = cv2.VideoCapture('coracao_video.mpg')
areas = [] # Lista para guardar a área de cada frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Suavização forte para remover a textura do tecido (ruído)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # 2. Threshold de Otsu (encontra automaticamente o melhor corte)
    # Como o coração é brilhante, invertemos o threshold se necessário.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Operações Morfológicas (Fechar buracos dentro da forma do coração)
    kernel = np.ones((9, 9), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 4. Encontrar contornos na imagem tratada
    contornos, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contornos:
        # Pega o maior contorno (presumivelmente o coração)
        maior_contorno = max(contornos, key=cv2.contourArea)
        
        # Calcula a área em PIXELS QUADRADOS
        area_atual = cv2.contourArea(maior_contorno)
        areas.append(area_atual)

        # Desenha o contorno EXATO (a forma dele) em vermelho no frame original
        cv2.drawContours(frame, [maior_contorno], -1, (0, 0, 255), 2)
        
        # Coloca o valor da área na tela para você acompanhar
        cv2.putText(frame, f"Area: {int(area_atual)} px", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostra o resultado final e a máscara para você ajustar
    cv2.imshow('Mascara (Processamento)', closing)
    cv2.imshow('Video Original + Contorno', frame)

    # O vídeo está rápido, um delay de 50ms ajuda a visualizar
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# No final, a lista 'areas' terá o tamanho do coração ao longo do tempo.
# Você pode plotar isso num gráfico depois para ver a sístole e diástole!