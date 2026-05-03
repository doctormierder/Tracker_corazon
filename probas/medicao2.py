import cv2
import numpy as np
import math

def nada(x):
    pass

# 1. Carregar frames para a memória (acelera o pré-processamento)
print("Carregando vídeo e pré-processando... Aguarde um momento.")
cap = cv2.VideoCapture('coracao_video2.avi') # Verifique o nome do seu vídeo
frames = []
frames_borrados = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    frames_borrados.append(blurred)
cap.release()
print(f"{len(frames)} frames carregados com sucesso.")

# 2. Pré-calcular as medianas para cada sensibilidade (0 a 100)
print("Calculando o eixo fixo para cada sensibilidade... Isso leva alguns segundos.")
dados_sensibilidade = {}

for sensibilidade in range(101):
    angulos = []
    centros_x = []
    centros_y = []
    comprimentos = []
    
    for blurred in frames_borrados:
        branco_maximo = np.max(blurred)
        ponto_de_corte = max(0, branco_maximo - sensibilidade)
        _, thresh = cv2.threshold(blurred, ponto_de_corte, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contornos, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contornos:
            maior_contorno = max(contornos, key=cv2.contourArea)
            casco_convexo = cv2.convexHull(maior_contorno)
            
            # Precisamos de 5 pontos para a elipse
            if len(casco_convexo) >= 5:
                (x, y), (eixo_menor, eixo_maior), angulo = cv2.fitEllipse(casco_convexo)
                angulos.append(angulo)
                centros_x.append(x)
                centros_y.append(y)
                comprimentos.append(eixo_maior)
    
    # Salva a mediana se encontrou o coração neste nível de sensibilidade
    if angulos:
        dados_sensibilidade[sensibilidade] = {
            'x': np.median(centros_x),
            'y': np.median(centros_y),
            'angulo': np.median(angulos),
            'comprimento': np.median(comprimentos)
        }
    else:
        dados_sensibilidade[sensibilidade] = None

print("Cálculos concluídos! Abrindo janela interativa...")

# 3. Interface Interativa "Ao Olho"
janela_principal = 'Encontrar Eixo Real (Linha Fixa)'
cv2.namedWindow(janela_principal)
cv2.createTrackbar('Sensibilidade', janela_principal, 30, 100, nada)

frame_idx = 0

while True:
    # Cria uma cópia do frame atual para desenhar por cima
    frame_original = frames[frame_idx].copy()
    
    # Lê a barra de sensibilidade
    sens = cv2.getTrackbarPos('Sensibilidade', janela_principal)
    
    # Busca os dados fixos que calculamos lá em cima
    dados = dados_sensibilidade.get(sens)
    
    if dados is not None:
        x = dados['x']
        y = dados['y']
        angulo = dados['angulo']
        comprimento = dados['comprimento']
        
        # Matemática para desenhar a linha baseada no ângulo central
        angulo_rad = math.radians(angulo)
        dx = (comprimento / 2) * math.sin(angulo_rad)
        dy = (comprimento / 2) * math.cos(angulo_rad)
        
        pt1 = (int(x - dx), int(y + dy))
        pt2 = (int(x + dx), int(y - dy))
        
        # Desenha a linha Fixa (Amarela)
        cv2.line(frame_original, pt1, pt2, (0, 255, 255), 2)
        
        # Desenha um ponto vermelho no centro exato para referência
        cv2.circle(frame_original, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        # Textos na tela
        cv2.putText(frame_original, f"Sensibilidade: {sens}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame_original, f"Angulo Fixo: {int(angulo)} graus", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame_original, "Sem dados (Não detectou nada)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Mostra a imagem
    cv2.imshow(janela_principal, frame_original)

    # Avança para o próximo frame para dar efeito de vídeo tocando
    frame_idx += 1
    if frame_idx >= len(frames):
        frame_idx = 0 # Reinicia o vídeo

    # Velocidade de reprodução (30ms) e tecla de saída 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()