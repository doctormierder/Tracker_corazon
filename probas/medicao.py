import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Carregar o vídeo e pré-processar os frames para a memória (para ser muito mais rápido)
# Troque o nome do arquivo para o vídeo que você quer analisar
cap = cv2.VideoCapture('coracao_video3.avi')
frames_borrados = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    frames_borrados.append(blurred)

cap.release()
print(f"Total de {len(frames_borrados)} frames carregados. Iniciando análise...")

# Listas para guardar o Desvio Padrão (variabilidade) de cada sensibilidade
variabilidade_angulos = []
variabilidade_posicao_x = []
variabilidade_posicao_y = []
sensibilidades = list(range(0, 101)) # 0 até 100

# 2. Analisar cada sensibilidade possível
for sensibilidade in sensibilidades:
    angulos = []
    posicoes_x = []
    posicoes_y = []
    
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
            
            if len(casco_convexo) >= 5:
                (x, y), (eixo_menor, eixo_maior), angulo = cv2.fitEllipse(casco_convexo)
                angulos.append(angulo)
                posicoes_x.append(x)
                posicoes_y.append(y)
    
    # 3. Calcular a variabilidade (Desvio Padrão) para esta sensibilidade
    # Se a sensibilidade não detectou nada no vídeo todo, colocamos 0 para evitar erros
    if angulos:
        variabilidade_angulos.append(np.std(angulos))
        variabilidade_posicao_x.append(np.std(posicoes_x))
        variabilidade_posicao_y.append(np.std(posicoes_y))
    else:
        variabilidade_angulos.append(0)
        variabilidade_posicao_x.append(0)
        variabilidade_posicao_y.append(0)

print("Análise concluída. Gerando gráficos...")

# 4. Plotar os Gráficos
plt.figure(figsize=(12, 8))

# Gráfico 1: Variabilidade do Ângulo
plt.subplot(2, 1, 1)
plt.plot(sensibilidades, variabilidade_angulos, color='orange', linewidth=2)
plt.title('Variabilidade da Orientação (Linha Amarela) vs Sensibilidade')
plt.ylabel('Desvio Padrão do Ângulo (Graus)')
plt.grid(True)

# Gráfico 2: Variabilidade da Posição Central (Centro de Massa)
plt.subplot(2, 1, 2)
plt.plot(sensibilidades, variabilidade_posicao_x, color='blue', label='Eixo X (Horizontal)', linewidth=2)
plt.plot(sensibilidades, variabilidade_posicao_y, color='green', label='Eixo Y (Vertical)', linewidth=2)
plt.title('Variabilidade do Centro do Coração vs Sensibilidade')
plt.xlabel('Sensibilidade (0 a 100)')
plt.ylabel('Desvio Padrão da Posição (Pixels)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()