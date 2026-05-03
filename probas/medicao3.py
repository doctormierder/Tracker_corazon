import cv2
import numpy as np
import math

def nada(x):
    pass

# ===== FUNCIONES ESTADÍSTICAS =====
def media(x):
    return np.mean(x)

def mediana(x):
    return np.median(x)

def winsorizada(x, limite=0.1):
    x = np.array(x)
    lower = np.percentile(x, limite * 100)
    upper = np.percentile(x, (1 - limite) * 100)
    x_w = np.clip(x, lower, upper)
    return np.mean(x_w)

# 1. Carregar frames
print("Carregando vídeo e pré-processando...")
cap = cv2.VideoCapture('coracao_video2.avi')

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
print(f"{len(frames)} frames carregados.")

# 2. Pré-calcular para cada sensibilidade
print("Calculando dados...")

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
            
            if len(casco_convexo) >= 5:
                (x, y), (eixo_menor, eixo_maior), angulo = cv2.fitEllipse(casco_convexo)
                angulos.append(angulo)
                centros_x.append(x)
                centros_y.append(y)
                comprimentos.append(eixo_maior)
    
    if angulos:
        dados_sensibilidade[sensibilidade] = {
            'media': {
                'x': media(centros_x),
                'y': media(centros_y),
                'angulo': media(angulos),
                'comprimento': media(comprimentos)
            },
            'mediana': {
                'x': mediana(centros_x),
                'y': mediana(centros_y),
                'angulo': mediana(angulos),
                'comprimento': mediana(comprimentos)
            },
            'winsor': {
                'x': winsorizada(centros_x),
                'y': winsorizada(centros_y),
                'angulo': winsorizada(angulos),
                'comprimento': winsorizada(comprimentos)
            }
        }
    else:
        dados_sensibilidade[sensibilidade] = None

print("Listo. Controles: 1=media, 2=mediana, 3=winsor")

# ===== INTERFAZ =====
janela = 'Eixo'
cv2.namedWindow(janela)
cv2.createTrackbar('Sensibilidade', janela, 30, 100, nada)

frame_idx = 0
modo = 'mediana'  # default

while True:
    frame_original = frames[frame_idx].copy()
    sens = cv2.getTrackbarPos('Sensibilidade', janela)
    
    dados = dados_sensibilidade.get(sens)
    
    if dados is not None:
        valores = dados[modo]
        
        x = valores['x']
        y = valores['y']
        angulo = valores['angulo']
        comprimento = valores['comprimento']
        
        angulo_rad = math.radians(angulo)
        dx = (comprimento / 2) * math.sin(angulo_rad)
        dy = (comprimento / 2) * math.cos(angulo_rad)
        
        pt1 = (int(x - dx), int(y + dy))
        pt2 = (int(x + dx), int(y - dy))
        
        cv2.line(frame_original, pt1, pt2, (0, 255, 255), 2)
        cv2.circle(frame_original, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        cv2.putText(frame_original, f"Sens: {sens} | Modo: {modo}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame_original, f"Angulo: {int(angulo)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame_original, "Sem dados", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow(janela, frame_original)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('1'):
        modo = 'media'
    elif key == ord('2'):
        modo = 'mediana'
    elif key == ord('3'):
        modo = 'winsor'

    frame_idx += 1
    if frame_idx >= len(frames):
        frame_idx = 0

cv2.destroyAllWindows()