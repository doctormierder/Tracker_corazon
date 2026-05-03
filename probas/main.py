# importanto librerias
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2








print(cv2.__version__)

webcam = cv2.VideoCapture(0)

carregarAlgoritmo = cv2.CascadeClassifier(r"venv_prueba\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")

imagem = cv2.imread("imagem_baixada.png")
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

faces = carregarAlgoritmo.detectMultiScale(imagemCinza)

cv2.imwrite("imagemescrita.png", imagemCinza)

print(faces)

if False:
    validacao, frame = webcam.read()
    while validacao:
        validacao, frame = webcam.read()
        cv2.imshow("titulo genérico", frame)
        key = cv2.waitKey(20)
        print(key)
        if key == 27:
            break

    cv2.imwrite("imagemsalvada.png", frame)

webcam.release()
cv2.destroyAllWindows()