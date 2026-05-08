# interfaz/ventana_principal.py
import sys
import os
from PyQt6.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QGridLayout, QTextEdit, 
                             QListWidget, QSplitter, QTreeView,
                             QFrame, QAbstractItemView, QSlider, QGroupBox)
from PyQt6.QtCore import QThread, Qt, QDir
from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QFileSystemModel

from corazon.video_worker import VideoWorker
from corazon.camera_worker import CameraWorker
from corazon.core.config import TrackerConfig 

class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical Tracker Pro - Control Total")
        self.resize(1400, 900)
        self.setAcceptDrops(True)
        
        self.config = TrackerConfig()
        self.thread = None
        self.worker = None

        self._construir_interfaz_profesional()

    def _construir_interfaz_profesional(self):
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_main = QVBoxLayout(widget_central) # Vertical principal

        # =================================================================
        # 1. BARRA SUPERIOR DE CONTROLES (2 Filas x 6 Columnas)
        # =================================================================
        panel_superior = QGroupBox("Ajustes de Escáner y Kits")
        panel_superior.setStyleSheet("color: #ccc; font-weight: bold;")
        grid_controles = QGridLayout(panel_superior)

        # Fila 0: Etiquetas (Ahora dinámicas para que actualicen el texto)
        grid_controles.addWidget(QLabel("Desenfoque (Blur)"), 0, 0)
        grid_controles.addWidget(QLabel("Tolerancia (Threshold)"), 0, 1)
        
        self.lbl_clahe = QLabel(f"CLAHE: {self.config.bio_clahe/2.0}")
        grid_controles.addWidget(self.lbl_clahe, 0, 2)
        
        self.lbl_bilateral = QLabel(f"Bilateral: {self.config.bio_bilateral}")
        grid_controles.addWidget(self.lbl_bilateral, 0, 3)
        
        self.lbl_tophat = QLabel(f"Top-Hat: {self.config.bio_tophat}")
        grid_controles.addWidget(self.lbl_tophat, 0, 4)
        
        self.lbl_sobel = QLabel(f"Sobel: {self.config.bio_sobel}%")
        grid_controles.addWidget(self.lbl_sobel, 0, 5)

        # Fila 1: Controles Reales
        self.slider_blur = QSlider(Qt.Orientation.Horizontal)
        self.slider_blur.setRange(1, 51)
        self.slider_blur.setValue(self.config.desenfoque)
        self.slider_blur.setSingleStep(2)
        self.slider_blur.valueChanged.connect(self.al_cambiar_desenfoque)
        grid_controles.addWidget(self.slider_blur, 1, 0)

        self.slider_tol = QSlider(Qt.Orientation.Horizontal)
        self.slider_tol.setRange(1, 100) # Cuantizado del 1 al 100
        self.slider_tol.setValue(self.config.tolerancia)
        self.slider_tol.valueChanged.connect(self.al_cambiar_tolerancia)
        grid_controles.addWidget(self.slider_tol, 1, 1)

        # --- REEMPLAZO DE LOS ESPACIOS VACÍOS POR LOS SLIDERS BIOLÓGICOS ---
        
        # Slider CLAHE (0 a 20 -> en el kit se divide a 0.0 - 10.0)
        self.slider_clahe = QSlider(Qt.Orientation.Horizontal)
        self.slider_clahe.setRange(0, 40)
        self.slider_clahe.setValue(self.config.bio_clahe)
        self.slider_clahe.valueChanged.connect(self.al_cambiar_clahe)
        grid_controles.addWidget(self.slider_clahe, 1, 2)

        # Slider Bilateral (0 a 100)
        self.slider_bilateral = QSlider(Qt.Orientation.Horizontal)
        self.slider_bilateral.setRange(0, 200)
        self.slider_bilateral.setValue(self.config.bio_bilateral)
        self.slider_bilateral.valueChanged.connect(self.al_cambiar_bilateral)
        grid_controles.addWidget(self.slider_bilateral, 1, 3)

        # Slider Top-Hat (0 a 50)
        self.slider_tophat = QSlider(Qt.Orientation.Horizontal)
        self.slider_tophat.setRange(0, 200)
        self.slider_tophat.setValue(self.config.bio_tophat)
        self.slider_tophat.valueChanged.connect(self.al_cambiar_tophat)
        grid_controles.addWidget(self.slider_tophat, 1, 4)

        # Slider Sobel (0 a 100%)
        self.slider_sobel = QSlider(Qt.Orientation.Horizontal)
        self.slider_sobel.setRange(0, 200)
        self.slider_sobel.setValue(self.config.bio_sobel)
        self.slider_sobel.valueChanged.connect(self.al_cambiar_sobel)
        grid_controles.addWidget(self.slider_sobel, 1, 5)

        layout_main.addWidget(panel_superior)

        # =================================================================
        # 2. CUERPO PRINCIPAL (Splitter Videos / Sidebar)
        # =================================================================
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- PARTE IZQUIERDA: CUADRANTE DE VIDEOS (2x2 PERFECTO) ---
        contenedor_izquierdo = QWidget()
        self.layout_grid_videos = QGridLayout(contenedor_izquierdo)
        self.layout_grid_videos.setSpacing(10)

        self.visor_video = self._crear_visor("1. VIDEO ORIGINAL", "#000")
        self.visor_mask1 = self._crear_visor("2. MÁSCARA 1 (Rastreo)", "#111")
        self.visor_tratada = self._crear_visor("3. IMAGEN TRATADA (Kit)", "#1a1a1a")
        self.visor_mask2 = self._crear_visor("4. MÁSCARA 2 (Línea Naranja)", "#0a0a0a")
        
        self.layout_grid_videos.addWidget(self.visor_video, 0, 0)
        self.layout_grid_videos.addWidget(self.visor_mask1, 0, 1)
        self.layout_grid_videos.addWidget(self.visor_tratada, 1, 0)
        self.layout_grid_videos.addWidget(self.visor_mask2, 1, 1)

        # --- PARTE DERECHA: BARRA LATERAL ---
        barra_lateral = QWidget()
        layout_lateral = QVBoxLayout(barra_lateral)

        # Dashboard Movido a la Barra Lateral
        self.lbl_confianza = QLabel("Confianza Eje: 0.0%")
        self.lbl_confianza.setStyleSheet("color: white; font-size: 22px; font-weight: bold; background: #222; padding: 10px;")
        layout_lateral.addWidget(self.lbl_confianza)


        self.consola_datos = QTextEdit()
        self.consola_datos.setReadOnly(True)
        self.consola_datos.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: 'Courier New';")
        layout_lateral.addWidget(QLabel("💬 CONSOLA DE ANÁLISIS"))
        layout_lateral.addWidget(self.consola_datos, stretch=3)

        self.model_fs = QFileSystemModel()
        self.model_fs.setRootPath("") 
        self.tree_explorer = QTreeView()
        self.tree_explorer.setModel(self.model_fs)
        self.tree_explorer.setRootIndex(self.model_fs.index(QDir.homePath())) 
        self.tree_explorer.hideColumn(1); self.tree_explorer.hideColumn(2); self.tree_explorer.hideColumn(3)
        self.tree_explorer.doubleClicked.connect(self.archivo_explorer_a_cola)
        
        layout_lateral.addWidget(QLabel("📁 EXPLORADOR DE ARCHIVOS"))
        layout_lateral.addWidget(self.tree_explorer, stretch=4)

        self.lista_cola = QListWidget()
        self.lista_cola.setStyleSheet("background-color: #2a2a2a; color: #3498db;")
        layout_lateral.addWidget(QLabel("📋 COLA DE PROCESAMIENTO"))
        layout_lateral.addWidget(self.lista_cola, stretch=2)

        # Botones
        layout_botones = QHBoxLayout()
        btn_play = QPushButton("▶ REPRODUCIR")
        btn_play.setStyleSheet("background-color: #28a745; color: white; padding: 8px;")
        btn_play.clicked.connect(self.reproducir_de_cola)
        
        btn_borrar = QPushButton("🗑")
        btn_borrar.clicked.connect(self.quitar_de_cola)
        layout_botones.addWidget(btn_play)
        layout_botones.addWidget(btn_borrar)
        layout_lateral.addLayout(layout_botones)

        btn_camara = QPushButton("🔬 ACTIVAR CÁMARA")
        btn_camara.clicked.connect(self.iniciar_camara)
        layout_lateral.addWidget(btn_camara)

        # Ensamblar
        self.splitter.addWidget(contenedor_izquierdo)
        self.splitter.addWidget(barra_lateral)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        layout_main.addWidget(self.splitter)

    def _crear_visor(self, texto, color_bg):
        label = QLabel(texto)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(f"background-color: {color_bg}; border: 1px solid #333; border-radius: 4px;")
        label.setMinimumSize(200, 150)
        return label

    # =================================================================
    # LÓGICA DE INTERFAZ Y ACTUALIZACIONES
    # =================================================================
    def al_cambiar_desenfoque(self, valor):
        # Aseguramos que sea impar
        self.config.desenfoque = valor if valor % 2 != 0 else valor + 1

    def al_cambiar_tolerancia(self, valor):
        self.config.tolerancia = valor

    def al_cambiar_clahe(self, valor):
        self.config.bio_clahe = valor
        self.lbl_clahe.setText(f"CLAHE: {valor/2.0}")

    def al_cambiar_bilateral(self, valor):
        self.config.bio_bilateral = valor
        self.lbl_bilateral.setText(f"Bilateral: {valor}")

    def al_cambiar_tophat(self, valor):
        self.config.bio_tophat = valor
        self.lbl_tophat.setText(f"Top-Hat: {valor}")

    def al_cambiar_sobel(self, valor):
        self.config.bio_sobel = valor
        self.lbl_sobel.setText(f"Sobel: {valor}%")
        
    def actualizar_metricas(self, stats):
        confianza = stats.get("confianza", 0)
        color = "#00ff00" if confianza >= 85 else "#ffaa00" if confianza >= 50 else "#ff0000"
        self.lbl_confianza.setText(f"Confianza Eje: {confianza}%")
        self.lbl_confianza.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold;")

    def actualizar_pantallas(self, img_p, img_m1, img_tratada, img_m2):
        def escalar(label, img):
            pix = QPixmap.fromImage(img)
            return pix.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        self.visor_video.setPixmap(escalar(self.visor_video, img_p))
        self.visor_mask1.setPixmap(escalar(self.visor_mask1, img_m1))
        self.visor_tratada.setPixmap(escalar(self.visor_tratada, img_tratada)) # Pantalla 3
        self.visor_mask2.setPixmap(escalar(self.visor_mask2, img_m2))         # Pantalla 4
        self.visor_video.setPixmap(escalar(self.visor_video, img_p))
        self.visor_mask1.setPixmap(escalar(self.visor_mask1, img_m1))
        self.visor_mask2.setPixmap(escalar(self.visor_mask2, img_m2))

    # =================================================================
    # GESTIÓN DE ARCHIVOS Y WORKERS (El código que ya tenías)
    # =================================================================
    def agregar_a_cola(self, ruta):
        ext = ('.mp4', '.avi', '.mpg', '.mpeg', '.mov')
        if ruta.lower().endswith(ext):
            items = [self.lista_cola.item(i).text() for i in range(self.lista_cola.count())]
            if ruta not in items:
                self.lista_cola.addItem(ruta)
                self.consola_datos.append(f">> Añadido a cola: {os.path.basename(ruta)}")

    def archivo_explorer_a_cola(self, index):
        ruta = self.model_fs.filePath(index)
        if not self.model_fs.isDir(index):
            self.agregar_a_cola(ruta)

    def reproducir_de_cola(self):
        item = self.lista_cola.currentItem()
        if item:
            self._detener_trabajador_actual()
            self.thread = QThread()
            self.worker = VideoWorker(item.text(), self.config)
            self._conectar_y_arrancar()

    def iniciar_camara(self):
        self._detener_trabajador_actual()
        self.thread = QThread()
        self.worker = CameraWorker(0, self.config)
        self._conectar_y_arrancar()

    def quitar_de_cola(self):
        fila = self.lista_cola.currentRow()
        if fila >= 0: self.lista_cola.takeItem(fila)

    def _conectar_y_arrancar(self):
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        
        # Conectamos las 3 pantallas y las métricas
        if hasattr(self.worker, 'frame_ready'):
            self.worker.frame_ready.connect(self.actualizar_pantallas)
        if hasattr(self.worker, 'stats_ready'):
            self.worker.stats_ready.connect(self.actualizar_metricas)
            
        self.thread.start()

    def _detener_trabajador_actual(self):
        if self.worker:
            self.worker.stop()
            self.thread.quit()
            self.thread.wait()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls(): event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            self.agregar_a_cola(url.toLocalFile())

    def closeEvent(self, event):
        self._detener_trabajador_actual()
        event.accept()