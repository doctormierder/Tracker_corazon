# interfaz/ventana_principal.py
import sys
import os
from PyQt6.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QGridLayout, QTextEdit, 
                             QListWidget, QSplitter, QTreeView,
                             QFrame, QAbstractItemView, QSlider, QGroupBox)
from PyQt6.QtCore import QThread, Qt, QDir, pyqtSlot
from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QFileSystemModel, QImage

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
        layout_main = QVBoxLayout(widget_central)

        # =================================================================
        # 1. BARRA SUPERIOR DE CONTROLES (2 Filas x 6 Columnas)
        # =================================================================
        panel_superior = QGroupBox("Ajustes de Escáner y Kits")
        panel_superior.setStyleSheet("color: #ccc; font-weight: bold;")
        grid_controles = QGridLayout(panel_superior)

        # Fila 0: Etiquetas
        grid_controles.addWidget(QLabel("Desenfoque (Blur)"), 0, 0)
        grid_controles.addWidget(QLabel("Tolerancia (Threshold)"), 0, 1)
        
        self.lbl_clahe = QLabel(f"CLAHE: {self.config.bio_clahe/2.0}")
        grid_controles.addWidget(self.lbl_clahe, 0, 2)
        
        self.lbl_bilateral = QLabel(f"Bilateral: {self.config.bio_bilateral}")
        grid_controles.addWidget(self.lbl_bilateral, 0, 3)
        
        self.lbl_mediana = QLabel(f"Mediana: {self.config.bio_mediana}")
        grid_controles.addWidget(self.lbl_mediana, 0, 4)
        
        self.lbl_cierre = QLabel(f"Cierre: {self.config.bio_cierre}")
        grid_controles.addWidget(self.lbl_cierre, 0, 5)

        # Fila 1: Sliders
        self.slider_blur = QSlider(Qt.Orientation.Horizontal)
        self.slider_blur.setRange(1, 51)
        self.slider_blur.setValue(self.config.desenfoque)
        self.slider_blur.setSingleStep(2)
        self.slider_blur.valueChanged.connect(self.al_cambiar_desenfoque)
        grid_controles.addWidget(self.slider_blur, 1, 0)

        self.slider_tol = QSlider(Qt.Orientation.Horizontal)
        self.slider_tol.setRange(1, 100)
        self.slider_tol.setValue(self.config.tolerancia)
        self.slider_tol.valueChanged.connect(self.al_cambiar_tolerancia)
        grid_controles.addWidget(self.slider_tol, 1, 1)

        self.slider_clahe = QSlider(Qt.Orientation.Horizontal)
        self.slider_clahe.setRange(0, 40)
        self.slider_clahe.setValue(self.config.bio_clahe)
        self.slider_clahe.valueChanged.connect(self.al_cambiar_clahe)
        grid_controles.addWidget(self.slider_clahe, 1, 2)

        self.slider_bilateral = QSlider(Qt.Orientation.Horizontal)
        self.slider_bilateral.setRange(0, 200)
        self.slider_bilateral.setValue(self.config.bio_bilateral)
        self.slider_bilateral.valueChanged.connect(self.al_cambiar_bilateral)
        grid_controles.addWidget(self.slider_bilateral, 1, 3)

        self.slider_mediana = QSlider(Qt.Orientation.Horizontal)
        self.slider_mediana.setRange(1, 21)
        self.slider_mediana.setValue(self.config.bio_mediana)
        self.slider_mediana.valueChanged.connect(self.al_cambiar_mediana)
        grid_controles.addWidget(self.slider_mediana, 1, 4)

        self.slider_cierre = QSlider(Qt.Orientation.Horizontal)
        self.slider_cierre.setRange(0, 31)
        self.slider_cierre.setValue(self.config.bio_cierre)
        self.slider_cierre.valueChanged.connect(self.al_cambiar_cierre)
        grid_controles.addWidget(self.slider_cierre, 1, 5)

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
        self.visor_ecg = self._crear_visor("2. ECG / CARDIOGRAMA", "#111") # <-- Cambio Crítico
        self.visor_tratada = self._crear_visor("3. IMAGEN TRATADA (Kit)", "#1a1a1a")
        self.visor_mask2 = self._crear_visor("4. MÁSCARA DEFINITIVA", "#0a0a0a")
        
        self.layout_grid_videos.addWidget(self.visor_video, 0, 0)
        self.layout_grid_videos.addWidget(self.visor_ecg, 0, 1)
        self.layout_grid_videos.addWidget(self.visor_tratada, 1, 0)
        self.layout_grid_videos.addWidget(self.visor_mask2, 1, 1)

        # --- PARTE DERECHA: BARRA LATERAL ---
        barra_lateral = QWidget()
        layout_lateral = QVBoxLayout(barra_lateral)

        # Panel de Métricas Fisiológicas
        self.lbl_confianza = QLabel("Confianza Eje: 0.0%")
        self.lbl_confianza.setStyleSheet("color: white; font-size: 22px; font-weight: bold; background: #222; padding: 10px;")
        layout_lateral.addWidget(self.lbl_confianza)

        estilo_metricas = "color: #00ff00; font-size: 14px; background: #1e1e1e; padding: 5px; font-family: Consolas;"
        
        self.lbl_ancho_inst = QLabel("Ancho Actual: 0.0 px")
        self.lbl_ancho_inst.setStyleSheet(estilo_metricas)
        layout_lateral.addWidget(self.lbl_ancho_inst)

        self.lbl_sys = QLabel("Sístole (1 | 3 | 10): 0.0 | 0.0 | 0.0")
        self.lbl_sys.setStyleSheet(estilo_metricas)
        layout_lateral.addWidget(self.lbl_sys)

        self.lbl_dia = QLabel("Diástole (1 | 3 | 10): 0.0 | 0.0 | 0.0")
        self.lbl_dia.setStyleSheet(estilo_metricas)
        layout_lateral.addWidget(self.lbl_dia)

        self.lbl_freq = QLabel("Frecuencia (2 | 10): 0.0 | 0.0 BPM")
        self.lbl_freq.setStyleSheet(estilo_metricas)
        layout_lateral.addWidget(self.lbl_freq)

        self.lbl_var = QLabel("Varianza (15): 0.0")
        self.lbl_var.setStyleSheet(estilo_metricas)
        layout_lateral.addWidget(self.lbl_var)

        # Explorador
        self.model_fs = QFileSystemModel()
        self.model_fs.setRootPath("") 
        self.tree_explorer = QTreeView()
        self.tree_explorer.setModel(self.model_fs)
        self.tree_explorer.setRootIndex(self.model_fs.index(QDir.homePath())) 
        self.tree_explorer.hideColumn(1); self.tree_explorer.hideColumn(2); self.tree_explorer.hideColumn(3)
        self.tree_explorer.doubleClicked.connect(self.archivo_explorer_a_cola)
        
        layout_lateral.addWidget(QLabel("📁 EXPLORADOR DE ARCHIVOS"))
        layout_lateral.addWidget(self.tree_explorer, stretch=4)

        # Cola
        self.lista_cola = QListWidget()
        self.lista_cola.setStyleSheet("background-color: #2a2a2a; color: #3498db;")
        layout_lateral.addWidget(QLabel("📋 COLA DE PROCESAMIENTO"))
        layout_lateral.addWidget(self.lista_cola, stretch=2)

        # Botones
        layout_botones = QHBoxLayout()
        btn_play = QPushButton("▶ REPRODUCIR")
        btn_play.setStyleSheet("background-color: #28a745; color: white; padding: 8px;")
        btn_play.clicked.connect(self.reproducir_de_cola)
        
        btn_borrar = QPushButton("🗑 QUITAR")
        btn_borrar.clicked.connect(self.quitar_de_cola)
        layout_botones.addWidget(btn_play)
        layout_botones.addWidget(btn_borrar)
        layout_lateral.addLayout(layout_botones)

        btn_camara = QPushButton("🔬 ACTIVAR CÁMARA")
        btn_camara.clicked.connect(self.iniciar_camara)
        layout_lateral.addWidget(btn_camara)

        # Ensamblar Splitter
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
    # SLIDERS E INTERFAZ
    # =================================================================
    def al_cambiar_desenfoque(self, valor):
        self.config.desenfoque = valor if valor % 2 != 0 else valor + 1

    def al_cambiar_tolerancia(self, valor):
        self.config.tolerancia = valor

    def al_cambiar_clahe(self, valor):
        self.config.bio_clahe = valor
        self.lbl_clahe.setText(f"CLAHE: {valor/2.0}")

    def al_cambiar_bilateral(self, valor):
        self.config.bio_bilateral = valor
        self.lbl_bilateral.setText(f"Bilateral: {valor}")

    def al_cambiar_mediana(self, valor):
        self.config.bio_mediana = valor
        val_real = valor if valor % 2 != 0 else valor + 1
        self.lbl_mediana.setText(f"Mediana: {val_real if valor > 1 else 'Apagado'}")

    def al_cambiar_cierre(self, valor):
        self.config.bio_cierre = valor
        val_real = valor if valor % 2 != 0 else valor + 1
        self.lbl_cierre.setText(f"Cierre: {val_real if valor > 0 else 'Apagado'}")
        
    # =================================================================
    # RECEPCIÓN DE DATOS (MULTIHILO)
    # =================================================================
    @pyqtSlot(dict)
    def actualizar_metricas(self, stats):
        """Mapeo a prueba de fallos de los datos del Analizador Fisiológico"""
        if "confianza" in stats:
            self.lbl_confianza.setText(f"Confianza Eje: {stats['confianza']}%")
            
        if "ancho_actual" in stats:
            self.lbl_ancho_inst.setText(f"Ancho Actual: {stats['ancho_actual']} px")
            
        if "sys_1" in stats:
            sys_1 = round(stats.get('sys_1', 0), 1)
            sys_3 = round(stats.get('sys_3', 0), 1)
            sys_10 = round(stats.get('max_prom', 0), 1)
            self.lbl_sys.setText(f"Sístole (1 | 3 | 10): {sys_1} | {sys_3} | {sys_10}")

        if "dia_1" in stats:
            dia_1 = round(stats.get('dia_1', 0), 1)
            dia_3 = round(stats.get('dia_3', 0), 1)
            dia_10 = round(stats.get('min_prom', 0), 1)
            self.lbl_dia.setText(f"Diástole (1 | 3 | 10): {dia_1} | {dia_3} | {dia_10}")

        if "freq_2" in stats:
            f_2 = round(stats.get('freq_2', 0), 1)
            f_10 = round(stats.get('freq_10', 0), 1)
            self.lbl_freq.setText(f"Frecuencia (2 | 10): {f_2} | {f_10} BPM")

        if "var_15" in stats:
            var_15 = round(stats.get('var_15', 0), 2)
            self.lbl_var.setText(f"Varianza (15): {var_15}")

    @pyqtSlot(QImage, QImage, QImage, QImage)
    def actualizar_pantallas(self, qt_f, qt_ecg, qt_tratada, qt_m2):
        """Limpieza y eficiencia total en la asignación de pixmaps"""
        def escalar(label, img):
            if img.isNull(): return QPixmap()
            pix = QPixmap.fromImage(img)
            return pix.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        self.visor_video.setPixmap(escalar(self.visor_video, qt_f))
        self.visor_ecg.setPixmap(escalar(self.visor_ecg, qt_ecg))
        self.visor_tratada.setPixmap(escalar(self.visor_tratada, qt_tratada))
        self.visor_mask2.setPixmap(escalar(self.visor_mask2, qt_m2))

    # =================================================================
    # GESTIÓN DE COLAS Y WORKERS
    # =================================================================
    def agregar_a_cola(self, ruta):
        ext = ('.mp4', '.avi', '.mpg', '.mpeg', '.mov')
        if ruta.lower().endswith(ext):
            items = [self.lista_cola.item(i).text() for i in range(self.lista_cola.count())]
            if ruta not in items:
                self.lista_cola.addItem(ruta)

    def archivo_explorer_a_cola(self, index):
        ruta = self.model_fs.filePath(index)
        if not self.model_fs.isDir(index):
            self.agregar_a_cola(ruta)

    def reproducir_de_cola(self):
        item = self.lista_cola.currentItem()
        # Si no hay selección pero la cola tiene elementos, fuerza el primero
        if not item and self.lista_cola.count() > 0:
            item = self.lista_cola.item(0)
            self.lista_cola.setCurrentItem(item)
            
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
        
        # Conexiones explícitas de señales
        if hasattr(self.worker, 'frame_ready'):
            self.worker.frame_ready.connect(self.actualizar_pantallas)
        if hasattr(self.worker, 'stats_ready'):
            self.worker.stats_ready.connect(self.actualizar_metricas)
            
        # Sistema de Autoplay para continuar con la cola
        if hasattr(self.worker, 'finished'):
            self.worker.finished.connect(self._video_terminado)
            
        self.thread.start()

    @pyqtSlot()
    def _video_terminado(self):
        self._detener_trabajador_actual()
        # Elimina el que se estaba reproduciendo y avanza
        if self.lista_cola.count() > 0:
            self.lista_cola.takeItem(0) 
            self.reproducir_de_cola()

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