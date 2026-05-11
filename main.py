# main.py
import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from interfaz.ventana_principal import VentanaPrincipal

# Forzar compatibilidad de servidor gráfico en Linux (Wayland/X11)
if sys.platform.startswith('linux'):
    os.environ["QT_QPA_PLATFORM"] = "xcb"

if __name__ == "__main__":
    # Configuración de alta densidad para pantallas 4K/Laptop
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    
    ventana = VentanaPrincipal()
    ventana.show()
    
    sys.exit(app.exec())