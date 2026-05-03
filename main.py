# main.py
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from interfaz.ventana_principal import VentanaPrincipal





if __name__ == "__main__":
    # Configuración de alta densidad para pantallas 4K/Laptop
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    
    # Podemos definir un estilo global aquí más adelante (Dark Mode, etc.)
    app.setStyle("Fusion") 
    
    ventana = VentanaPrincipal()
    ventana.show()
    
    sys.exit(app.exec())