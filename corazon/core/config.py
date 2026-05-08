# corazon/core/config.py
# corazon/core/config.py
from dataclasses import dataclass

@dataclass
class TrackerConfig:
    """Objeto central de configuración del sistema de rastreo."""
    
    # --- SELECCIÓN DE ALGORITMO ---
    kit_procesamiento: str = "biologico"
    
    # --- PARÁMETROS BASE (Sliders Generales) ---
    sensibilidad: int = 35
    blur_imagen: int = 5
    desenfoque: int = 15      # El del hexágono suave (Screen 3)
    tolerancia: int = 50      # Centro dinámico (50 = Mediana)
    
    # --- MODULADORES KIT BIOLÓGICO (Los 4 Espacios Libres) ---
    # Estos valores se mapearán a tus sliders en la interfaz
    bio_clahe: int = 0        # Espacio 1: ClipLimit de CLAHE (0-10)
    bio_bilateral: int = 0    # Espacio 2: Fuerza del Filtro Bilateral (0-100)
    bio_tophat: int = 0       # Espacio 3: Tamaño del Kernel Top-Hat (0-50)
    bio_sobel: int = 0        # Espacio 4: Intensidad de Bordes Sobel (0-100)
    
    # --- INTERRUPTORES VISUALES (Toggles) ---
    mostrar_cyan: bool = True
    mostrar_rosa: bool = True
    mostrar_elipse: bool = True
    
    # --- CONFIGURACIÓN DE UI/VIDEO ---
    fps_objetivo: int = 30