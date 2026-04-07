import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class VolumeController:
    def __init__(self):
        """Inicializa el control de audio de Windows a través de pycaw """
        # Acceder a los altavoces del sistema
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        
        # Obtener el rango de volumen (típicamente de -65.25 a 0.0)
        self.vol_range = self.volume.GetVolumeRange()
        self.min_vol = self.vol_range[0]
        self.max_vol = self.vol_range[1]
        
        # Variables para la interfaz visual
        self.vol_bar = 400
        self.vol_per = 0

    def procesar_volumen(self, distancia, lm_list):
        """
        Mapea la distancia de los dedos al volumen del sistema[cite: 140, 157].
        Solo aplica el cambio si el meñique está bajado.
        """
        # 1. Mapeo: Convertimos distancia (píxeles) a rango de volumen (dB), barra y porcentaje 
        # Ajusta [20, 200] según el tamaño de tu mano y distancia a la cámara
        vol = np.interp(distancia, [20, 200], [self.min_vol, self.max_vol])
        self.vol_bar = np.interp(distancia, [20, 200], [400, 150])
        self.vol_per = np.interp(distancia, [20, 200], [0, 100])

        # 2. Lógica del Gesto Intencional (Meñique bajado) [cite: 141, 167]
        # Punto 20: Punta del meñique | Punto 18: Articulación inferior del meñique
        # En OpenCV, el eje Y crece hacia abajo; si Y20 > Y18, el dedo está "bajado".
        aplicar_cambio = False
        if len(lm_list) >= 21:
            if lm_list[20][2] > lm_list[18][2]:
                self.volume.SetMasterVolumeLevel(vol, None)
                aplicar_cambio = True

        return self.vol_per, self.vol_bar, aplicar_cambio

    def obtener_volumen_actual(self):
        """Devuelve el nivel de volumen actual para el log de eventos [cite: 160]"""
        return self.volume.GetMasterVolumeLevelScalar() * 100