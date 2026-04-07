import cv2
import mediapipe as mp
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8), # indice
    (5, 9), (9, 10), (10, 11), (11, 12), # medio
    (9, 13), (13, 14), (14, 15), (15, 16), # anular
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # menique
]

class HandDetector:
    def __init__(self, mode=False, max_hands=1, detection_con=0.7, track_con=0.5):
        """
        Inicializa el detector de manos usando MediaPipe Tasks API.
        :param mode: Si es True, trata cada imagen como una detección nueva.
        :param max_hands: Número máximo de manos a detectar.
        :param detection_con: Confianza mínima de detección.
        :param track_con: Confianza mínima de seguimiento.
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = float(detection_con)
        self.track_con = float(track_con)
        self.lm_list = []
        self.results = None

        # Configuración de MediaPipe Hands Tasks API
        model_path = os.path.join('models', 'hand_landmarker.task')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {os.path.abspath(model_path)}")
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.detection_con,
            min_hand_presence_confidence=self.track_con,
            min_tracking_confidence=self.track_con
        )
        self.detector = vision.HandLandmarker.create_from_options(self.options)

    def encontrar_manos(self, img, draw=True):
        """
        Detecta las manos en un frame y opcionalmente dibuja las conexiones.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        self.results = self.detector.detect(mp_image)

        if self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                if draw:
                    h, w, c = img.shape
                    # Dibuja los 21 puntos y sus conexiones
                    for connection in HAND_CONNECTIONS:
                        lm_start = hand_landmarks[connection[0]]
                        lm_end = hand_landmarks[connection[1]]
                        cx1, cy1 = int(lm_start.x * w), int(lm_start.y * h)
                        cx2, cy2 = int(lm_end.x * w), int(lm_end.y * h)
                        cv2.line(img, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
                        
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        return img

    def obtener_posicion(self, img, hand_no=0, draw=True):
        """
        Extrae la lista de coordenadas (x, y) de los 21 puntos de una mano.
        """
        self.lm_list = []
        if self.results and self.results.hand_landmarks:
            if hand_no < len(self.results.hand_landmarks):
                mi_mano = self.results.hand_landmarks[hand_no]
                for id, lm in enumerate(mi_mano):
                    # Convertir coordenadas normalizadas a píxeles
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lm_list.append([id, cx, cy])
                    
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        return self.lm_list

    def calcular_distancia(self, p1, p2, img=None, draw=True):
        """
        Calcula la distancia euclidiana entre dos puntos (ej. pulgar e índice).
        """
        if not self.lm_list or p1 >= len(self.lm_list) or p2 >= len(self.lm_list):
            return 0, img, [0, 0, 0, 0, 0, 0]
            
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        distancia = math.hypot(x2 - x1, y2 - y1)

        if img is not None and draw:
            # Dibujar línea y círculos entre los dedos para feedback visual [cite: 166]
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

        return distancia, img, [x1, y1, x2, y2, cx, cy]