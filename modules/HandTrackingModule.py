import cv2
import mediapipe as mp
import math

class HandDetector:
    def __init__(self, mode=False, max_hands=1, detection_con=0.7, track_con=0.5):
        """
        Inicializa el detector de manos usando MediaPipe.
        :param mode: Si es True, trata cada imagen como una detección nueva.
        :param max_hands: Número máximo de manos a detectar.
        :param detection_con: Confianza mínima de detección.
        :param track_con: Confianza mínima de seguimiento.
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        # Configuración de MediaPipe Hands 
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils

    def encontrar_manos(self, img, draw=True):
        """
        Detecta las manos en un frame y opcionalmente dibuja las conexiones.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    # Dibuja los 21 puntos y sus conexiones 
                    self.mp_draw.draw_landmarks(
                        img, hand_lms, self.mp_hands.HAND_CONNECTIONS
                    )
        return img

    def obtener_posicion(self, img, hand_no=0, draw=True):
        """
        Extrae la lista de coordenadas (x, y) de los 21 puntos de una mano.
        """
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            mi_mano = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(mi_mano.landmark):
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