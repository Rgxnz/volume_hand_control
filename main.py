import cv2
import time
import math
from modules.HandTrackingModule import HandDetector
from modules.VolumeHandControl import VolumeController
from dao.mongodb_dao import MongoDAO

def main():
    # 1. Inicialización de componentes
    cap = cv2.VideoCapture(0) # Captura de vídeo funcional [cite: 145]
    detector = HandDetector(detection_con=0.8) # MediaPipe Hand Landmarker [cite: 147]
    vol_control = VolumeController() # Lógica pycaw [cite: 148, 157]
    dao = MongoDAO() # DAO con patrón Singleton [cite: 161]
    
    # 2. Registro de inicio de sesión en MongoDB Atlas [cite: 142, 159]
    inicio_ts = time.time()
    id_sesion = dao.registrar_sesion(inicio_ts)
    
    vol_anterior = vol_control.obtener_volumen_actual()

    while True:
        success, img = cap.read()
        if not success: break

        # 3. Procesamiento de imagen y detección [cite: 146, 147, 166]
        img = detector.encontrar_manos(img)
        lm_list = detector.obtener_posicion(img, draw=False)

        # 4. Indicador visual de conexión a DB (Criterio de evaluación) [cite: 170]
        db_status = "DB: OK" if dao.connected else "DB: --"
        color_db = (0, 255, 0) if dao.connected else (0, 0, 255)
        cv2.putText(img, db_status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_db, 2)

        if len(lm_list) != 0:
            # Coordenadas del pulgar (4) e índice (8) [cite: 140, 147]
            x1, y1 = lm_list[4][1], lm_list[4][2]
            x2, y2 = lm_list[8][1], lm_list[8][2]
            
            # Calcular distancia entre dedos [cite: 140, 156]
            distancia = math.hypot(x2 - x1, y2 - y1)
            
            # 5. Aplicar volumen y verificar gesto del meñique [cite: 141, 157, 167]
            vol_per, vol_bar, aplicado = vol_control.procesar_volumen(distancia, lm_list)
            
            # Dibujar elementos visuales (barra y porcentaje) [cite: 166]
            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
            cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f'{int(vol_per)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

            # 6. Persistencia de eventos si el cambio fue aplicado [cite: 142, 160, 169]
            if aplicado:
                vol_nuevo = vol_per
                dao.guardar_evento_volumen(vol_anterior, vol_nuevo, distancia)
                vol_anterior = vol_nuevo
                cv2.circle(img, (x2, y2), 15, (0, 255, 0), cv2.FILLED) # Feedback de "aplicado"

        # Mostrar la ventana [cite: 146]
        cv2.imshow("Control de Volumen HandTracking", img)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7. Finalización y registro de duración de sesión [cite: 142, 159]
    duracion = time.time() - inicio_ts
    dao.finalizar_sesion(id_sesion, duracion)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()