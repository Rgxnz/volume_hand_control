import os
import time
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

# Carga las variables desde el archivo .env [cite: 158]
load_dotenv()

class MongoDAO:
    _instance = None  # Almacena la instancia única (Singleton) [cite: 161]

    def __new__(cls):
        """Implementación del patrón Singleton para la conexión [cite: 161]"""
        if cls._instance is None:
            cls._instance = super(MongoDAO, cls).__new__(cls)
            try:
                # Lectura de configuración [cite: 152]
                uri = os.getenv("MONGODB_URI")
                db_name = os.getenv("DATABASE_NAME")
                
                # Configuración del cliente con timeout de 5 segundos
                cls._instance.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
                cls._instance.db = cls._instance.client[db_name]
                
                # Verificación de conexión (Ping) 
                cls._instance.client.admin.command('ping')
                cls._instance.connected = True
                print("Conexión exitosa a MongoDB Atlas.")
            except (ConnectionFailure, Exception) as e:
                cls._instance.connected = False
                print(f"Error de conexión a MongoDB: {e}")
        return cls._instance

    def registrar_sesion(self, inicio_ts):
        """Guarda el inicio de una nueva sesión [cite: 142, 159]"""
        if not self.connected: return None
        
        sesion = {
            "fecha_inicio": inicio_ts,
            "tipo": "control_volumen_manos",
            "estado": "activa"
        }
        # Inserta en la colección 'sesiones' [cite: 161]
        return self.db.sesiones.insert_one(sesion).inserted_id

    def finalizar_sesion(self, id_sesion, duracion):
        """Actualiza la sesión con la duración final """
        if not self.connected: return
        
        self.db.sesiones.update_one(
            {"_id": id_sesion},
            {"$set": {"duracion_segundos": duracion, "estado": "finalizada"}}
        )

    def guardar_evento_volumen(self, vol_anterior, vol_nuevo, distancia):
        """Registra cada cambio de volumen detectado [cite: 142, 160]"""
        if not self.connected: return
        
        evento = {
            "timestamp": time.time(),
            "volumen_anterior": vol_anterior,
            "volumen_nuevo": vol_nuevo,
            "distancia_dedos": distancia
        }
        # Inserta en la colección 'eventos_volumen' [cite: 161, 169]
        self.db.eventos_volumen.insert_one(evento)