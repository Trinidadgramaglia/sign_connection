import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import os
import subprocess
from signconnection.ml_logic.preprocessor import preprocess_features
import numpy as np

path = 'modelv2'

app = FastAPI()
loaded_model = tf.saved_model.load(path)

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las orígenes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = os.path.join(os.path.dirname(__file__), 'tmp')
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)


print(path)
@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    """
    Realiza una predicción de lenguaje de señas a partir de un video.
    """
    try:
        # Guardar el archivo temporalmente
        video_path = os.path.join(TEMP_DIR, video.filename)
        with open(video_path, 'wb') as buffer:
            buffer.write(await video.read())

        # Imprimir los detalles del archivo en la consola
        print(f"Nombre del archivo: {video.filename}")
        print(f"Tipo de archivo: {video.content_type}")

        # Convertir el archivo de WebM a MP4 usando ffmpeg si es necesario
        if video.content_type == "video/webm":
            mp4_path = video_path.replace('.webm', '.mp4')
            command = ['ffmpeg', '-i', video_path, mp4_path]
            subprocess.run(command, check=True)
            # Eliminar el archivo webm original
            os.remove(video_path)
            # Actualizar video_path para apuntar al archivo convertido
            video_path = mp4_path

        # Leer el contenido del archivo convertido para obtener su tamaño real
        with open(video_path, 'rb') as f:
            content = f.read()
        print(f"Tamaño real del archivo: {len(content)} bytes")

        # Preprocesar el video
        video_tensor = preprocess_features(video_path)
        if video_tensor is None:
            raise HTTPException(status_code=500, detail="Error al preprocesar el video")

        # Imprimir el tensor preprocesado en la consola para verificar
        print('video path:', video_path)
        print('Preprocesado:', video_tensor.shape)
        labels= ['Lejos','Agua','Comida','Donde','Hambre','Nombre','Gracias','Ayuda','Dar','Recibir']


        y_pred = loaded_model.serve(video_tensor)
        y_pred_enc = np.argmax(y_pred, axis=1)
        predict = labels[y_pred_enc[0]]
        print('prediccion', predict)

        return {"message": predict}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)




@app.get("/")
def root():
    return {
        'greeting': 'Hello!!'
    }
