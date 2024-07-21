import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
# import torch
# from datetime import datetime
# import pytz

from signconnection.ml_logic.registry import load_model
from signconnection.ml_logic.preprocessor import preprocess_features

app = FastAPI()
app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/predict")
def predict(video: UploadFile = File(...)):
    """
    Realiza una predicción de lenguaje de señas a partir de un video.
    """
    return {
        'greeting': 'Predict'
    }
    # try:
    #     # Guardar el archivo de video temporalmente
    #     video_path = f'tmp/{video.filename}'
    #     with open(video_path, 'wb') as buffer:
    #         buffer.write(video.file.read())

    #     # Preprocesar el video
    #     video_tensor = preprocess_features(video_path)
    #     video_tensor = video_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')

    #     # Realizar la predicción
    #     with torch.no_grad():
    #         outputs = app.state.model(video_tensor)
    #         predicted_class = torch.argmax(outputs, dim=1).item()

    #     # Mapear la clase predicha al nombre de la seña
    #     sign_dict = {
    #         '013': 'Lejos', '022': 'Agua', '023': 'Comida', '028': 'Donde',
    #         '033': 'Hambre', '039': 'Nombre', '051': 'Gracias', '056': 'Ayuda',
    #         '063': 'Dar', '064': 'Recibir'
    #     }
    #     index_to_label = {index: label for index, label in enumerate(sign_dict.values())}
    #     predicted_label = index_to_label[predicted_class]

    #     return {
    #         'predicted_class': predicted_class,
    #         'predicted_label': predicted_label
    #     }

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/")
def root():
    return {
        'greeting': 'Hello'
    }
