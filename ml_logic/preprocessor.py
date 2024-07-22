import cv2
from pathlib import Path
import os
import numpy as np

def cortar_bordes(frame, left_crop, right_crop):
    height, width, _ = frame.shape
    frame_cropped = frame[:, left_crop:width-right_crop, :]
    return frame_cropped

def preprocess_features(path):
    video_folder = Path(path)
    videos_data = []
    video_count = 0
    num_frames = 10
    resolution = (426,240)
    X=[]

    for video_file in video_folder.glob('*.mp4'):
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"No se pudo abrir el video {video_file}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Intervalo de frames para obtener los frames espaciados uniformemente
        interval = total_frames // num_frames

        # Dict de cada video
        video_dict = {
            'video_name': video_file.name,
            'frames': []
        }

        # Extraer frames
        for i in range(num_frames):
            frame_number = i * interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break

            # cajar resolucion
            frame_resized = cv2.resize(frame,resolution, interpolation=cv2.INTER_AREA)

            # convertir el frame a RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            frame_cropped=cortar_bordes(frame_rgb,93,94)

            #matriz de pixeles del frame en la lista de frames
            video_dict['frames'].append(frame_cropped)

        cap.release()
        videos_data.append(video_dict)
        video_count += 1

    #Guardar X (frames)
    for video in videos_data:
        frames = video['frames']
        X.append(frames)
    X = np.array(X)

    #Normalizar
    X=X/255

    return X
