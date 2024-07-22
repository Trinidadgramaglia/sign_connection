import cv2
from pathlib import Path
import os
import numpy as np

def cortar_bordes(frame, left_crop, right_crop):
    height, width, _ = frame.shape
    frame_cropped = frame[:, left_crop:width-right_crop, :]
    return frame_cropped

def preprocess_features(video_path):
    videos_data = []
    num_frames = 10
    resolution = (426, 240)
    X = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"No se pudo abrir el video {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames

    video_dict = {
        'video_name': os.path.basename(video_path),
        'frames': []
    }

    for i in range(num_frames):
        frame_number = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_cropped = cortar_bordes(frame_rgb, 93, 94)
        video_dict['frames'].append(frame_cropped)

    cap.release()
    videos_data.append(video_dict)

    for video in videos_data:
        frames = video['frames']
        X.append(frames)
    X = np.array(X)
    X = X / 255
    print('hecho!!')
    return X
