import cv2
import mediapipe as mp
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import Image
import numpy as np
import os
import json
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import pickle
import shutil


# Function to extract the sign from the file name
def get_sign_from_name(file):
    sign_dict = {
    '013': 'Lejos',
    '022': 'Agua',
    '023': 'Comida',
    '028': 'Donde',
    '033': 'Hambre',
    '039': 'Nombre',
    '051': 'Gracias',
    '056': 'Ayuda',
    '063': 'Dar',
    '064': 'Recibir'
    }
    prefix = file.split('_')[0]
    return sign_dict.get(prefix, "unknown")

def process_video2(video_file, frame_indices):
    
    resolution = (640, 360)
    
    # Initialize MediaPipe Hands and Pose
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(os.path.join(videos_folder, video_file))
    if not cap.isOpened():
        print(f"Failed to open video {video_file}")
        return None
    
    category = get_sign_from_name(video_file)
    
    video_data = {
        'video_name': video_file,
        'category': category,
        'frames': []
    }
    cuadro = 0
    for frame_number in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to desired resolution
        frame_resized = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)
       
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)             
            
        # Process hands
        hand_results = hands.process(rgb_frame)

        # Process body
        pose_results = pose.process(rgb_frame)

        frame_data = {'frame': cuadro, 'left_hand': [], 'right_hand': [], 'pose': []}

        # Initialize lists to store landmarks
        left_hand_landmarks = [None] * 21  # Use None initially
        right_hand_landmarks = [None] * 21

        # Check if hand_results.multi_hand_landmarks is not None and iterate over it
        if hand_results.multi_hand_landmarks is not None:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                if hand_landmarks and hand_landmarks.landmark:
                    # Determine if the hand is left or right
                    if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST.value].x < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP.value].x:
                        # Store left hand landmarks
                        left_hand_landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]
                    else:
                        # Store right hand landmarks
                        right_hand_landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]

        else:
            # If no hand landmarks are detected, use zeros
            left_hand_landmarks = [0, 0, 0] * 21
            right_hand_landmarks = [0, 0, 0] * 21

        # Replace None with zeros if no hand was detected
        left_hand_landmarks = [[0, 0, 0] if landmark is None else landmark for landmark in left_hand_landmarks]
        right_hand_landmarks = [[0, 0, 0] if landmark is None else landmark for landmark in right_hand_landmarks]
        

        
        pose_landmarks = [None] * 33  # Assuming 33 landmarks for the full body pose
        
        for pose_landmark in pose_results.pose_landmarks.landmark:
            if pose_landmark:
                # Store pose landmarks
                pose_landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in pose_results.pose_landmarks.landmark]

            else:
                indicate
        
        frame_data['left_hand'].append(left_hand_landmarks)
        frame_data['right_hand'].append(right_hand_landmarks)       
        frame_data['pose'].append(pose_landmarks)   
        
        video_data['frames'].append(frame_data)
        cuadro += 1 

    cap.release()
    return video_data

        
# Function to get landmarks or fill with zeros if not present
def get_landmarks(frame_data, key, num_points):
    if key in frame_data and isinstance(frame_data[key], list) and len(frame_data[key]) > 0:
        try:
            return [coord for point in frame_data[key][0] for coord in point]
        except TypeError:
            # If there's a TypeError, return a list of zeros
            return [0.0] * num_points * 3
    else:
        return [0.0] * num_points * 3


def preprocesado_holistic(path): 
    scalerfile = 'scaler.sav'
    scaler = pickle.load(open(scalerfile, 'rb'))
    videos_folder = path
    video_files = [f for f in os.listdir(videos_folder) if f.endswith('.mp4')]
    num_frames = 9
    
    for video_file in video_files:
        cap = cv2.VideoCapture(os.path.join(videos_folder, video_file))
        if not cap.isOpened():
            print(f"Failed to open video {video_file}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = total_frames // num_frames
        frame_indices = [i * interval for i in range(num_frames)]
        cap.release()

        video_data = process_video2(video_file, frame_indices)

    X_prueba = []
    Y_prueba = []

    video_info = video_data
    category = video_info.get('category', 'unknown')
    frames = video_info.get('frames', 'unknown')
    X_prueba.append(frames)
    Y_prueba.append(category)

    X_data = []
    X_flat2 = []

    for video in X_prueba:
        video_frames = []
        for frame_data in video:
            left_hand = get_landmarks(frame_data, 'left_hand', 21)
            right_hand = get_landmarks(frame_data, 'right_hand', 21)
            pose = get_landmarks(frame_data, 'pose', 33)

            # Concatenate hand and pose data
            frame_flat = left_hand + right_hand + pose
            video_frames.append(frame_flat)
        X_data.append(video_frames)

    X_flat2 = np.array(X_data)

    # Reshape for MinMaxScaler (flatten the array)
    X_prueba_norm = []

    num_videos = X_flat2.shape[0]
    num_frames = X_flat2.shape[1]
    num_landmarks = X_flat2.shape[2]

    X_flat0 = X_flat2.reshape(-1, num_landmarks)

    # Apply Min-Max Normalization
    #scaler = MinMaxScaler()
    X_flat0_normalized = scaler.transform(X_flat0)

    # Reshape back to original shape
    X_prueba_norm = X_flat0_normalized.reshape(num_videos, num_frames, num_landmarks)
    return X_prueba_norm