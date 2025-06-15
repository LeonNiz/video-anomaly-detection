import cv2
import numpy as np
import os
from tqdm import tqdm


def extract_video_features(video_path, routine_map=None):
    cap = cv2.VideoCapture(video_path)
    features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        motion = (gray > 25).astype(np.float32)

        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        motion_ratio = np.mean(motion)

        if routine_map is not None:
            deviation = np.abs(motion - routine_map)
            deviation_score = np.mean(deviation)
        else:
            deviation_score = 0.0

        features.append([mean_intensity, std_intensity, motion_ratio, deviation_score])

    cap.release()
    return np.array(features)
