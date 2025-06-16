import cv2
import numpy as np
import os
from tqdm import tqdm
from scipy.stats import entropy


def extract_video_features(video_path, routine_map=None):
    cap = cv2.VideoCapture(video_path)
    features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_float = gray.astype(np.float32)

        motion = (gray_float > 25).astype(np.float32)

        mean_intensity = np.mean(gray_float)
        std_intensity = np.std(gray_float)
        motion_ratio = np.mean(motion)

        if routine_map is not None:
            deviation = np.abs(motion - routine_map)
            deviation_score = np.mean(deviation)
        else:
            deviation_score = 0.0

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)

        hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-6)  # נרמול
        hist_entropy = entropy(hist)

        features.append([
            mean_intensity,
            std_intensity,
            motion_ratio,
            deviation_score,
            laplacian_var,
            edge_density,
            hist_entropy
        ])

    cap.release()
    return np.array(features)
