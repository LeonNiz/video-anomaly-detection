import os
from tqdm import tqdm
from scripts.extract_features import extract_video_features
import numpy as np


def detect_anomaly_by_clustering(test_dir, kmeans_model, routine_map, normal_cluster):
    predictions = {}
    video_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.avi')])

    for idx, fname in enumerate(tqdm(video_files, desc="Clustering detection")):
        path = os.path.join(test_dir, fname)
        feats = extract_video_features(path, routine_map)

        distances = kmeans_model.transform(feats)
        anomaly_scores = distances[:, normal_cluster]  # המרחק מהקלאסטר הנורמלי

        # נרמול לסקאלה של 0–1
        scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-6)
        predictions[idx + 1] = scores.tolist()

    return predictions
