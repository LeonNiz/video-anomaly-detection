import os
from tqdm import tqdm
import numpy as np
import cv2
from scripts.extract_features import extract_video_features


def detect_anomaly_by_clustering(test_dir, kmeans_model, routine_map, normal_cluster):
    predictions = {}
    video_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.avi')])
    os.makedirs("results/heatmaps", exist_ok=True)

    for idx, fname in enumerate(tqdm(video_files, desc="Clustering detection")):
        path = os.path.join(test_dir, fname)
        feats = extract_video_features(path, routine_map)

        distances = kmeans_model.transform(feats)
        anomaly_scores = distances[:, normal_cluster]

        scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-6)
        predictions[idx + 1] = scores.tolist()

        heatmap = np.expand_dims(scores, axis=0)
        heatmap_scaled = np.clip((heatmap * 255), 0, 255).astype(np.uint8)

        heatmap_resized = cv2.resize(heatmap_scaled, (len(scores), 50), interpolation=cv2.INTER_NEAREST)

        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        cv2.imwrite(f"results/heatmaps/video_{idx + 1}_heatmap.png", heatmap_colored)

    return predictions
