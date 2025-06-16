import cv2
import numpy as np
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scripts.extract_features import extract_video_features


def build_routine_map(training_videos_dir):
    print("Building routine map from training videos...")
    routine_map = None
    video_files = sorted([f for f in os.listdir(training_videos_dir) if f.endswith('.avi')])

    for video_file in tqdm(video_files):
        video_path = os.path.join(training_videos_dir, video_file)
        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            binary_motion = (gray > 25).astype(np.float32)

            if routine_map is None:
                routine_map = np.zeros_like(gray, dtype=np.float32)

            routine_map += binary_motion

        cap.release()

    if routine_map.max() > 0:
        routine_map /= routine_map.max()

    return routine_map


def build_kmeans_model(training_dir, routine_map, k_range=(2, 6)):
    print("Extracting features for KMeans clustering...")
    all_features = []

    for fname in sorted(os.listdir(training_dir)):
        if not fname.endswith('.avi'):
            continue
        path = os.path.join(training_dir, fname)
        feats = extract_video_features(path, routine_map)
        all_features.extend(feats)

    all_features = np.array(all_features)
    print(f"Total features: {len(all_features)}")

    best_k = k_range[0]
    best_score = -1
    best_model = None

    print("Searching best K using Silhouette Score...")
    for k in range(k_range[0], k_range[1] + 1):
        model = KMeans(n_clusters=k, random_state=0).fit(all_features)
        score = silhouette_score(all_features, model.labels_)
        print(f"  K={k}, silhouette score={score:.4f}")
        if score > best_score:
            best_k = k
            best_score = score
            best_model = model

    print(f"Selected best K={best_k} with silhouette score={best_score:.4f}")

    # קלאסטר הנורמלי = עם הכי הרבה מופעים
    unique, counts = np.unique(best_model.labels_, return_counts=True)
    normal_cluster = unique[np.argmax(counts)]

    return best_model, normal_cluster
