import cv2
import numpy as np
import os
from tqdm import tqdm


def detect_anomaly(testing_videos_dir, routine_map, threshold=0.4):
    print("Running anomaly detection on test videos...")

    anomaly_scores = {}
    video_files = sorted([f for f in os.listdir(testing_videos_dir) if f.endswith('.avi')])

    for idx, video_file in enumerate(tqdm(video_files, desc="Processing videos")):
        video_path = os.path.join(testing_videos_dir, video_file)
        cap = cv2.VideoCapture(video_path)

        scores = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion = (gray > 25).astype(np.float32)

            deviation = np.abs(motion - routine_map)
            score = deviation.mean()

            scores.append(score)

        cap.release()

        anomaly_scores[idx + 1] = scores

    return anomaly_scores
