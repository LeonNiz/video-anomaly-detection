import scipy.io
import os
import numpy as np


def load_all_ground_truths(ground_truth_dir, debug=False):
    #טוענים את כל הקבצי תיוג מתקיית ground_truth ומחזיר מילון
    all_labels = {}

    for filename in sorted(os.listdir(ground_truth_dir)):
        if not filename.endswith('_label.mat'):
            continue

        video_idx = int(filename.split('_')[0])
        path = os.path.join(ground_truth_dir, filename)
        mat = scipy.io.loadmat(path)

        if 'volLabel' not in mat:
            print(f"Warning: volLabel not found in {filename}")
            continue

        vol_label = mat['volLabel']
        frame_dict = {}

        for frame_idx in range(vol_label.shape[1]):
            boxes = vol_label[0, frame_idx]
            bboxes = []

            if boxes.size == 0:
                continue

            # טפל גם במקרה שבו זה קובץ עם box בודד וגם בריבוי תיבות
            if boxes.ndim == 2:
                for b in boxes:
                    b = np.array(b).flatten()
                    if b.size >= 4:
                        x1, y1, x2, y2 = map(int, b[:4])
                        bboxes.append([x1, y1, x2 - x1, y2 - y1])
            elif boxes.ndim == 1 and boxes.size >= 4:
                b = boxes.flatten()
                x1, y1, x2, y2 = map(int, b[:4])
                bboxes.append([x1, y1, x2 - x1, y2 - y1])

            if bboxes:
                frame_dict[frame_idx] = bboxes  # פריים מתחיל ב־0 (כמו OpenCV)

        if debug:
            print(f"Video {video_idx}: {len(frame_dict)} annotated frames")
            some = list(frame_dict.items())[:3]
            for frame, boxes in some:
                print(f" Frame {frame}: {boxes}")

        all_labels[video_idx] = frame_dict

    return all_labels
