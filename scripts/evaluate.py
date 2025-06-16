import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from utils.load_labels import load_all_ground_truths
import os

def evaluate_model(predictions, ground_truth_dir, save_plot_path='results/roc_curve.png'):
    print("Evaluating predictions...")

    gt_all = load_all_ground_truths(ground_truth_dir)
    y_true = []
    y_score = []

    for vid_idx, scores in predictions.items():
        frame_count = len(scores)
        true_labels = np.zeros(frame_count)

        if vid_idx in gt_all:
            for f_idx in gt_all[vid_idx].keys():
                if 0 <= f_idx - 1 < frame_count:
                    true_labels[f_idx] = 1

        y_true.extend(true_labels.tolist())
        y_score.extend(scores)

    print(f"[DEBUG] Total anomaly frames: {int(np.sum(true_labels))} / {len(true_labels)}")
    print(f"[DEBUG] Sample scores: {y_score[:5]}")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)

    best_f1, best_thresh = 0, 0
    for t in np.linspace(0, 1, 100):
        preds = [1 if s >= t else 0 for s in y_score]
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_thresh = t
            best_f1 = score

    y_pred = [1 if s >= best_thresh else 0 for s in y_score]
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f"Best threshold by F1: {best_thresh:.3f} (F1={best_f1:.3f})")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"AUC-ROC:   {auc_score:.3f}")

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
    plt.savefig(save_plot_path)
    print(f"ROC curve saved to: {save_plot_path}")
    plt.show()
