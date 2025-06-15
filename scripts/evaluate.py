import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score
import matplotlib.pyplot as plt
from utils.load_labels import load_all_ground_truths


def evaluate_model(predictions, ground_truth_dir, threshold=0.07, save_plot_path='results/roc_curve.png'):
    #מחשב מדדי הערכה בין תוצאות הזיהוי לתיוג האמיתי
    print("Evaluating predictions...")

    # טען את תיבות האמת (frame-level, הופך כל פריים עם bbox ל־1)
    gt_all = load_all_ground_truths(ground_truth_dir)

    y_true = []
    y_score = []

    for vid_idx, scores in predictions.items():
        frame_count = len(scores)
        true_labels = np.zeros(frame_count)

        if vid_idx in gt_all:
            for f_idx in gt_all[vid_idx].keys():
                if 0 <= f_idx - 1 < frame_count:
                    true_labels[f_idx] = 1  # יש תיבת אנומליה בפריים הזה

        y_true.extend(true_labels.tolist())
        y_score.extend(scores)

    print(f"[DEBUG] Total anomaly frames: {int(np.sum(true_labels))} / {len(true_labels)}")
    print(f"[DEBUG] Sample scores: {y_score[:5]}")

    # ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)

    # סף בינארי לקביעת דיוק
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"AUC-ROC:   {auc_score:.3f}")

    # גרף ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_plot_path)
    print(f"ROC curve saved to: {save_plot_path}")
    plt.show()
