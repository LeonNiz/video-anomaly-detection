from scripts.build_routine_map import build_routine_map, build_kmeans_model
from scripts.detect_anomaly_cluster import detect_anomaly_by_clustering
from scripts.evaluate import evaluate_model


def main():
    print("Building routine map...")
    routine_map = build_routine_map("data/training_videos")

    print("Training KMeans model...")
    kmeans_model, normal_cluster = build_kmeans_model("data/training_videos", routine_map)

    print("Running clustering-based detection...")
    predictions = detect_anomaly_by_clustering("data/testing_videos", kmeans_model, routine_map, normal_cluster)

    print("Evaluating results...")
    evaluate_model(predictions, "data/ground_truth")


if __name__ == "__main__":
    main()
