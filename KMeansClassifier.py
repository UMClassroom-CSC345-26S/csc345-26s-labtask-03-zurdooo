import pandas as pd
from sklearn.cluster import KMeans

# File paths
DATA_FILE = "AllCars.csv"
CLUSTER_CARS_FILE = "ClusterCars.csv"
CLUSTER_ACCURACY_FILE = "ClusterAccuracy.csv"


def main():
    # Load data
    data = pd.read_csv(DATA_FILE)
    features = data[["Volume", "Doors"]]

    # Train KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
    data["Cluster"] = kmeans.fit_predict(features)

    # Find the most common style for each cluster
    cluster_styles = {}
    for cluster_id in range(5):
        cluster_data = data[data["Cluster"] == cluster_id]
        most_common_style = cluster_data["Style"].mode()[0]
        cluster_styles[cluster_id] = most_common_style

    # Assign the cluster style to each car
    data["ClusterStyle"] = data["Cluster"].map(cluster_styles)

    # Save cluster cars to CSV
    cluster_cars = data[["Volume", "Doors", "Style", "ClusterStyle"]]
    cluster_cars.to_csv(CLUSTER_CARS_FILE, index=False)

    # Calculate accuracy for each cluster
    accuracy_rows = []
    for cluster_id, cluster_style in cluster_styles.items():
        cluster_data = data[data["Cluster"] == cluster_id]
        size = len(cluster_data)
        
        correct_predictions = (cluster_data["Style"] == cluster_style).sum()
        accuracy = correct_predictions / size if size > 0 else 0.0
        
        accuracy_rows.append({
            "ClusterStyle": cluster_style,
            "SizeOfCluster": size,
            "Accuracy": round(accuracy, 4),
        })

    # Save accuracy results to CSV
    accuracy_df = pd.DataFrame(accuracy_rows)
    accuracy_df.to_csv(CLUSTER_ACCURACY_FILE, index=False)


if __name__ == "__main__":
    main()
