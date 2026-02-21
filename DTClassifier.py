import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# File paths
DATA_FILE = "AllCars.csv"
TREE_IMAGE_FILE = "TreeCars.png"
TREE_CSV_FILE = "TreeCars.csv"


def main():
    # Load data
    data = pd.read_csv(DATA_FILE)
    features = data[["Volume", "Doors"]]
    labels = data["Style"]

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    # Train the decision tree classifier
    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(x_train, y_train)

    # Make predictions
    predictions = classifier.predict(x_test)

    # Save results to CSV
    results = x_test.copy()
    results["Style"] = list(y_test)
    results["PredictedStyle"] = list(predictions)
    results = results[["Volume", "Doors", "Style", "PredictedStyle"]]

    # Calculate accuracy
    correct_predictions = sum(predictions == y_test)
    total_predictions = len(y_test)
    accuracy = correct_predictions / total_predictions

    # Add accuracy row to results
    results.loc["Accuracy"] = {
        "Volume": "",
        "Doors": "",
        "Style": "Accuracy",
        "PredictedStyle": f"{accuracy:.4f}",
    }
    results.to_csv(TREE_CSV_FILE, index=False)

    # Plot and save the decision tree, use matplot lib tree feature
    plt.figure(figsize=(12, 8))
    plot_tree(
        classifier,
        feature_names=["Volume", "Doors"],
        class_names=sorted(labels.unique()),
        filled=True,
        rounded=True,
    )
    plt.tight_layout()
    plt.savefig(TREE_IMAGE_FILE, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
