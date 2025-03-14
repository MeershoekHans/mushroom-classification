import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import urllib.request
import os
import io

# Function to download the UCI Mushroom dataset
def download_mushroom_dataset():
    print("Downloading UCI Mushroom dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    
    # Create a directory for the data if it doesn't exist
    os.makedirs("mushroom_data", exist_ok=True)
    
    # Download the data
    local_filename = os.path.join("mushroom_data", "mushroom.data")
    urllib.request.urlretrieve(url, local_filename)
    
    # Define column names based on the UCI dataset description
    column_names = [
        "class",
        "cap-shape", 
        "cap-surface", 
        "cap-color", 
        "bruises", 
        "odor", 
        "gill-attachment", 
        "gill-spacing", 
        "gill-size", 
        "gill-color", 
        "stalk-shape", 
        "stalk-root", 
        "stalk-surface-above-ring", 
        "stalk-surface-below-ring", 
        "stalk-color-above-ring", 
        "stalk-color-below-ring", 
        "veil-type", 
        "veil-color", 
        "ring-number", 
        "ring-type", 
        "spore-print-color", 
        "population", 
        "habitat"
    ]
    
    # Read the data
    df = pd.read_csv(local_filename, header=None, names=column_names)
    
    # Convert class labels: 'e' (edible) to 0, 'p' (poisonous) to 1
    df["class"] = df["class"].map({"e": 0, "p": 1})
    
    # Create train and test datasets
    train_df = df.sample(frac=0.8, random_state=123)
    test_df = df.drop(train_df.index)
    
    # Add an Id column for consistency with the original code
    train_df = train_df.reset_index().rename(columns={"index": "Id"})
    test_df = test_df.reset_index().rename(columns={"index": "Id"})
    
    # Save the datasets
    train_df.to_csv(os.path.join("mushroom_data", "train_data.csv"), index=False)
    test_df.to_csv(os.path.join("mushroom_data", "test_data.csv"), index=False)
    
    print(f"Dataset downloaded and processed. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df

# Main function to run the mushroom classification
def main():
    # Download and prepare the dataset
    train_df, test_df = download_mushroom_dataset()
    
    print("\nTraining a mushroom classification model...")
    
    # Prepare data: separate features and target
    X = train_df.drop(['Id', 'class'], axis=1)
    y = train_df['class']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    # Print dataset shapes
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    
    # Create a pipeline with preprocessing and model
    print("\nCreating and fitting the initial model...")
    pipe = make_pipeline(
        OneHotEncoder(handle_unknown="infrequent_if_exist"),
        RandomForestClassifier(n_estimators=500, random_state=123)
    )
    
    # Fit the model
    pipe.fit(X_train, y_train)
    
    # Make predictions and calculate accuracy
    y_pred = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Initial model accuracy: {accuracy:.4f}")
    
    # Define parameter grid for hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    param_grid = {
        'randomforestclassifier__n_estimators': [100],
        'randomforestclassifier__max_depth': [10],
        'randomforestclassifier__class_weight': [{0: 1, 1: 5}]  # Class weights to prioritize correctly identifying poisonous mushrooms
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=pipe, param_grid=param_grid, scoring='recall', cv=5, n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Make predictions with the best model
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"\nBest model performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Make predictions on the test dataset
    print("\nMaking predictions on the test dataset...")
    feature_cols = X_train.columns
    test_data_for_prediction = test_df[feature_cols]
    test_predictions = best_model.predict(test_data_for_prediction)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({'Id': test_df['Id'], 'poisonous': test_predictions})
    submission_file = os.path.join("mushroom_data", "mushroom_predictions.csv")
    submission_df.to_csv(submission_file, index=False)
    
    print(f"\nPredictions saved to {submission_file}")
    print(f"Number of poisonous mushrooms predicted: {sum(test_predictions)}")
    print(f"Total number of mushrooms in test set: {len(test_predictions)}")
    
    print("\nMushroom classification completed successfully!")

if __name__ == "__main__":
    main()