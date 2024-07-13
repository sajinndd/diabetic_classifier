import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
from urllib.parse import urlparse
import sys
import os

os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/sitharamycertification/diabetic_classifier.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'sitharamycertification'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '393f3996ca5c573b028d1d87c4dba8c2c90fa32f'

# Set the experiment name
mlflow.set_experiment("diabetes_experiment")


# Start an MLflow run
with mlflow.start_run() as run:
    # Read the dataset
    dataset_path = 'data/diabetes_data.csv'
    df = pd.read_csv(dataset_path)

    # Check if the file exists and log it as an artifact
    if os.path.exists('data/diabetes_data.csv'):
        print("Dataset file found. Logging artifact...")
        mlflow.log_artifact(dataset_path)
        mlflow.set_tag("Dataset", dataset_path)
    else:
        print("Dataset file not found. Skipping artifact logging.")

    # Print data shape
    print(f'Data shape: {df.shape}')

    # Split data into features and target
    X = df.drop(columns=['diabetes'])
    y = df['diabetes'].values

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Set KNN parameter
    n_neighbors = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    mlflow.log_param('n_neighbors', n_neighbors)

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multi-class classification
    precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multi-class classification
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('training_f1_score', f1)
    mlflow.log_metric('training_precision_score', precision)


    print(f'Model accuracy: {accuracy}')
    print(f'Model F1 Score: {f1}')
    print(f'Model Precision: {precision}')

    # For remote server only
    remote_server_uri = "https://dagshub.com/sitharamycertification/diabetic_classifier.mlflow"
    mlflow.set_tracking_uri(remote_server_uri)

    tracking_uri_type = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_uri_type != "file":
        # Log the model
        mlflow.sklearn.log_model(knn, "knn_model", registered_model_name="DiabeticPredModel") #

    # Debug: Print run details
    print(f'Run ID: {run.info.run_id}')
    print(f'Artifact URI: {mlflow.get_artifact_uri()}')
    print(f'Artifacts logged to: {mlflow.get_artifact_uri("data/diabetes_data.csv")}')

print("Script execution completed.")