import os

os.environ["MLFLOW_TRACKING_URI"] = '/content/content/MyDrive/ma-information-extraction-codebase/graph_networks/logs/mlflow'

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = "Information Extraction"

