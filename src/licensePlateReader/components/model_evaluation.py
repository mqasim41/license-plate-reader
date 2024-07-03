import tensorflow as tf
from pathlib import Path
from ultralytics import YOLO
import mlflow
import torch
import mlflow.pytorch
from urllib.parse import urlparse
from licensePlateReader.entity.config_entity import EvaluationConfig
from licensePlateReader.utils.common import read_yaml, create_directories,save_json
import dagshub



class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config


    @staticmethod
    def load_model(path: Path):
        model = YOLO(path)
        return model
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self.metrics = self.model.val()
        self.model.export(
            format=self.config.format,
        )
        self.save_score()

    def save_score(self):
        self.results_dict = self.metrics.results_dict
        
        save_json(path=Path("scores.json"), data=self.results_dict)

    
    def log_into_mlflow(self):
        dagshub.init(repo_owner='mqasim41', repo_name='license-plate-reader', mlflow=True)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                self.results_dict
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.pytorch.log_model(self.model, "model", registered_model_name="YOLOv8")
            else:
                mlflow.pytorch.log_model(self.model, "model")
