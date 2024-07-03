import os
import urllib.request as request
from zipfile import ZipFile
import time
from licensePlateReader.entity.config_entity import TrainingConfig
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics import settings
import dagshub
import mlflow

# Update a setting



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = YOLO()
        self.model.load_state_dict(torch.load(self.config.updated_base_model_path))
    
    def set_mlflow_uri(self):
        settings.update({"mlflow": True})
        dagshub.init(repo_owner='mqasim41', repo_name='license-plate-reader', mlflow=True)
        mlflow.set_registry_uri(self.config.mlflow_uri)

    def train(self):
        self.set_mlflow_uri()
        self.model.train(
            data="config/yolo_config.yaml",
            epochs=self.config.params_epochs,
        )

    
    


