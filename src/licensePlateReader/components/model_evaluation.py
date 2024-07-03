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


