import os
import urllib.request as request
from zipfile import ZipFile
import time
from licensePlateReader.entity.config_entity import TrainingConfig
from pathlib import Path
import torch
from ultralytics import YOLO


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = YOLO()
        self.model.load_state_dict(torch.load(self.config.updated_base_model_path))
       

    
    def train(self):

        self.model.train(
            data="config/yolo_config.yaml",
            epochs=self.config.params_epochs,
        )
    


