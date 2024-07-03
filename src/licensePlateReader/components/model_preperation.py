import os
import urllib.request as request
from zipfile import ZipFile
from pathlib import Path
from licensePlateReader.config.configuration import PrepareBaseModelConfig
import tensorflow as tf
import torch
from ultralytics import YOLO


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = YOLO("yolov8n.pt") 

        self.save_model(path=self.config.base_model_path, model=self.model)


    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):

        model.summary()
        return model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
    


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        torch.save(model.state_dict(), path)


