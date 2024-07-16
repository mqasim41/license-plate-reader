from licensePlateReader.config.configuration import BaseModelConfig
from ultralytics import settings
import dagshub
import mlflow
from licensePlateReader.utils.common import download_mlflow_artifact

class PrepareBaseModel:
    def __init__(self, config: BaseModelConfig):
        self.config = config

    def set_mlflow_uri(self):
        settings.update({"mlflow": True})
    
    def get_base_model(self):
        pass


    




