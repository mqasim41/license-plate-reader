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
        dagshub.init(repo_owner='mqasim41', repo_name='license-plate-reader', mlflow=True)
        mlflow.set_registry_uri(self.config.mlflow_uri)
    
    def get_base_model(self):
        self.set_mlflow_uri()
        download_mlflow_artifact(self.config.source_URL,local_dir=self.config.base_model_path)


    




