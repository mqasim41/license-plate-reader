from licensePlateReader.entity.config_entity import TrainingConfig
from ultralytics import YOLO
from ultralytics import settings
import dagshub
import mlflow
class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = YOLO(self.config.base_model_path)
        
    
    def set_mlflow_uri(self):
        settings.update({"mlflow": True})
        dagshub.init(repo_owner='mqasim41', repo_name='license-plate-reader', mlflow=True)
        mlflow.set_registry_uri(self.config.mlflow_uri)

    def train(self):
        self.model.train(
            data="config/yolo_config.yaml",
            epochs=self.config.params_epochs,
            imgsz=self.config.params_image_size,
            resume = self.config.params_resume
        )
        self.model.val()
        self.model.export(
            format=self.config.format,
        )

    
    


