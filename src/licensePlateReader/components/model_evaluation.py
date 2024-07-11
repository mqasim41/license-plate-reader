from pathlib import Path
from ultralytics import YOLO
from licensePlateReader.entity.config_entity import EvaluationConfig


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config


    @staticmethod
    def load_model(path: Path):
        model = YOLO(path)
        return model
    
    
    def evaluation(self):
        pass
        
    


