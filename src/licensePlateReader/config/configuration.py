import os
from licensePlateReader.constants import *
from licensePlateReader.entity.config_entity import DataIngestionConfig, BaseModelConfig
from licensePlateReader.entity.config_entity import TrainingConfig,EvaluationConfig
from licensePlateReader.utils.common import read_yaml, create_directories


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    def get_prepare_base_model_config(self) -> BaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = BaseModelConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            mlflow_uri="https://dagshub.com/mqasim41/license-plate-reader.mlflow",
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
        )

        return prepare_base_model_config
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            base_model_path=Path(prepare_base_model.updated_base_model_path),
            mlflow_uri=training.mlflow_uri,
            format=training.format,
            params_epochs=params.EPOCHS,
            params_image_size=params.IMAGE_SIZE,
            params_resume=params.RESUME,
        )

        return training_config
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="runs/detect/train/weights/best.pt",
            training_data="artifacts/data_ingestion/data",
        )
        return eval_config
      