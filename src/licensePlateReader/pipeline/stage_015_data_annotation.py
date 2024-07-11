from licensePlateReader.config.configuration import ConfigurationManager
from licensePlateReader.components.data_annotation import DataAnnotation
from licensePlateReader import logger

STAGE_NAME = "Data Annotation stage"

class DataAnnotationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_annotation_config = config.get_data_annotation_config()
        data_annotation = DataAnnotation(config=data_annotation_config)
        data_annotation.process_folder()
        


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataAnnotationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
