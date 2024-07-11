from licensePlateReader import logger
from licensePlateReader.entity.config_entity import DataIngestionConfig
from licensePlateReader.utils.common import download_file, extract_zip_file

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

     
    def download__data_file(self)-> str:
        '''
        Fetch data from the url
        '''
        if self.config.source_URL == -1:
            return
        else:
            download_file(self.config.source_URL, self.config.local_data_file, self.config.root_dir)
        
    
    def extract__data_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        if self.config.source_URL == -1:
            return
        else:
            extract_zip_file(self.config.unzip_dir, self.config.local_data_file)


