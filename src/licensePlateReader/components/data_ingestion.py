from licensePlateReader import logger
from licensePlateReader.entity.config_entity import DataIngestionConfig
from licensePlateReader.utils.common import download_file, extract_zip_file
from ultralytics import YOLO
import cv2
import os
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.lpd_model = YOLO(self.config.lpd_path)
        
    def process_video_yolo(self, sampling_rate=30, crop_width=224, crop_height=70):
        """
        Process video frames using the YOLO model and save detected regions as cropped and resized images.
        """
        
        if not os.path.exists(self.config.frames_dir):
            os.makedirs(self.config.frames_dir)

        cap = cv2.VideoCapture(self.config.local_video_file)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {self.config.local_video_file}")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sampling_rate == 0:
                results = self.lpd_model(frame)

                for result in results:
                    for i, box in enumerate(result.boxes):
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                        cropped_img = frame[y1:y2, x1:x2]

                        # Apply grayscale filter
                        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

                        # Resize the image to the desired dimensions
                        resized_img = cv2.resize(gray_img, (crop_width, crop_height))

                        # Save the processed image
                        output_path = os.path.join(self.config.frames_dir, f'frame_{frame_count}_crop_{i}.jpg')
                        cv2.imwrite(output_path, resized_img)
            frame_count += 1

        cap.release()
        print(f"Processed {frame_count // sampling_rate} frames and saved detected regions to '{self.config.frames_dir}'")

     
    def download_data_file(self)-> str:
        '''
        Fetch data from the url
        '''
        if self.config.source_URL == -1:
            return
        else:
            logger.info(f"Downloading data from {self.config.source_URL} into file {self.config.video_dir}")
            if self.config.from_video == True:
                download_file(self.config.source_URL, self.config.local_video_file, self.config.video_dir)
                self.process_video_yolo()
            else:
                download_file(self.config.source_URL, self.config.local_data_file, self.config.root_dir)
        
    
    def extract_data_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        
        if self.config.source_URL == -1:
            return
        else:
            if self.config.from_video == True:
                return
            else:
                extract_zip_file(self.config.root_dir, self.config.local_data_file)


