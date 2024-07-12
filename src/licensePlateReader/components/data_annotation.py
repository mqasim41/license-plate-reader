import cv2
import os
import random
from licensePlateReader import logger
from licensePlateReader.entity.config_entity import DataAnnotationConfig
from paddleocr import PaddleOCR, draw_ocr
import matplotlib.pyplot as plt
import logging

logging.getLogger('ppocr').setLevel(logging.ERROR)
logging.getLogger('paddleocr').setLevel(logging.ERROR)

class DataAnnotation:
    def __init__(self, config: DataAnnotationConfig):
        self.config = config
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            det_algorithm='DB',
            rec_algorithm='CRNN',
            det_limit_side_len=600,  # Adjust based on your image size
            rec_image_shape="3, 32, 370"
        )

    def char_to_label(self, char):
        """
        Convert character to label number for YOLO format.
        """
        if char.isalpha():
            return ord(char.upper()) - ord('A')  # Labels A-Z as 0-25
        elif char.isdigit():
            return ord(char) - ord('0') + 26  # Labels 0-9 as 26-35
        else:
            return None  # Ignore other characters

    def process_file(self, img_path, image_output_dir, label_output_dir, target_size=None):
        """
        Processes a single image file for OCR and prints YOLO format bounding boxes.
        """
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # Check if the image is already in grayscale
        if len(img.shape) == 2 or img.shape[2] == 1:
            gray_img = img  # Already grayscale
        else:
            # Convert to grayscale (black and white)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize image while maintaining aspect ratio if target_size is specified
        if target_size:
            h, w = gray_img.shape[:2]
            scale = min(target_size[1] / h, target_size[0] / w)
            resized_img = cv2.resize(gray_img, (int(w * scale), int(h * scale)))
        else:
            resized_img = gray_img

        # Perform OCR on the image (Replace with your OCR function call)
        result = self.ocr.ocr(resized_img, cls=True)

        if result is not None and len(result) > 0 and result[0] is not None:
            # Extract detected words and their bounding boxes
            detected_words = result[0]
            # Prepare file names for saving YOLO format and image
            file_name = os.path.splitext(os.path.basename(img_path))[0]
            yolo_file_path = os.path.join(label_output_dir, f"{file_name}.txt")
            img_save_path = os.path.join(image_output_dir, f"{file_name}.jpg")  # Ensure the extension is correct

            # Ensure directories exist
            os.makedirs(os.path.dirname(yolo_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(img_save_path), exist_ok=True)

            # Process each detected word to get character bounding boxes
            char_boxes = []
            char_texts = []
            char_labels = []
            with open(yolo_file_path, 'w') as f:
                for word_info in detected_words:
                    word_box = word_info[0]  # Bounding box for the detected word
                    word_text = word_info[1][0]  # Recognized word text

                    # Calculate the width of the word box
                    word_width = word_box[2][0] - word_box[0][0]  # (x2 - x1)
                    word_height = word_box[2][1] - word_box[0][1]  # (y2 - y1)

                    # Split word into individual characters
                    char_width = word_width / len(word_text)

                    for i, char in enumerate(word_text):
                        # Calculate bounding box for each character
                        char_box = [
                            [word_box[0][0] + i * char_width, word_box[0][1]],  # Top-left
                            [word_box[0][0] + (i + 1) * char_width, word_box[0][1]],  # Top-right
                            [word_box[0][0] + (i + 1) * char_width, word_box[2][1]],  # Bottom-right
                            [word_box[0][0] + i * char_width, word_box[2][1]]  # Bottom-left
                        ]
                        char_boxes.append(char_box)
                        char_texts.append(char)

                        # Convert character to label number
                        label = self.char_to_label(char)
                        char_labels.append(label)

                        # Print YOLO format bounding box
                        if label is not None:
                            x_center = (char_box[0][0] + char_box[1][0]) / 2 / resized_img.shape[1]  # x_center
                            y_center = (char_box[0][1] + char_box[2][1]) / 2 / resized_img.shape[0]  # y_center
                            width = (char_box[1][0] - char_box[0][0]) / resized_img.shape[1]  # width
                            height = (char_box[2][1] - char_box[1][1]) / resized_img.shape[0]  # height

                            f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            cv2.imwrite(img_save_path, resized_img)

        else:
            print(f"No text detected in {img_path}")

    def process_folder(self, target_size=None):
        """
        Processes all image files in a folder for OCR.
        """
        if self.config.from_video == False:
            return

        target_size = (self.config.image_size[0], self.config.image_size[1])
        
        all_files = [f for f in os.listdir(self.config.frames_dir) if os.path.isfile(os.path.join(self.config.frames_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(all_files)
        
        val_split = int(0.1 * len(all_files))
        val_files = all_files[:val_split]
        train_files = all_files[val_split:]
        
        train_image_dir = os.path.join(self.config.data_save_dir, 'images', 'train')
        val_image_dir = os.path.join(self.config.data_save_dir, 'images', 'val')
        train_label_dir = os.path.join(self.config.data_save_dir, 'labels', 'train')
        val_label_dir = os.path.join(self.config.data_save_dir, 'labels', 'val')

        for filename in train_files:
            img_path = os.path.join(self.config.frames_dir, filename)
            self.process_file(img_path, train_image_dir, train_label_dir, target_size=target_size)
            print(f"Processed {filename} for training")

        for filename in val_files:
            img_path = os.path.join(self.config.frames_dir, filename)
            self.process_file(img_path, val_image_dir, val_label_dir, target_size=target_size)
            print(f"Processed {filename} for validation")
