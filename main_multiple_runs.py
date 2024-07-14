import subprocess
from licensePlateReader.utils.common import update_yaml
import shutil
import os

model_paths = [
    'yolov10m',
    'runs/detect/train/weights/best.pt',
    'runs/detect/train3/weights/best.pt',
    'runs/detect/train4/weights/best.pt',
    'runs/detect/train5/weights/best.pt',
    'runs/detect/train6/weights/best.pt',
    'runs/detect/train7/weights/best.pt',
    'runs/detect/train8/weights/best.pt',
    'runs/detect/train9/weights/best.pt',
    'runs/detect/train10/weights/best.pt',
    'runs/detect/train11/weights/best.pt',
    'runs/detect/train12/weights/best.pt',
    'runs/detect/train13/weights/best.pt',
    'runs/detect/train14/weights/best.pt',
    'runs/detect/train15/weights/best.pt',
]

video_urls = [
    'https://drive.google.com/file/d/1w0QwtVtKR-dtGCxJIUxsUGmwBu6MaCWa/view?usp=sharing',
    'https://drive.google.com/file/d/1MoEiy6msKNUOtAaxDSUlXsd-eQvTZAtN/view?usp=sharing',
    'https://drive.google.com/file/d/1b2ZJgY8Gdq8yhj5_8eE1zwIFHLpMk6B1/view?usp=sharing',
    'https://drive.google.com/file/d/1l9Qd31874enlCUWPJ8_dW82Vh8MQEUUS/view?usp=sharing',
    'https://drive.google.com/file/d/1T5n3_GtXeCNnQ1CaufTiERr9VeHlpGUm/view?usp=sharing',
    'https://drive.google.com/file/d/10lXlz3twYD_qzLil7KkqAinXra12EM3D/view?usp=sharing',
    'https://drive.google.com/file/d/1IbocTf5_Z0ftTDCXXWgC73Zf2up3MR9t/view?usp=sharing',
    'https://drive.google.com/file/d/1mxPuqGl17tuIfYrAX7_BlHJc1ctdpcm0/view?usp=sharing',
    'https://drive.google.com/file/d/1XrPTu7OkqeGBqU0VldZCVBoFjjgNPG6c/view?usp=sharing',
    'https://drive.google.com/file/d/1as8XDTNtlpLVDvaGp70rBSQ1lHM7YjJb/view?usp=sharing',
    'https://drive.google.com/file/d/1Y8YGojbsAu3y3dxTWdV_orZm_qZ_Ce40/view?usp=sharing',
    'https://drive.google.com/file/d/1GMnig9BhxCvsAZjmqYwfYhsz5Xb2Fkj_/view?usp=sharing',
    'https://drive.google.com/file/d/1gynuH6P5HongQFh94GgH0CLrq2ER_x32/view?usp=sharing',
    'https://drive.google.com/file/d/1WLeTRpN8JF9LR9WXn9xdzi3VGZyjQvFP/view?usp=sharing',
    'https://drive.google.com/file/d/15Vp7e1_i_sFons9AHk93NmfstOI4qnjz/view?usp=sharing',
              ]
for url in range(len(video_urls)): 
    update_yaml('config/config.yaml','data_ingestion','source_URL',video_urls[url])
    update_yaml('config/config.yaml','prepare_base_model','updated_base_model_path',model_paths[url])
    subprocess.run(["python3", "main.py"])

    folder_path = 'artifacts/data_ingestion'
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents have been deleted.")
    else:
        print(f"Folder '{folder_path}' does not exist.")

