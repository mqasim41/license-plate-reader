import subprocess
from licensePlateReader.utils.common import update_yaml


video_urls = [
    'https://drive.google.com/file/d/1btsWnfgit7vpBszgWArs0nlYFnJHLqiv/view?usp=drive_link',
    'https://drive.google.com/file/d/1btsWnfgit7vpBszgWArs0nlYFnJHLqiv/view?usp=drive_link'
              ]
for url in video_urls: 
    update_yaml('config/config.yaml','data_ingestion',url)
    subprocess.run(["python3", "main.py"])
