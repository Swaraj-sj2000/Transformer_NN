import os
import json
import requests
from GPT.utils.logging import get_logger

logger=get_logger(__name__)
def download_ds(url,output_path):

    path = output_path

    if not os.path.exists(path):
        r=requests.get(url)
        r.raise_for_status()
        with open(path, "w",encoding='utf-8') as f:
            f.write(r.text)
        
        logger.info("Downloading dataset")

    else: logger.warning("Dataset already exists")