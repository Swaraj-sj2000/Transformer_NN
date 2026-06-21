import os
import json
import requests

def download_ds(url,output_path):

    path = output_path

    if not os.path.exists(path):
        r=requests.get(url)
        r.raise_for_status()
        with open(path, "w",encoding='utf-8') as f:
            f.write(r.text)
        print("Downloaded Shakespeare dataset....")

    else: print("Dataset already exists...")
