import json
import time
import os

class TrainLogger:
    def __init__(self,path="logs/train_log.json"):
        os.makedirs(os.path.dirname(path),exist_ok=True)
        self.path=path

    def log(self,**kwargs):
        kwargs['time']=time.time()
        with open(self.path,'a') as f:
            f.write(json.dump(kwargs)+"\n")
