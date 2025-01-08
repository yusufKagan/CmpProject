import torch
import os
import ultralytics
from IPython import display
from ultralytics import YOLO
from ultralytics.data.augment import Compose, RandomHSV, RandomFlip



if __name__ == '__main__':
    #print("CUDA available? =", torch.cuda.is_available())
    #transforms = Compose([
    #    RandomHSV(hgain=0.015, sgain=0.7, vgain=0.4), 
    #    RandomFlip(direction='horizontal', p=0.5),    
    #])
    model = YOLO("yolov8n.yaml")
    
    model.train(
        data="mask.yaml",
        epochs=150,
        augment=True,
    )