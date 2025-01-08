import numpy as np
import pandas as pd 
import torch
import torchvision
from torchvision.transforms import v2
import torchvision.models.detection as models
import xml.etree.ElementTree as ET
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset,random_split
import torch.optim as opt
from sklearn.model_selection import train_test_split
import torch.nn as nn
import cv2 
import cvzone
import torchvision.ops

imgs = "archive/images/"
labels = "archive/annotations/"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = {"without_mask": 1, "with_mask": 2,"mask_weared_incorrect":3}

class CustomSSDDataset(Dataset):
 
    def __init__(self, imgs,labels, transforms=None, classes=None, device=None):
 

        self.device= device
        self.transforms = transforms
        
        self.images_dir= imgs
        self.annotations_dir= labels
 
        self.annotations = [f.split('.')[0] for f in os.listdir(self.annotations_dir) if f.endswith('.xml')]
 
        self.classes = classes
 
    def __len__(self):
        return len(self.annotations)
 
    def __getitem__(self, index):
 
        annotation      = self.annotations[index]
        annotation_path = os.path.join(self.annotations_dir, f"{annotation}.xml")
        image_name, boxes, labels = self.parse_voc_xml(annotation_path)
 
        img_path   = os.path.join(self.images_dir, image_name)
        img        = Image.open(img_path).convert("RGB")
 
        boxes  = torch.as_tensor(boxes, dtype=torch.float32, device=self.device)
        labels = torch.as_tensor(labels, dtype=torch.int64, device=self.device)
 
        target = {"boxes": boxes, "labels": labels}
 
        if self.transforms:
            img, target = self.transforms(img, target)
 
        return img, target
 
    def parse_voc_xml(self, xml_file):
 
        tree = ET.parse(xml_file)
        root = tree.getroot()
 
        image_name = root.find("filename").text
 
        boxes  = []
        labels = []
 
        for obj in root.findall("object"):
 
            label = obj.find("name").text
            if label not in self.classes:
                continue
 
            label_idx = self.classes[label]
 
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_idx)
 
        return image_name, boxes, labels
    
data_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=(300, 300)),
    v2.RandomCrop(size=(240, 240)),
    v2.Resize(size=(300, 300)),
    v2.Lambda(lambda x : x.to(device)),
    v2.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1 
    ),
    v2.ToDtype(torch.float32, scale=True),
])


def collate_fn(batch):
    return tuple(zip(*batch))

dataset = CustomSSDDataset(imgs=imgs,labels=labels,classes=CLASSES,transforms=data_transform,device=device)

train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=4, shuffle=True,collate_fn=collate_fn)
validation_loader = torch.utils.data.DataLoader( val_dataset, batch_size=4,shuffle=False, collate_fn=collate_fn)

torch.cuda.is_available()
model = models.ssd300_vgg16(num_classes=len(CLASSES)+1)
model.to(device=device)
optimizer = opt.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)

from torchvision.ops import box_iou

def calculate_iou(pred_boxes, gt_boxes):
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return torch.tensor(0.0)
    iou_matrix = box_iou(pred_boxes, gt_boxes)
    return iou_matrix

def calculate_map(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    matched_boxes = 0
    total_gt_boxes = sum(len(gt) for gt in gt_boxes)
    for pred, scores, gt in zip(pred_boxes, pred_scores, gt_boxes):
        if len(pred) == 0 or len(gt) == 0:
            continue
        iou_matrix = box_iou(pred, gt)
        max_iou, _ = iou_matrix.max(dim=1)
        matched_boxes += (max_iou >= iou_threshold).sum().item()
    precision = matched_boxes / sum(len(pred) for pred in pred_boxes)
    recall = matched_boxes / total_gt_boxes
    return precision * recall 

loss_log = []
val_loss_log = []
iou_log = []
map_log = []

best = torch.jit.script(model)
best_val = -2
epoch = 3

for i in range(epoch):
    model.train()
    running_loss = 0.0
    trainloss = 0
    n = 0
    for j, (images, targets) in enumerate(train_loader):
        loss_dict = model(images, targets)
        closs = loss_dict["classification"]
        rloss = loss_dict["bbox_regression"]

        loss = rloss + closs
        trainloss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n = j

    avg_train_loss = trainloss / (n + 1)
    print(f"Epoch {i + 1}/{epoch}, Train Loss: {avg_train_loss}")
    loss_log.append(avg_train_loss)

    model.eval()
    val_loss = 0.0
    n_val = 0
    total_iou = 0.0
    pred_boxes = []
    pred_scores = []
    gt_boxes = []

    with torch.no_grad():
        for images, targets in validation_loader:
            predictions = model(images)
            for p, t in zip(predictions, targets):
                pred_boxes.append(p["boxes"].cpu())
                pred_scores.append(p["scores"].cpu())
                gt_boxes.append(t["boxes"].cpu())

                total_iou += calculate_iou(p["boxes"].cpu(), t["boxes"].cpu()).mean().item()

            n_val += 1

    avg_iou = total_iou / n_val
    mAP = calculate_map(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)

    if mAP > best_val and i >= 0:
        best_val = mAP
        best = torch.jit.script(model)
        print("Best model so far")

    print(f"Epoch {i + 1}/{epoch},  Avg IoU: {avg_iou:.4f}, mAP: {mAP:.4f}")
    iou_log.append(avg_iou)
    map_log.append(mAP)

print("Finished Training")
                            
model_scripted = torch.jit.script(model)
model_scripted.save('last3_2.pt')         
best.save('best3_2.pt')   

import matplotlib.pyplot  as plt
import matplotlib.patches as patches

#plt.figure(figsize=(10, 6))
#loss_log = [_data.cuda() for _data in loss_log] 
#val_loss_log = [_data.cuda() for _data in val_loss_log] 
#plt.plot(range(1, epoch + 1), loss_log, label="Train Loss", color="blue", marker="o")
#plt.plot(range(1, epoch + 1), val_loss_log, label="Validation Loss", color="orange", marker="x")
#plt.xlabel("Epochs")
#plt.ylabel("Loss")
#plt.title("Training and Validation Loss Over Epochs")
#plt.legend()
#plt.grid(True)
#plt.tight_layout()
#plt.show()

model = torch.jit.load('best3_2.pt')

from copy import deepcopy
 
 
def plot_image(img_tensor, annotation):
   
    fig, ax = plt.subplots(1)
    img = img_tensor.cpu().data
    ax.imshow(img.permute(1, 2, 0))
   
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box
 
        rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
 
        ax.add_patch(rect)
 
    plt.show()

model.eval()
for img, target in validation_loader:
   
    preds = model([i.to(device) for i in img])

    a = {k: v.cpu().detach() for k, v in preds[1][0].items()}

    tt = torchvision.ops.nms(
        a["boxes"],
        a["scores"],
        0.5
    )

    tsts = {
        "boxes": [],
        "scores": [],
        "labels": []
    }

    for i, (b, s, l) in enumerate(zip(a["boxes"], a["scores"], a["labels"])):

        if (i in tt) and (s > 0.10):
            tsts["boxes"].append(b)
            tsts["scores"].append(s)
            tsts["labels"].append(l)

   
    b = deepcopy(target)
    c = {"boxes": b[0]["boxes"].cpu()}

    plot_image(img[0], tsts)
    plot_image(img[0], c)
