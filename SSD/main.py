from pathlib import Path
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import v2
from torchvision.io import decode_image
from torchvision import tv_tensors
from torch.utils.data import Dataset, DataLoader, Subset,random_split
import os
import xml.etree.ElementTree as ET
import torchvision.models
import torch.optim as opt
from copy import deepcopy
from torch.optim.lr_scheduler import StepLR


plt.rcParams["savefig.bbox"] = 'tight'
imgs = "archive/images/"
labels = "archive/annotations/"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = {"without_mask": 1, "with_mask": 2,"mask_weared_incorrect":3}

class CustomSSDDataset(Dataset):
 
    def __init__(self, imgs,labels, transforms=None, classes=None):
        self.transforms = transforms
        self.images_dir= imgs
        self.annotations_dir=labels
        self.annotations = [f.split('.')[0] for f in os.listdir(self.annotations_dir) if f.endswith('.xml')]
        self.classes = classes
 
    def __len__(self):
        return len(self.annotations)
 
    def __getitem__(self, index):
        annotation = self.annotations[index]
        annotation_path = os.path.join(self.annotations_dir, f"{annotation}.xml")
        image_name, boxes, labels = self.parse_voc_xml(annotation_path)

        img_path = os.path.join(self.images_dir, image_name)
        img = decode_image(img_path, mode="RGB")
        C, H, W = img.size()
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes = tv_tensors.BoundingBoxes(
            boxes,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(H, W)
        )

        labels = torch.as_tensor(labels, dtype=torch.int64)
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
            boxes.append([xmin,ymin,xmax, ymax])
            labels.append(label_idx)
 
        return image_name, boxes, labels
    
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
def plot(images_and_boxes):
    fig, axes = plt.subplots(1, len(images_and_boxes), figsize=(12 * len(images_and_boxes), 8))
    if len(images_and_boxes) == 1:
        axes = [axes]

    for ax, (image, boxes) in zip(axes, images_and_boxes):
        # Display the image
        ax.imshow(image.permute(1, 2, 0).numpy())  # Assuming PyTorch tensor format (C, H, W)
        
        # Draw each bounding box
        for box in boxes:  # Assuming boxes are in XYXY format
            x_min, y_min, x_max, y_max = box.numpy()
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        ax.axis('off')

    plt.tight_layout()
    plt.show()

transform = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.GaussianBlur(kernel_size=(3,3)),
    v2.SanitizeBoundingBoxes(),
    v2.ToDtype(dtype=torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    
])
def collate_fn(batch):
    return tuple(zip(*batch))

dataset = CustomSSDDataset(imgs=imgs,labels=labels,classes=CLASSES,transforms=transform)

train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=4, shuffle=True,collate_fn=collate_fn)
validation_loader = torch.utils.data.DataLoader( val_dataset, batch_size=4,shuffle=False, collate_fn=collate_fn)

model =  torchvision.models.detection.ssd300_vgg16(num_classes=len(CLASSES)+1)
model.to(device=device)
optimizer = opt.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=0.0005)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)


best = torch.jit.script(model)
best_val = -2
epoch = 150
for i in range(epoch):

    model.train()
    running_loss = 0.0
    trainloss=0
    n=0
    for j, (images, targets) in enumerate(train_loader):
        
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images,targets)
        
        closs = loss_dict["classification"]
        rloss = loss_dict["bbox_regression"]


        loss = rloss+2.5*closs
        trainloss += loss.data

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
        
        optimizer.step()
        n = j
    
    #scheduler.step()
    #lr=scheduler.get_last_lr()
    avg_train_loss = trainloss / (n + 1)
    print(f"Epoch {i+1}/{epoch}, Train Loss: {avg_train_loss}")

    if(i%10==0):
            
        val_loss = 0.0
        n_val = 0
        total_iou = 0.0
        pred_boxes = []
        pred_scores = []
        gt_boxes = []
        
        model.eval()
        with torch.no_grad():
            for images, targets in validation_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                predictions = model(images)
                #predictions = model(images).to(device)
                for p, t in zip(predictions, targets):
                    pred_boxes.append(p["boxes"].cpu())
                    pred_scores.append(p["scores"].cpu())
                    gt_boxes.append(t["boxes"].cpu())

                    total_iou += calculate_iou(p["boxes"].cpu(), t["boxes"].cpu()).mean().item()

                n_val += 1

        avg_iou = total_iou / n_val
        mAP = calculate_map(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)

        if mAP > best_val:
            best_val = mAP
            best = torch.jit.script(model)
            print("Best model so far")

        print(f"Epoch {i + 1}/{epoch},  Avg IoU: {avg_iou:.4f}, mAP: {mAP:.4f}")

                            
model_scripted = torch.jit.script(model)
model_scripted.save('deney_b3_l-3.pt')         
best.save('best_deney_b3_l-3.pt')   
print('Finished Training')

         


model = torch.jit.load('deney_b3_l-3.pt')

 
 
def plot_image(img_tensor, annotation):
   
    fig, ax = plt.subplots(1)
    img = img_tensor.cpu().data
 
    # Display the image
    ax.imshow(img.permute(1, 2, 0))
   
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box
 
        # Create a Rectangle patch
        rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
 
        # Add the patch to the Axes
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

        if (i in tt) and (s > 0.30):
            tsts["boxes"].append(b)
            tsts["scores"].append(s)
            tsts["labels"].append(l)

   
    b = deepcopy(target)
    c = {"boxes": b[0]["boxes"].cpu()}

    plot_image(img[0], tsts)
    plot_image(img[0], c)