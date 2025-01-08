import cv2
import torch
import numpy as np
import torchvision.transforms.v2 as v2
from PIL import Image

model = torch.jit.load('best300.pt')
model.eval()

tra = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

input_dir = "deney.mp4"
out_dir = "bideo.mp4"

vid = cv2.VideoCapture(input_dir)
fps = int(vid.get(cv2.CAP_PROP_FPS))
f_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
f_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    out_dir, cv2.VideoWriter_fourcc(*'mp4v'), fps, (f_width, f_height)
)

while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break

    input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = tra(input_image).to("cuda")

    with torch.no_grad():
        detections = model([input_tensor])

    for box, score, label in zip(detections[1][0]["boxes"], detections[1][0]["scores"], detections[1][0]["labels"]):
        if score > 0.3:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"Class {label}: {score:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

    out.write(frame)

vid.release()
out.release()
cv2.destroyAllWindows()
