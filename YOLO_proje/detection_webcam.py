from ultralytics import YOLO

model = YOLO("./runs/detect/train13/weights/best.pt")

model.predict(source=0, show=True)  