from ultralytics import YOLO

model = YOLO("./runs/detect/train13/weights/best.pt")

results = model.predict(source="https://www.youtube.com/watch?v=4xFY3aPF7E4",imgsz=640,save=True)


