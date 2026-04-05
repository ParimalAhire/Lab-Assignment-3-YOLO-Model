from ultralytics import YOLO

model = YOLO("models/detection.pt")

results = model.predict(
    source="test_images/detection",
    imgsz=416,
    conf=0.25,
    save=True
)

print("Testing completed")