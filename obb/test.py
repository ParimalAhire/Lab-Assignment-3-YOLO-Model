from ultralytics import YOLO

model = YOLO("models/obb.pt")

results = model.predict(
    source="test_images/obb",
    imgsz=416,
    save=True
)

print("OBB testing done")