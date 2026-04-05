from ultralytics import YOLO

model = YOLO("../models/classification.pt")

results = model.predict(
    source="../test_images/classification",
    save=True
)

print("Classification testing done")