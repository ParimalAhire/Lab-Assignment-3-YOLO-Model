from ultralytics import YOLO
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import cv2
import os
import time

print("YOLO Multi-Task Local Deployment")
print("Using YOLOv8 Nano Models")

print("\nSelect task:")
print("1 - Detection")
print("2 - Classification")
print("3 - Pose")
print("4 - OBB")

choice = input("Enter choice: ")

# Model selection
if choice == "1":
    model = YOLO("models/detection.pt")
    task = "Detection"

elif choice == "2":
    model = YOLO("models/classification.pt")
    task = "Classification"

elif choice == "3":
    model = YOLO("models/pose.pt")
    task = "Pose"

elif choice == "4":
    model = YOLO("models/obb.pt")
    task = "OBB" 

else:
    print("Invalid choice")
    exit()

print(task,"selected")

# File picker
root = Tk()
root.withdraw()
root.attributes('-topmost', True)

file_path = askopenfilename(title="Select Image")

if file_path == "":
    print("No file selected")
    exit()

print("Selected file:", file_path)

# Prediction
start = time.time()

results = model.predict(
    source=file_path,
    save=True,
    conf=0.25
)

end = time.time()

print("\nInference completed")
print("Execution time:", round(end-start,2),"seconds")

# Detection count (only for detection type)
try:
    boxes = results[0].boxes
    if boxes is not None:
        print("Objects detected:", len(boxes))
except:
    pass

# Display result image
save_dir = results[0].save_dir
image_name = os.path.basename(file_path)

result_image_path = os.path.join(save_dir, image_name)

if os.path.exists(result_image_path):

    img = cv2.imread(result_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8,6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(task+" Result")

    plt.show()

else:
    print("Result image not found")

print("\nDeployment completed successfully")