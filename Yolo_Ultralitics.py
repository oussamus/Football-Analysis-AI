from ultralytics import YOLO

# Load a pre-trained YOLO model
model = YOLO("yolo11x")

# Start tracking objects in a video
results = model.predict(source="Videos\HIGHLIGHTS_ CS St-Laurent 1-0 CS Longueuil _ Ligue 1 QC.mp4", save=True)
print(results[0]) # Results of the first frame in the video 
print('******************************************************')
for box in results[0].boxes:
    print(box)