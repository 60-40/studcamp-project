import torch
import cv2
import numpy as np

# Load YOLOv5 model (version 7) - YOLOv5s (small) used as an example
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# OpenCV for video capture (from webcam 0)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert frame to a format YOLOv5 expects (RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform object detection
    results = model(img_rgb)
    
    # Convert results into numpy array and draw bounding boxes
    results.render()  # Render boxes on the frame
    
    # Display the resulting frame with bounding boxes
    cv2.imshow('YOLOv5 Real-Time Detection', np.squeeze(results.render()))

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture when done
cap.release()
cv2.destroyAllWindows()
