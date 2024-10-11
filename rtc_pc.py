import cv2
import torch

# Load YOLOv5 model (use the 'yolov5s.pt' for the small model, 'yolov5m.pt' for medium, etc.)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Video stream URL
video_url = "http://192.168.2.165:8081/?action=stream"

# Open the video stream
cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    # Read frame from the stream
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image")
        break

    # Perform inference
    results = model(frame)

    # Render results on the frame
    frame = results.render()[0]

    # Display the frame
    cv2.imshow("YOLOv5 Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
