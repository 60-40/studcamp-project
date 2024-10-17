import cv2
from ultralytics import YOLO
import time

# Load the trained model
model_path = "nano_best.pt"
model = YOLO(model_path)

names = {0: 'basket', 1: 'button', 2: 'cube', 3: 'sphere'}

num = 5 # process every num_th picture
count = 0

# Function to process webcam feed in real-time
def process_webcam():
    global count
    # Open a connection to the webcam (0 is the default camera)
    # cap = cv2.VideoCapture('http://192.168.2.165:8080/?action=stream')
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        count = count + 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        if (count % num == 0):
            count = 0
            # Run YOLO inference on the frame
            results = model(frame)
            for i in range(results[0].boxes.shape[0]):
                print(names[int(results[0].boxes.cls[i])], 'x=', float(results[0].boxes.xywh[i][0]), 'y=', float(results[0].boxes.xywh[i][1]))

            # Render the detections on the frame
            processed_frame = results[0].plot()  # Draw bounding boxes, labels, etc.
            
            # Display the processed frame
            cv2.imshow('YOLO Webcam Inference', processed_frame)
            
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Running YOLO on webcam. Press 'q' to quit.")
    process_webcam()
