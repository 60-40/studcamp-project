import cv2
from ultralytics import YOLO

# Load the trained model
model_path = "best.pt"
model = YOLO(model_path)

# Function to process webcam feed in real-time
def process_webcam():
    # Open a connection to the webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Run YOLO inference on the frame
        results = model(frame)
        
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
