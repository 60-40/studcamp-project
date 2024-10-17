import cv2
from ultralytics import YOLO

# Load the trained model
model_path = "nano_best.pt"
model = YOLO(model_path)

x_aim = 360
y_aim = 240

names = {0: 'basket', 1: 'button', 2: 'cube', 3: 'sphere'}

num = 5  # process every num_th picture
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
        count += 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        if count % num == 0:
            count = 0
            # Run YOLO inference on the frame
            results = model(frame)

            # Render the detections on the frame
            processed_frame = results[0].plot()  # Draw bounding boxes, labels, etc.

            # Dictionary to store the largest area instance for each class
            largest_objects = {}

            for i in range(results[0].boxes.shape[0]):
                x = float(results[0].boxes.xywh[i][0])
                y = float(results[0].boxes.xywh[i][1])
                name = names[int(results[0].boxes.cls[i])]
                area = (float(results[0].boxes.xyxy[i][2]) - float(results[0].boxes.xyxy[i][0])) * (float(results[0].boxes.xyxy[i][3]) - float(results[0].boxes.xyxy[i][1]))
                conf = float(results[0].boxes.conf[i])

                # Check if the object class is already in the dictionary
                if name in largest_objects:
                    # Update if the current instance has a larger area
                    if area > largest_objects[name][3]:
                        largest_objects[name] = [name, x, y, area, conf]
                else:
                    # Add the object to the dictionary
                    largest_objects[name] = [name, x, y, area, conf]

            # Convert dictionary values to a list
            found_obj = list(largest_objects.values())

            for obj in found_obj:
                print(obj[0], 'x=', obj[1], 'y=', obj[2], 'area=', obj[3], 'conf=', obj[4])
                cv2.circle(processed_frame, (int(obj[1]), int(obj[2])), 3, (255, 255, 0), 3)

            cv2.circle(processed_frame, (x_aim, y_aim), 3, (0, 255, 255), 3)

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
