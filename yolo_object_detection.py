import cv2
import socket
import time
import numpy as np
from ultralytics import YOLO


# Target object to track
target = 'button'

x_aim = 0  # Target x-position for centering object
y_aim = 0  # Target y-position for centering object

cam_pos = 0 # 0 - камера вперед, 1 - камера на куб/шарик
claw = 0 # 0 - клешня за жопой, 1 - схватить куб/шарик, 2 - закинуть в корзину, 3 - нажать на кнопку

if (target == 'cube') or (target == 'sphere'):
    x_aim = 320 
    y_aim = 240 

if (target == 'button'):
    x_aim = 320
    y_aim = 240 

if (target == 'basket'):
    x_aim = 150 
    y_aim = 100 


# Load the trained model
model_path = "nano_best.pt"
model = YOLO(model_path)
model.to('cuda')
names = {0: 'basket', 1: 'button', 2: 'cube', 3: 'sphere'}

num = 2  # process every num_th picture
count = 0

# Setup socket connection
robot_ip = '192.168.2.165'  # Robot's IP address
robot_port = 5000  # Port to send data
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Using UDP

# Function to process webcam feed in real-time
def process_webcam():
    global count
    # Open a connection to the webcam
    cap = cv2.VideoCapture('http://192.168.2.165:8080/?action=stream')
    
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

            results = model(frame)
            processed_frame = results[0].plot()  # Draw bounding boxes, labels, etc.

            largest_objects = {}

            for i in range(results[0].boxes.shape[0]):
                x = float(results[0].boxes.xywh[i][0])
                y = float(results[0].boxes.xywh[i][1])
                name = names[int(results[0].boxes.cls[i])]
                #area = (float(results[0].boxes.xyxy[i][2]) - float(results[0].boxes.xyxy[i][0])) * (float(results[0].boxes.xyxy[i][3]) - float(results[0].boxes.xyxy[i][1]))
                area = float(results[0].boxes.xywh[i][2])*float(results[0].boxes.xywh[i][3])
                conf = float(results[0].boxes.conf[i])

                if name in largest_objects:
                    if area > largest_objects[name][3]:
                        if name != "button":
                            largest_objects[name] = [name, x, y, area, conf]
                    else:
                        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        roi = hsv_image[max(0, y - 5):min(frame.shape[0], y + 5), max(0, x - 5):min(frame.shape[1], x + 5)]
                
                        if roi.shape[0] > 0 and roi.shape[1] > 0: # Проверка, что область не пустая
                            mask1 = cv2.inRange(roi, np.array([43, 46, 92]), np.array([71, 200, 255]))
                            mask2 = cv2.inRange(roi, np.array([71, 79, 115]), np.array([112, 255, 255]))
                            combined_mask = cv2.bitwise_or(mask1, mask2)

                            if np.count_nonzero(combined_mask) > 0: # Проверяем, есть ли пиксели в маске
                                largest_objects[name] = [name, x, y, area, conf]
                                
                else:
                    largest_objects[name] = [name, x, y, area, conf]

                
                        

            # Extract coordinates for the target object
            e_x, e_y = None, None
            if target in largest_objects:
                target_data = largest_objects[target]
                x_received = target_data[1]
                y_received = target_data[2]
                
                # Calculate errors in x and y
                e_x = x_aim - x_received
                e_y = y_aim - y_received
                
                cam_pos = 0

                if (e_x < 20) and (e_y < 20) and ((target == 'cube') or (target == 'sphere')):
                    cam_pos = 1

                if (e_x < 20) and (e_y < 10) and ((target == 'cube') or (target == 'sphere')):
                    claw = 1
                    cam_pos = 0

                e_x_top = 0
                e_y_top = 0

                # Send e_x and e_y to the robot via UDP
                message = f"{e_x},{e_y},{e_x_top},{e_y_top}".encode()  # Format: "e_x,e_y"
                sock.sendto(message, (robot_ip, robot_port))
                print('sent: ', message)

            else:
                e_x = 0
                e_y = 0
                e_x_top = 0
                e_y_top = 0
                # Send e_x and e_y to the robot via UDP
                message = f"{e_x},{e_y},{e_x_top},{e_y_top}".encode()  # Format: "e_x,e_y"
                sock.sendto(message, (robot_ip, robot_port))
                print('sent: ', message)

            # Convert dictionary values to a list 
            found_obj = list(largest_objects.values())

            for obj in found_obj:
                cv2.circle(processed_frame, (int(obj[1]), int(obj[2])), 3, (255, 255, 0), 3)

            cv2.circle(processed_frame, (x_aim, y_aim), 20, (0, 255, 255), 3)

            cv2.imshow('YOLO Webcam Inference', processed_frame)
            
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    sock.close()  # Close the socket when done

if __name__ == "__main__":
    print("Running YOLO on webcam. Press 'q' to quit.")
    process_webcam()
