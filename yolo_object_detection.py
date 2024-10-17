import cv2
import time
from ultralytics import YOLO
from control_devices.motor import MotorController


# PID parameters for x and y axes
Kp_x, Ki_x, Kd_x = 0.15, 0.5, 0.1  
Kp_y, Ki_y, Kd_y = 1.2, 0.5, 0.1 

# Initial values for previous errors and integral terms
prev_e_x, prev_e_y = 0, 0
integral_x, integral_y = 0, 0

# Target object to track
target = 'cube'
x_aim = 320  # Target x-position for centering object
y_aim = 240  # Target y-position for centering object

# Motor controller object
motorController = MotorController()

# Function to calculate wheel values based on forward and turn rates
def calculate_wheel_values(turn_rate, fwd_rate):
    left_val = (fwd_rate - turn_rate)/10
    right_val = (fwd_rate + turn_rate)/10
    
    left_val = max(min(left_val, 100), -100)
    right_val = max(min(right_val, 100), -100)
    
    return left_val, right_val

def pid_control(e_x, e_y, dt):
    global prev_e_x, prev_e_y, integral_x, integral_y
    
    # X-axis PID control
    if (e_x > 0 > prev_e_x) or (e_x < 0 < prev_e_x):
        integral_x = 0  # Reset integral if error sign changes
    integral_x += e_x * dt
    if integral_x > 20:
        integral_x = 20
    if integral_x < -20:
        integral_x = -20
    derivative_x = (e_x - prev_e_x) / dt
    output_x = Kp_x * e_x + Ki_x * integral_x + Kd_x * derivative_x
    
    # Y-axis PID control
    if (e_y > 0 > prev_e_y) or (e_y < 0 < prev_e_y):
        integral_y = 0  # Reset integral if error sign changes
    integral_y += e_y * dt
    if integral_y > 20:
        integral_y = 20
    if integral_y < -20:
        integral_y = -20
    derivative_y = (e_y - prev_e_y) / dt
    output_y = Kp_y * e_y + Ki_y * integral_y + Kd_y * derivative_y
    
    # Save current error as previous error for next loop
    prev_e_x = e_x
    prev_e_y = e_y
    
    return output_x, output_y


# Main loop to control the robot
def control_robot(e_x, e_y, dt):
    # Get the PID outputs for both axes
    fwd_rate, turn_rate = pid_control(e_x, e_y, dt)
    
    # Calculate the wheel values based on PID outputs
    left_val, right_val = calculate_wheel_values(fwd_rate, turn_rate)
    
    # Send values to the motors
    motorController.set_left_speed(int(left_val))
    motorController.set_right_speed(int(right_val))
    
    # Print the motor control values for debugging
    print(f"Left wheel: {left_val}, Right wheel: {right_val}")




# Load the trained model
model_path = "nano_best.pt"
model = YOLO(model_path)

names = {0: 'basket', 1: 'button', 2: 'cube', 3: 'sphere'}

num = 5  # process every num_th picture
count = 0

# Function to process webcam feed in real-time
def process_webcam():
    global count
    # Open a connection to the webcam (0 is the default camera)
    cap = cv2.VideoCapture('http://192.168.2.165:8080/?action=stream')
    # cap = cv2.VideoCapture(0)
    
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

            # Extract coordinates for the target object
            if target in largest_objects:
                target_data = largest_objects[target]
                x_received = target_data[1]
                y_received = target_data[2]
                
                # Calculate errors in x and y
                e_x = x_aim - x_received
                e_y = y_aim - y_received

                # Control the robot based on the current errors
                control_robot(e_x, e_y, 0.1)

            else:
                motorController.set_left_speed(0)
                motorController.set_right_speed(0)

            # Convert dictionary values to a list
            found_obj = list(largest_objects.values())

            for obj in found_obj:
                # print(obj[0], 'x=', obj[1], 'y=', obj[2], 'area=', obj[3], 'conf=', obj[4])
                cv2.circle(processed_frame, (int(obj[1]), int(obj[2])), 3, (255, 255, 0), 3)

            cv2.circle(processed_frame, (x_aim, y_aim), 20, (0, 255, 255), 3)

            # Display the processed frame
            cv2.imshow('YOLO Webcam Inference', processed_frame)
            
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            motorController.set_left_speed(0)
            motorController.set_right_speed(0)
            break
    
    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Running YOLO on webcam. Press 'q' to quit.")
    process_webcam()
