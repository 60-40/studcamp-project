from control_devices.motor import MotorController
import socket
import json
import time

# PID parameters for x and y axes
Kp_x, Ki_x, Kd_x = 0.1, 0.01, 0.05  
Kp_y, Ki_y, Kd_y = 0.1, 0.01, 0.05 

# Initial values for previous errors and integral terms
prev_e_x, prev_e_y = 0, 0
integral_x, integral_y = 0, 0

# Target object to track
target = 'cube'
x_aim = 360  # Target x-position for centering object
y_aim = 240  # Target y-position for centering object

# Motor controller object
motorController = MotorController()

# Function to calculate wheel values based on forward and turn rates
def calculate_wheel_values(fwd_rate, turn_rate):
    left_val = fwd_rate - turn_rate
    right_val = fwd_rate + turn_rate
    
    left_val = max(min(left_val, 100), -100)
    right_val = max(min(right_val, 100), -100)
    
    return left_val, right_val

# Function to compute PID output
def pid_control(e_x, e_y, dt):
    global prev_e_x, prev_e_y, integral_x, integral_y
    
    # X-axis PID control
    integral_x += e_x * dt
    derivative_x = (e_x - prev_e_x) / dt
    output_x = Kp_x * e_x + Ki_x * integral_x + Kd_x * derivative_x
    
    # Y-axis PID control
    integral_y += e_y * dt
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
    motorController.set_left_speed(left_val)
    motorController.set_right_speed(right_val)
    
    # Print the motor control values for debugging
    print(f"Left wheel: {left_val}, Right wheel: {right_val}")

# Function to receive object data from PC
def receive_object_data():
    server_ip = '0.0.0.0'  # Robot's IP
    port = 12345
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((server_ip, port))
    sock.listen(1)
    print(f"Listening for data on {server_ip}:{port}")

    conn, addr = sock.accept()
    print(f"Connection established with {addr}")

    try:
        while True:
            # Receive data from the PC
            data = conn.recv(1024)
            if not data:
                break

            # Parse the received data
            largest_objects = json.loads(data.decode('utf-8'))

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

    except KeyboardInterrupt:
        motorController.set_left_speed(0)
        motorController.set_right_speed(0)
        print("Program interrupted. Motors stopped.")
    finally:
        conn.close()


receive_object_data()
