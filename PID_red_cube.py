from control_devices.motor import MotorController
import numpy as np
import time
import socket

# PID parameters for x and y axes
Kp_x, Ki_x, Kd_x = 0, 0, 0  
Kp_y, Ki_y, Kd_y = 5, 0, 0

# Initial values for previous errors and integral terms
prev_e_x, prev_e_y = 0, 0
integral_x, integral_y = 0, 0

# Motor controller object
motorController = MotorController()

# Function to calculate wheel values based on forward and turn rates
def calculate_wheel_values(fwd_rate, turn_rate):
    left_val = (fwd_rate - turn_rate) / 10
    right_val = (fwd_rate + turn_rate) / 10
    
    left_val = max(min(left_val, 99), -99)
    right_val = max(min(right_val, 99), -99)
    
    return left_val, right_val

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
    turn_rate, fwd_rate = pid_control(e_x, e_y, dt)
    
    # Calculate the wheel values based on PID outputs
    left_val, right_val = calculate_wheel_values(fwd_rate, turn_rate)
    
    # Send values to the motors
    motorController.set_left_speed(int(left_val))
    motorController.set_right_speed(int(right_val))
    
    # Print the motor control values for debugging
    print(f"Left wheel: {left_val}, Right wheel: {right_val}")

# Setup socket to receive e_x and e_y
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
robot_ip = '0.0.0.0'  # Listen on all interfaces
robot_port = 5000
sock.bind((robot_ip, robot_port))

while True:
    data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
    message = data.decode().split(',')
    
    if len(message) == 2:
        try:
            e_x = float(message[0])
            e_y = float(message[1])

            time_start = time.time()
            time_end = time.time()
            dt = time_end - time_start
            
            # Control the robot based on the current errors
            control_robot(e_x, e_y, dt)
        except ValueError:
            print("Received invalid data.")

# Close the socket when done
sock.close()
