from control_devices.motor import MotorController
import time

# PID parameters for x and y axes
Kp_x, Ki_x, Kd_x = 0.1, 0.01, 0.05  
Kp_y, Ki_y, Kd_y = 0.1, 0.01, 0.05 

# Initial values for previous errors and integral terms
prev_e_x, prev_e_y = 0, 0
integral_x, integral_y = 0, 0

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


# Assume some error values for e_x and e_y (you will eventually calculate these based on real sensor data)
e_x, e_y = 10, -5  # Example errors in x and y axes

# Control loop
try:
    while True:
        # Get the time step (dt), you can adjust this based on your loop frequency
        dt = 0.1  # Assuming a control loop running at 10Hz (0.1s per loop)
        
        # Control the robot based on current errors
        control_robot(e_x, e_y, dt)
        
        # Simulate a loop delay (assuming 10Hz control loop)
        time.sleep(dt)
except KeyboardInterrupt:
    # Stop the motors when the program is interrupted
    motorController.set_left_speed(0)
    motorController.set_right_speed(0)
    print("Program interrupted. Motors stopped.")
