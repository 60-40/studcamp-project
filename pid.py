import numpy as np
import time
import socket
from xr_motor import RobotDirection

go = RobotDirection()

t_del_top = 2 # seconds
max_speed = 90 # pix/s

left_prev = 0
right_prev = 0

# PID parameters for x and y axes
# Kp_x, Ki_x, Kd_x = 1, 150, 0.001
# Kp_y, Ki_y, Kd_y = 4, 1000, 0.002
Kp_x, Ki_x, Kd_x = 1, 1, 0.08
Kp_y, Ki_y, Kd_y = 4, 5, 0.5


# PID parameters for top view
Kp_x_top, Ki_x_top, Kd_x_top = 20, 0, 0  
Kp_y_top, Ki_y_top, Kd_y_top = 2, 0, 0

# Initial values for previous errors and integral terms
prev_e_x, prev_e_y = 0, 0
integral_x, integral_y = 0, 0
prev_e_x_top, prev_e_y_top = 0, 0
integral_x_top, integral_y_top = 0, 0
time_end = 0
dt = 0 
dt_top = 0

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
	derivative_x = (e_x - prev_e_x) / dt if dt != 0 else 0
	output_x = Kp_x * e_x + Ki_x * integral_x + Kd_x * derivative_x

	if e_x == 0:
		integral_x = 0

	print('turn ', Kp_x * e_x, Ki_x * integral_x, Kd_x * derivative_x)
    
	# Y-axis PID control
	integral_y += e_y * dt
	derivative_y = (e_y - prev_e_y) / dt if dt != 0 else 0
	output_y = Kp_y * e_y + Ki_y * integral_y + Kd_y * derivative_y
	print('forward ', Kp_y * e_y, Ki_y * integral_y, Kd_y * derivative_y)

	if e_y == 0:
		integral_y = 0
    
	# Save current error as previous error for next loop
	prev_e_x = e_x
	prev_e_y = e_y

	return output_x, output_y

def pid_control_top(e_x_top, e_y_top, dt_top):
    global prev_e_x_top, prev_e_y_top, integral_x_top, integral_y_top
    
    # X-axis PID control
    integral_x_top += e_x_top * dt_top
    derivative_x_top = (e_x_top - prev_e_x) / dt_top
    output_x_top = Kp_x_top * e_x_top + Ki_x_top * integral_x_top + Kd_x_top * derivative_x_top
    
    # Y-axis PID control
    integral_y_top += e_y_top * dt_top
    derivative_y_top = (e_y_top - prev_e_y_top) / dt_top
    output_y_top = Kp_y_top * e_y_top + Ki_y_top * integral_y_top + Kd_y_top * derivative_y_top
    
    # Save current error as previous error for next loop
    prev_e_x_top = e_x_top
    prev_e_y_top = e_y_top

    return output_x_top, output_y_top

# Main loop to control the robot
def control_robot(e_x, e_y, dt):
	if dt > 1:
		dt = 0

	# Get the PID outputs for both axes
	turn_rate, fwd_rate = pid_control(e_x, e_y, dt)
    
	# Calculate the wheel values based on PID outputs
	left_val, right_val = calculate_wheel_values(fwd_rate, turn_rate)
    
	# Send values to the motors
	go.set_speed(1, int(left_val))
	go.set_speed(2, int(right_val))


def control_robot_top(e_x_top, e_y_top, dt):
	global left_prev, right_prev

	if dt > 1:
		dt = 0
	
	e_x_new = e_x_top - t_del_top * max_speed * (left_prev - right_prev)/2
	e_y_new = e_y_top - t_del_top * max_speed * (left_prev + right_prev)/2

    # Get the PID outputs for both axes
	turn_rate, fwd_rate = pid_control_top(e_x_new, e_y_new, dt)
    
    # Calculate the wheel values based on PID outputs
	left_val, right_val = calculate_wheel_values(fwd_rate, turn_rate)
    
	left_prev = left_val
	right_prev = right_val
    # Send values to the motors
	go.set_speed(1, int(left_val))
	go.set_speed(2, int(right_val))
    


def start_pid_server():

# Setup socket to receive e_x and e_y
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	robot_ip = '0.0.0.0'  # Listen on all interfaces
	robot_port = 5000
	sock.bind((robot_ip, robot_port))

	while True:
		data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
		message = data.decode().split(',')
		dt = 0
   
		if len(message) == 4:
			try:
				if dt > 1: dt = 0
				# if dt == 0: dt = 0.0001
				e_x = float(message[0])
				e_y = float(message[1])
				e_x_top = float(message[2])
				e_y_top = float(message[3])
				
#
				# time_start = time.time()
				control_robot(e_x, e_y, 0.05)
				#control_robot_top(e_x_top, e_y_top, dt)
				# time_end = time.time()
				# dt = time_end - time_start
            # Control the robot based on the current errors
			except Exception as ex:
				print(ex)
#            print("Received invalid data.")

# Close the socket when done
