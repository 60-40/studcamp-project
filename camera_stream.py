#!/bin/python3
import cv2
import socket
import numpy as np
import struct

# Set up camera capture
cap = cv2.VideoCapture(1)  # USB camera

# Set the desired resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set up the UDP connection
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_ip = "192.168.2.75"  # IP address of the laptop
udp_port = 12345  # Arbitrary port number

# Maximum packet size (UDP MTU is 65507 bytes for IPv4, minus header overhead)
MAX_PACKET_SIZE = 65507
HEADER_SIZE = 2  # 1 byte for frame_id and 1 byte for chunk_id
MAX_CHUNK_SIZE = MAX_PACKET_SIZE - HEADER_SIZE  # Ensure data and headers fit in packet

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break
    
    # Encode the frame as a JPEG
    result, encoded_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not result:
        continue

    # Convert encoded frame to bytes
    data = encoded_frame.tobytes()

    # Frame ID to handle reassembly on the receiver side
    frame_id = struct.pack('B', 1)  # Use 1 byte to indicate frame ID (e.g., 0 or 1)

    # Calculate number of chunks needed
    num_chunks = (len(data) + MAX_CHUNK_SIZE - 1) // MAX_CHUNK_SIZE  # Ceiling division

    for i in range(num_chunks):
        # Extract the chunk of data
        chunk = data[i * MAX_CHUNK_SIZE:(i + 1) * MAX_CHUNK_SIZE]

        # Send each chunk prefixed with its chunk ID and frame ID
        chunk_id = struct.pack('B', i)  # Use 1 byte to indicate chunk ID
        sock.sendto(frame_id + chunk_id + chunk, (udp_ip, udp_port))
        print(frame_id, chunk_id)

cap.release()
sock.close()