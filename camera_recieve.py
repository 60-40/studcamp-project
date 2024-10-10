import cv2
import socket
import numpy as np
import struct

# Set up the UDP connection
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_ip = "0.0.0.0"  # Listen on all available interfaces
udp_port = 12345  # Same port number as the sender
sock.bind((udp_ip, udp_port))

# Maximum packet size (UDP MTU is 65507 bytes for IPv4, minus header overhead)
MAX_PACKET_SIZE = 65507
HEADER_SIZE = 2  # 1 byte for frame_id and 1 byte for chunk_id
MAX_CHUNK_SIZE = MAX_PACKET_SIZE - HEADER_SIZE  # Ensure data and headers fit in packet

# Dictionary to store chunks of each frame
frame_chunks = {}

while True:
    # Receive data
    data, addr = sock.recvfrom(MAX_PACKET_SIZE)

    # Extract frame ID and chunk ID
    frame_id = struct.unpack('B', data[0:1])[0]
    chunk_id = struct.unpack('B', data[1:2])[0]
    chunk_data = data[2:]

    # If this is the first chunk of the frame, initialize the list
    if frame_id not in frame_chunks:
        frame_chunks[frame_id] = [None] * (chunk_id + 1)

    # Store the chunk data
    frame_chunks[frame_id][chunk_id] = chunk_data

    # Check if all chunks of the frame are received
    if all(chunk is not None for chunk in frame_chunks[frame_id]):
        # Combine all chunks into a single byte array
        full_data = b''.join(frame_chunks[frame_id])

        # Decode the JPEG image
        frame = cv2.imdecode(np.frombuffer(full_data, dtype=np.uint8),
                             cv2.IMREAD_COLOR)

        # Display the frame
        cv2.imshow('Received Frame', frame)

        # Clear the chunks for the next frame
        del frame_chunks[frame_id]

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
sock.close()
