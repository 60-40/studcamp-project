import cv2

# Replace this with the Raspberry Pi's IP address
raspberry_pi_ip = "192.168.2.165"  # Change this to your Raspberry Pi's IP
video_url = f"http://{raspberry_pi_ip}:1234/video_feed"

# Start capturing video
cap = cv2.VideoCapture(video_url)

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to capture video")
        break

    # Display the resulting frame
    cv2.imshow("Raspberry Pi Video Stream", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
