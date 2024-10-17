import cv2
from ultralytics import YOLO

# Load the trained model
model_path = "best.pt"
model = YOLO(model_path)

# Function to process a video
def process_video(input_video, output_video):
    # Open video capture
    cap = cv2.VideoCapture(input_video)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Open video writer to save the output
    out = cv2.VideoWriter(output_video, codec, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference on each frame
        results = model(frame)
        
        # Render the detections
        processed_frame = results[0].plot()  # Rendered frame with bounding boxes
        
        # Save the processed frame
        out.write(processed_frame)
        
        # Optionally display the frame
        cv2.imshow("YOLO Inference", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Input and output video paths
input_video_path = 'video2.mp4'  # Your input video
output_video_path = 'output_video.mp4'  # Path to save processed video

# Run video processing
process_video(input_video_path, output_video_path)
