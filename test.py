import cv2
import mediapipe as mp
import time
import gc
import os

# Ensure the video file exists
video_path = "demo.mp4"
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Error: The video file '{video_path}' was not found.")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, model_complexity=0)
mp_drawing = mp.solutions.drawing_utils

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    raise RuntimeError("Error: Could not open video file. Check if the file path is correct.")

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 20.0  # Default to 20 FPS if unknown

# Define the video codec & output file (use H.264 instead of MP4V)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # More stable codec
out = cv2.VideoWriter('output_with_hand_tracking.avi', fourcc, fps, (frame_width, frame_height))
# Process video frame by frame
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or read error. Exiting...")
        break

    # Resize frame to reduce memory usage (optional)
    frame = cv2.resize(frame, (640, 480))

    # Convert the frame to RGB (for MediaPipe)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(image)

    # Draw hand center tracking
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box coordinates
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])

            # Compute the center of the bounding box
            h, w, _ = frame.shape
            center_x = int((x_min + x_max) / 2 * w)
            center_y = int((y_min + y_max) / 2 * h)

            # Draw the center point
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
            cv2.putText(frame, f'Center: ({center_x}, {center_y})', (center_x + 10, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Write the processed frame to output video
    out.write(frame)

    # Save processed frames instead of displaying them
    cv2.imwrite(f'frames/frame_{frame_count:04d}.jpg', frame)

    # Pause for 30ms per frame to prevent CPU overload
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Free up memory every 50 frames
    frame_count += 1
    if frame_count % 50 == 0:
        gc.collect()

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved as 'output_with_hand_tracking.mp4'.")