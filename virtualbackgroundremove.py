import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Correct background image path (Ensure it's a valid file)
background_path = "bg.jpg"  # Change this to your actual image filename
background = cv2.imread(background_path)

# Check if the background image is loaded correctly
if background is None:
    print(f"Error: Background image '{background_path}' not found! Please check the path.")
    exit()

# Start Webcam Feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize background to match frame size
    background_resized = cv2.resize(background, (frame.shape[1], frame.shape[0]))

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get segmentation mask
    results = selfie_segmentation.process(rgb_frame)
    mask = results.segmentation_mask

    # Threshold mask to create binary segmentation
    mask = np.where(mask > 0.5, 1, 0).astype(np.uint8)

    # Apply mask to extract the foreground
    foreground = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Create inverse mask and apply background replacement
    inverse_mask = cv2.bitwise_not(mask)
    background_applied = cv2.bitwise_and(background_resized, background_resized, mask=inverse_mask)
    
    # Merge foreground and background to get the final output
    final_output = cv2.add(foreground, background_applied)

    # Display output
    cv2.imshow("Virtual Background", final_output)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
