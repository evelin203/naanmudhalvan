import mediapipe as mp
import cv2
import math

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

# Example alphabet mapping (replace with your actual mapping)
# This mapping now uses distances between landmarks, making it more robust
alphabet_mapping = {
    "A": [(4, 8, 0.15), (4, 12, 0.15)],  # Thumb next to index finger (distance)
    "B": [(5, 8, 0.1), (5, 12, 0.1), (5, 16, 0.1), (5, 20, 0.1)],  # All fingers extended (distance from palm)
    "C": [(8, 12, 0.15), (12, 16, 0.15), (16, 20, 0.15)],  # Fingers curved
    "D": [(4, 8, 0.35)],  # Index finger extended, others curled
    "E": [(4, 8, 0.15), (8, 12, 0.1), (12, 16, 0.1), (16, 20, 0.1)]  ,# All fingertips curled close to the palm
    "F": [(4, 8, 0.05), (8, 12, 0.4), (12, 16, 0.4), (16, 20, 0.4)] , # Thumb touching index, others extended
    "G": [(8, 5, 0.4), (4, 5, 0.25), (12, 5, 0.1), (16, 5, 0.1), (20, 5, 0.1)], # Index finger points up, thumb extended to the side
    "H": [(8, 5, 0.4), (12, 5, 0.4), (16, 5, 0.1), (20, 5, 0.1), (4, 5, 0.1)],   # Index and middle finger extended, other fingers curled
    "I": [(20, 5, 0.4), (4, 5, 0.1), (8, 5, 0.1), (12, 5, 0.1), (16, 5, 0.1)],  # Little finger extended, other fingers curled
    "J": [(8, 5, 0.4), (4, 5, 0.4), (12, 5, 0.1), (16, 5, 0.1), (20, 5, 0.1)],   # Little finger curves in a J shape
    "K": [(8, 5, 0.4), (12, 5, 0.4), (4, 5, 0.4), (16, 5, 0.1), (20, 5, 0.1)], # Index and middle finger extended, thumb pointing up
    "L": [(20, 0, 0.8)],  # Index finger and thumb extended at right angles
    "M": [(4, 0, 0.2), (8, 0, 0.2), (12, 0, 0.2), (16, 0, 0.2), (20, 0, 0.2)],  # Fingers curled, thumb tucked in
    "N": [(8, 12, 0.1), (12, 16, 0.1), (4, 0, 0.2), (20, 5, 0.1), (8, 5, 0.3)],  # Index and middle finger crossed
    "O": [(4, 8, 0.3), (8, 12, 0.3), (12, 16, 0.3), (16, 20, 0.3), (4, 20, 0.3)],  # Fingertips forming an enclosed shape
    "P": [(8, 5, 0.4), (12, 5, 0.4), (4, 0, 0.2), (16, 5, 0.1), (20, 5, 0.1)],   # Index and middle finger extended, other fingers curled and touching thumb
    "Q": [(8, 12, 0.2), (12, 16, 0.2), (16, 20, 0.2), (4, 5, 0.1)],  # Hand curved, index finger points down
    "R": [(8, 12, 0.1), (12, 16, 0.1), (4, 0, 0.2), (20, 5, 0.1), (16, 5, 0.3)],   # Index and middle finger crossed slightly
    "S": [(4, 0, 0.2), (8, 0, 0.2), (12, 0, 0.2), (16, 0, 0.2), (20, 0, 0.2)],  # Fist with thumb across fingers
    "T": [(4, 8, -0.05), (8, 12, 0.3), (4, 5, 0.2), (16, 5, 0.1), (20, 5, 0.1)],   # Thumb between index and middle finger
    "U": [(8, 5, 0.4), (12, 5, 0.4), (16, 5, 0.4), (4, 5, 0.1), (20, 5, 0.1)],   # Index and middle finger extended and together
    "V": [(8, 12, 0.3), (12, 16, 0.1), (8, 5, 0.4), (12, 5, 0.4), (16, 5, 0.1)],   # Index and middle finger extended and apart
    "W": [(8, 12, 0.3), (12, 16, 0.3), (16, 20, 0.3), (8, 5, 0.4), (12, 5, 0.4), (16, 5, 0.4)], # Index, middle, and ring finger extended and apart
    "X": [(8, 0, 0.25), (12, 8, 0.1), (12, 5, 0.3), (4, 5, 0.1), (16, 5, 0.1), (20, 5, 0.1)],  # Index finger curled into a hook
    "Y": [(4, 20, 0.3), (8, 5, 0.1), (12, 5, 0.1), (16, 5, 0.1), (20, 5, 0.4)],   # Thumb and little finger extended
    "Z": [(4, 0, 0.1), (8, 0, 0.1), (12, 0, 0.1), (16, 0, 0.1), (20, 0, 0.1)] # Trace Z with index finger
}

def distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)

def recognize_sign(hand_landmarks):
    if not hand_landmarks:
        return None

    for letter, conditions in alphabet_mapping.items():
        match = True
        for condition in conditions:
            landmark_index1, landmark_index2, threshold = condition

            if not (0 <= landmark_index1 <= 20 and 0 <= landmark_index2 <= 20):
                print(f"Invalid landmark index: {landmark_index1} or {landmark_index2} for letter {letter}")
                match = False
                break

            try:
                dist = distance(hand_landmarks.landmark[landmark_index1], hand_landmarks.landmark[landmark_index2])
                if dist > threshold:
                    match = False
                    break
            except IndexError as e:
                print(f"IndexError: {e} while processing letter {letter}: {e}")
                return None

        if match:
            return letter

    return None

# Example usage (replace with your actual image/video processing loop)
def process_frame(image):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Recognize the sign
            recognized_sign = recognize_sign(hand_landmarks)

            if recognized_sign:
                print(f"Recognized Sign: {recognized_sign}")
                cv2.putText(image, recognized_sign, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                print("Sign not recognized")
                cv2.putText(image, "Sign not recognized", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            return recognized_sign  # Return the first recognized sign

    return None  # No hands detected

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        recognized_sign = process_frame(frame)

        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()