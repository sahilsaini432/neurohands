from ast import parse
import mediapipe as mp
import cv2

# open camera and start capturing frames
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if not cap.isOpened():
    print("Error: could not access the camera.")
    exit()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
print(list(mp_hands.HandLandmark))
with mp_hands.Hands(static_image_mode=False, max_num_hands=2,min_detection_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("failed to capture video")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

        # Display frame
        cv2.imshow("Hand Detection", frame)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()