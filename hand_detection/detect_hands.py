from ast import parse
import mediapipe as mp
import cv2
from dataclasses import asdict

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
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)
        
        processed_hands = []
        
        if result.multi_handedness:
            for handedness in result.multi_handedness:
                processed_hands.append(handedness.classification[0].label)
        
        if result.multi_hand_landmarks:
            index = 0
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                
                # Display the hand label on the frame
                h, w, _ = frame.shape
                x, y = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, processed_hands[index], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                index = index + 1

        # Display frame
        cv2.imshow("Hand Detection", frame)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()