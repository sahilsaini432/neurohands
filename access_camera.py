import cv2
import numpy as np

# Open the default camera (usually the first one connected to the system)
cap = cv2.VideoCapture(0)

# define skin color for capture
lower_skin = np.array([0,20,70], dtype=np.uint8)
upper_skin = np.array([20,255,255], dtype=np.uint8)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Capture frames in a loop
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("failed to capture video")
        break
    
    # If frame capture was successful
    # convert the frame to hsv color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # create a mask for skin color using the defined boundaries
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # apply a series of dilations and erosions to remove noise
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=4)
    mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=2)

    # blur the mask to help remove noise
    mask = cv2.GaussianBlur(mask, (5,5), 100)

    # find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # if any contours are found, assume the largest ones in the hand
    if len(contours) > 0:
        # find the largest contour
        max_contour = max(contours, key=cv2.contourArea)
        # get the bounding box for the largest contour (i.e., the hand)
        x,y,w,h = cv2.boundingRect(max_contour)

        # draw a rectangle (Square) around the detected hand
        color = (0, 255,0)
        thickness = 2
        cv2.rectangle(frame, (x,y), (x+w, y+h), color=color, thickness=thickness)

    # Display the resulting frame in a window named 'Camera'
    cv2.imshow('Camera', frame)
    
    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
