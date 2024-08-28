import cv2

# Open the default camera (usually the first one connected to the system)

cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Capture frames in a loop
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame capture was successful
    if ret:
        # Display the resulting frame in a window named 'Camera'
        cv2.imshow('Camera', frame)
        
        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Error: Could not read frame.")
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
