import cv2

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # Specify the correct file path

# Create a video capture object to access the webcam
cam = cv2.VideoCapture(0)  # Correct usage of VideoCapture

while True:
    _, img = cam.read()  # Read a frame from the camera

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale

    faces = face_cascade.detectMultiScale(gray, 1.3, 4)  # Detect faces in the grayscale frame

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)  # Draw rectangles around detected faces

    cv2.imshow("Face Detection", img)  # Display the image with detected faces

    key = cv2.waitKey(10)
    if key == 27:  # Exit on ESC key press
        break

cam.release()
cv2.destroyAllWindows()
