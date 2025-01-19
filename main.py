import cv2

# Load Haar Cascade for face detection
cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

def detect_faces(frame):
    # Convert frame to grayscale (Haar Cascade works with grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return frame

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)  # 0 for the default camera

    print("Press 'q' to exit...")
    while True:
        ret, frame = cap.read()  # Capture a single frame
        if not ret:
            break

        # Detect faces in the frame
        frame = detect_faces(frame)

        # Display the frame
        cv2.imshow("Face Detection", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

def detect_faces_in_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    frame = detect_faces(image)

    # Display the image with detected faces
    cv2.imshow("Face Detection - Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_detected_faces(image_path, output_path="output.jpg"):
    image = cv2.imread(image_path)
    frame = detect_faces(image)
    cv2.imwrite(output_path, frame)
    print(f"Saved detected faces to {output_path}")



if __name__ == "__main__":
    main()
