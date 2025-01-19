import cv2

# Load Haar Cascade for face detection
cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

def detect_faces(frame):
    """
    Detect faces in the given frame and draw bounding boxes around them.
    """
    # Convert frame to grayscale (Haar Cascade works with grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes and labels around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# def detect_faces_in_image(image_path, output_path="output.jpg"):
    """
    Detect faces in an image, display it, and save the output image with bounding boxes.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image. Check the file path.")
        return
    
    # Detect faces in the image
    frame = detect_faces(image)

    # Display the image with detected faces
    cv2.imshow("Face Detection - Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the output image
    cv2.imwrite(output_path, frame)
    print(f"Saved detected faces to {output_path}")

def detect_faces_from_webcam():
    """
    Use the webcam to detect faces in real-time.
    """
    # Open webcam
    cap = cv2.VideoCapture(0)  # 0 for the default camera

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to exit...")
    while True:
        ret, frame = cap.read()  # Capture a single frame
        if not ret:
            print("Error: Unable to read frame from webcam.")
            break

        # Detect faces in the frame
        frame = detect_faces(frame)

        # Display the frame
        cv2.imshow("Face Detection - Webcam", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Main function to select between webcam or image-based face detection.
    """
    print("Choose an option:")
    print("1. Detect faces from webcam")
    # print("2. Detect faces in an image")
    print("2. Exit")

    choice = input("Enter your choice (1/2): ").strip()
    if choice == "1":
        detect_faces_from_webcam()
    # elif choice == "2":
    #     image_path = input("Enter the path to the image: ").strip()
    #     output_path = input("Enter the path to save the output image (default: output.jpg): ").strip()
    #     if not output_path:
    #         output_path = "output.jpg"
    #     detect_faces_in_image(image_path, output_path)
    elif choice == "2":
        print("Exiting the program.")
    else:
        print("Invalid choice! Please enter 1 or 2.")

if __name__ == "__main__":
    main()
