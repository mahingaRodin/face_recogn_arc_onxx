import cv2


def test_face_detection():
    """
    Test Haar Cascade face detection on live video.
    
    The Haar Cascade is a pre-trained classifier that detects frontal faces
    using rectangular features. It's fast enough for real-time CPU processing.
    """
    # Load Haar Cascade XML file (comes with OpenCV)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Verify cascade loaded
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from: {cascade_path}")
    
    print("✓ Haar Cascade loaded successfully")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Try camera index 0/1/2.")
    
    print("✓ Camera opened")
    print("Controls: Press 'q' to quit")
    print("\nDetecting faces...\n")
    
    while True:
        # Read frame
        ok, frame = cap.read()
        if not ok:
            break
        
        # Convert to grayscale (Haar works on grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        # Returns array of rectangles: [[x, y, w, h], ...]
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,     # How much to reduce image size at each scale
            minNeighbors=5,      # How many neighbors each candidate needs
            minSize=(60, 60),    # Minimum face size (smaller faces ignored)
        )
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame,
                (x, y),              # Top-left corner
                (x + w, y + h),      # Bottom-right corner
                (0, 255, 0),         # Green color
                2                     # Line thickness
            )
            
            # Display face count
            cv2.putText(
                frame,
                f"Faces: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        
        # Show result
        cv2.imshow("Face Detection Test - Press 'q' to quit", frame)
        
        # Check for quit
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            print("✓ Detection test completed")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main entry point."""
    try:
        test_face_detection()
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()