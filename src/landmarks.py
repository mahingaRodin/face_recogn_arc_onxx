import cv2
import numpy as np
import mediapipe as mp


# MediaPipe FaceMesh landmark indices for our 5 keypoints
IDX_LEFT_EYE = 33      # Left eye center (pupil area)
IDX_RIGHT_EYE = 263    # Right eye center (pupil area)
IDX_NOSE_TIP = 1       # Nose tip (most forward point)
IDX_MOUTH_LEFT = 61    # Left corner of mouth
IDX_MOUTH_RIGHT = 291  # Right corner of mouth


def test_landmark_detection():
    """
    Test 5-point landmark extraction using MediaPipe FaceMesh.
    
    Pipeline:
        1. Haar detects face bounding box (fast, CPU-friendly)
        2. MediaPipe FaceMesh detects 468 landmarks (more accurate)
        3. Extract only 5 specific landmarks we need
        4. Enforce left/right ordering (important for alignment)
    """
    # Initialize Haar Cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade")
    
    # Initialize MediaPipe FaceMesh
    # static_image_mode=False → Optimized for video (uses tracking)
    # max_num_faces=1 → Only detect one face (faster)
    # refine_landmarks=True → Better accuracy around eyes/mouth
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    print("✓ Haar Cascade loaded")
    print("✓ MediaPipe FaceMesh initialized")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened")
    
    print("✓ Camera opened")
    print("Controls: Press 'q' to quit\n")
    print("Look at camera to see 5 landmark points...\n")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        H, W = frame.shape[:2]  # Frame dimensions
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Detect faces with Haar (fast pre-filter)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        
        # Draw all Haar face boxes (green rectangles)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Step 2: Run MediaPipe FaceMesh on full frame
        # (In practice, we could crop to Haar box for speed, but this is simpler)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        
        # Step 3: Extract 5 keypoints if face detected
        if result.multi_face_landmarks:
            # Get first (and only) face's landmarks
            landmarks = result.multi_face_landmarks[0].landmark
            
            # Extract our 5 keypoints
            indices = [IDX_LEFT_EYE, IDX_RIGHT_EYE, IDX_NOSE_TIP, 
                      IDX_MOUTH_LEFT, IDX_MOUTH_RIGHT]
            
            keypoints = []
            for idx in indices:
                lm = landmarks[idx]
                # MediaPipe returns normalized coords [0,1], convert to pixels
                x_px = lm.x * W
                y_px = lm.y * H
                keypoints.append([x_px, y_px])
            
            kps = np.array(keypoints, dtype=np.float32)  # Shape: (5, 2)
            
            # Step 4: Enforce left/right ordering (important for alignment)
            # Sometimes MediaPipe swaps left/right, we need consistent ordering
            if kps[0, 0] > kps[1, 0]:  # If left eye is to the right of right eye
                kps[[0, 1]] = kps[[1, 0]]  # Swap them
            
            if kps[3, 0] > kps[4, 0]:  # If left mouth is to the right of right mouth
                kps[[3, 4]] = kps[[4, 3]]  # Swap them
            
            # Draw the 5 keypoints
            for (px, py) in kps.astype(int):
                cv2.circle(
                    frame,
                    (int(px), int(py)),
                    4,                    # Radius
                    (0, 255, 0),         # Green
                    -1                    # Filled circle
                )
            
            # Display confirmation label
            cv2.putText(
                frame,
                "5pt landmarks detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        # Show result
        cv2.imshow("5-Point Landmarks Test - Press 'q' to quit", frame)
        
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            print("✓ Landmark test completed")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main entry point."""
    try:
        test_landmark_detection()
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()