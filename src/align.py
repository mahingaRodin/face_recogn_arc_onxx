import os
import time
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np

# Import from our working haar_5pt detector
from .haar_5pt import Haar5ptDetector, align_face_5pt


def draw_text(img, text: str, xy=(10, 30), scale=0.8, thickness=2):
    """Helper: Draw white text with black shadow for readability."""
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)  # Shadow
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 
                scale, (255, 255, 255), thickness, cv2.LINE_AA)  # Text


def test_face_alignment(
    camera_index: int = 0,
    output_size: Tuple[int, int] = (112, 112),
    mirror: bool = True,
):
    """
    Test face alignment with live camera preview.
    
    Args:
        camera_index: Camera device index (0=default)
        output_size: Aligned face size (width, height) - must be (112, 112) for ArcFace
        mirror: Flip camera horizontally (True = selfie mode)
    
    Displays:
        Window 1: Live camera with detection box + 5 keypoints
        Window 2: Aligned face (112×112, upright)
    
    The alignment process:
        1. Detect face (Haar + MediaPipe)
        2. Get 5 keypoints in image coordinates
        3. Compute similarity transform matrix M
        4. Warp image: aligned = cv2.warpAffine(frame, M, (112, 112))
    """
    # Initialize detector
    detector = Haar5ptDetector(
        min_size=(70, 70),      # Minimum face size
        smooth_alpha=0.80,       # Temporal smoothing factor
        debug=True,              # Print debug messages
    )
    
    # Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {camera_index}")
    
    print("✓ Camera opened")
    print("✓ Detector initialized")
    print("\nControls:")
    print("  q - Quit")
    print("  s - Save aligned face\n")
    
    # Setup output directory for saved faces
    save_dir = Path("data/debug_aligned")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Variables
    out_w, out_h = int(output_size[0]), int(output_size[1])
    blank = np.zeros((out_h, out_w, 3), dtype=np.uint8)  # Black placeholder
    last_aligned = blank.copy()  # Keep last good aligned face
    
    # FPS tracking
    fps_time = time.time()
    fps_count = 0
    fps = 0.0
    
    while True:
        # Read frame
        ok, frame = cap.read()
        if not ok:
            break
        
        # Mirror for selfie effect
        if mirror:
            frame = cv2.flip(frame, 1)
        
        # Detect faces
        faces = detector.detect(frame, max_faces=1)
        
        # Prepare visualization
        vis = frame.copy()
        aligned = None
        
        if faces:
            face = faces[0]
            
            # Draw bounding box
            cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), 
                         (0, 255, 0), 2)
            
            # Draw 5 keypoints
            for (x, y) in face.kps.astype(int):
                cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
            
            # Perform alignment
            aligned, M = align_face_5pt(frame, face.kps, out_size=output_size)
            
            # Keep last good aligned face (prevents black screen on brief misses)
            if aligned is not None and aligned.size > 0:
                last_aligned = aligned
            
            draw_text(vis, "Face detected + aligned", (10, 30), 0.7, 2)
        else:
            draw_text(vis, "No face detected", (10, 30), 0.7, 2)
        
        # Calculate FPS
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_time = time.time()
        
        # Display info
        draw_text(vis, f"FPS: {fps:.1f}", (10, 60), 0.7, 2)
        draw_text(vis, f"Align: 5pt -> {out_w}x{out_h}", (10, 90), 0.7, 2)
        
        # Show windows
        cv2.imshow("align - camera (press 'q' to quit)", vis)
        cv2.imshow("align - aligned 112x112", last_aligned)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            print("✓ Alignment test completed")
            break
        
        if key == ord("s"):
            timestamp = int(time.time() * 1000)
            save_path = save_dir / f"{timestamp}.jpg"
            cv2.imwrite(str(save_path), last_aligned)
            print(f"✓ Saved: {save_path}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main entry point."""
    try:
        test_face_alignment(
            camera_index=0,
            output_size=(112, 112),  # ArcFace standard
            mirror=True,             # Selfie mode
        )
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()