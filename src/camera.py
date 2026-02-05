import cv2
import time


def test_camera(camera_index=0):
    """
    Test camera access and display live feed.
    
    Args:
        camera_index (int): Camera device index (0=default, 1=external, etc.)
    
    Raises:
        RuntimeError: If camera cannot be opened
    """
    # Attempt to open the camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise RuntimeError(
            f"Camera not opened. Try changing index (0/1/2).\n"
            f"On macOS: Check System Settings → Privacy & Security → Camera\n"
            f"On Windows: Check Settings → Privacy → Camera"
        )
    
    print("✓ Camera opened successfully")
    print("Controls: Press 'q' to quit")
    print("\nDisplaying live feed...")
    
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    fps = 0.0
    
    while True:
        # Read frame from camera
        ok, frame = cap.read()
        
        if not ok:
            print("⚠ Failed to read frame. Camera may have disconnected.")
            break
        
        # Calculate FPS every second
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()
        
        # Display FPS on frame
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        # Show the frame
        cv2.imshow("Camera Test - Press 'q' to quit", frame)
        
        # Check for quit key
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            print("\n✓ Camera test completed successfully")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main entry point for camera validation."""
    try:
        test_camera(camera_index=0)
    except RuntimeError as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check camera permissions in system settings")
        print("  2. Close other apps using the camera")
        print("  3. Try: test_camera(camera_index=1) for external camera")


if __name__ == "__main__":
    main()