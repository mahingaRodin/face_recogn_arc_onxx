from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class FaceKpsBox:
    """
    Face detection result with keypoints.
    
    Attributes:
        x1, y1: Top-left corner of bounding box
        x2, y2: Bottom-right corner of bounding box
        score: Detection confidence (0-1)
        kps: 5 keypoints as (5, 2) array [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # Shape: (5, 2), dtype: float32


# ============================================================================
# Face Alignment Functions
# ============================================================================

def _estimate_norm_5pt(
    kps_5x2: np.ndarray,
    out_size: Tuple[int, int] = (112, 112)
) -> np.ndarray:
    """
    Compute 2×3 affine transformation matrix for face alignment.
    
    This function calculates how to warp a face with arbitrary keypoints
    to match the ArcFace standard template positions.
    
    Args:
        kps_5x2: 5 keypoints as (5, 2) array in [x, y] format
                 Order: [left_eye, right_eye, nose, left_mouth, right_mouth]
        out_size: Output image size (width, height)
    
    Returns:
        M: 2×3 affine transformation matrix for cv2.warpAffine
    
    Transform Type:
        Similarity transform (rotation + scale + translation)
        - Preserves angles and proportions
        - Does NOT handle perspective distortion
    
    Template:
        ArcFace standard 112×112 template (from InsightFace)
        These are the "ideal" positions where facial features should be
    """
    kps = kps_5x2.astype(np.float32)
    
    # ArcFace 112×112 standard template positions
    # These positions were empirically determined to work well for ArcFace model
    dst = np.array([
        [38.2946, 51.6963],  # Left eye center
        [73.5318, 51.5014],  # Right eye center
        [56.0252, 71.7366],  # Nose tip
        [41.5493, 92.3655],  # Left mouth corner
        [70.7299, 92.2041],  # Right mouth corner
    ], dtype=np.float32)
    
    # Scale template if output size is not 112×112
    out_w, out_h = int(out_size[0]), int(out_size[1])
    if (out_w, out_h) != (112, 112):
        scale_x = out_w / 112.0
        scale_y = out_h / 112.0
        dst = dst * np.array([scale_x, scale_y], dtype=np.float32)
    
    # Estimate similarity transform (rotation + scale + translation)
    # LMEDS is robust to outliers
    M, _ = cv2.estimateAffinePartial2D(kps, dst, method=cv2.LMEDS)
    
    # Fallback: If estimation fails, use first 3 points only
    if M is None:
        M = cv2.getAffineTransform(
            np.array([kps[0], kps[1], kps[2]], dtype=np.float32),  # Eyes + nose
            np.array([dst[0], dst[1], dst[2]], dtype=np.float32),
        )
    
    return M.astype(np.float32)


def align_face_5pt(
    frame_bgr: np.ndarray,
    kps_5x2: np.ndarray,
    out_size: Tuple[int, int] = (112, 112)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align face to canonical pose using 5 keypoints.
    
    Args:
        frame_bgr: Input image (BGR format)
        kps_5x2: 5 keypoints as (5, 2) array
        out_size: Output size (width, height)
    
    Returns:
        aligned: Aligned face image (out_size)
        M: Transformation matrix used (2×3)
    
    Example:
        aligned, M = align_face_5pt(frame, keypoints, out_size=(112, 112))
    """
    # Compute transformation matrix
    M = _estimate_norm_5pt(kps_5x2, out_size=out_size)
    
    # Warp image
    out_w, out_h = int(out_size[0]), int(out_size[1])
    aligned = cv2.warpAffine(
        frame_bgr,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,        # Bilinear interpolation
        borderMode=cv2.BORDER_CONSTANT, # Fill borders with constant color
        borderValue=(0, 0, 0),          # Black border
    )
    
    return aligned, M


# ============================================================================
# Helper Functions
# ============================================================================

def _clip_box_xyxy(box: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    Clip bounding box to image boundaries.
    
    Args:
        box: [x1, y1, x2, y2] bounding box
        W, H: Image width and height
    
    Returns:
        Clipped box
    """
    bb = box.astype(np.float32).copy()
    bb[0] = np.clip(bb[0], 0, W - 1)  # x1
    bb[1] = np.clip(bb[1], 0, H - 1)  # y1
    bb[2] = np.clip(bb[2], 0, W - 1)  # x2
    bb[3] = np.clip(bb[3], 0, H - 1)  # y2
    return bb


def _bbox_from_5pt(
    kps: np.ndarray,
    pad_x: float = 0.55,
    pad_y_top: float = 0.85,
    pad_y_bot: float = 1.15
) -> np.ndarray:
    """
    Build face bounding box from 5 keypoints.
    
    Strategy: Find min/max of keypoints, then add asymmetric padding
    to include forehead and chin.
    
    Args:
        kps: 5 keypoints (5, 2)
        pad_x: Horizontal padding multiplier
        pad_y_top: Top padding multiplier (forehead)
        pad_y_bot: Bottom padding multiplier (chin)
    
    Returns:
        [x1, y1, x2, y2] bounding box
    
    Why Asymmetric Padding?
        - More padding on top (forehead)
        - More padding on bottom (chin)
        - This creates a "face-like" box that looks centered
    """
    k = kps.astype(np.float32)
    
    x_min = float(np.min(k[:, 0]))
    x_max = float(np.max(k[:, 0]))
    y_min = float(np.min(k[:, 1]))
    y_max = float(np.max(k[:, 1]))
    
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    
    # Add padding
    x1 = x_min - pad_x * w
    x2 = x_max + pad_x * w
    y1 = y_min - pad_y_top * h    # Extra padding for forehead
    y2 = y_max + pad_y_bot * h    # Extra padding for chin
    
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _ema(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    """
    Exponential Moving Average (temporal smoothing).
    
    Formula: smoothed = alpha * previous + (1 - alpha) * current
    
    Args:
        prev: Previous value (None on first frame)
        cur: Current value
        alpha: Smoothing factor [0, 1]
               - 0.0 = no smoothing (instant)
               - 0.9 = heavy smoothing (slow)
               - 0.8 = balanced (recommended)
    
    Returns:
        Smoothed value
    """
    if prev is None:
        return cur.astype(np.float32)
    return (alpha * prev + (1.0 - alpha) * cur).astype(np.float32)


def _kps_span_ok(kps: np.ndarray, min_eye_dist: float = 12.0) -> bool:
    """
    Sanity check on keypoint geometry.
    
    Checks:
        1. Eye distance is reasonable (not collapsed)
        2. Mouth is below nose (correct face orientation)
    
    Args:
        kps: 5 keypoints (5, 2)
        min_eye_dist: Minimum eye distance in pixels
    
    Returns:
        True if geometry is valid
    """
    k = kps.astype(np.float32)
    left_eye, right_eye, nose, left_mouth, right_mouth = k
    
    # Check 1: Eyes not too close
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    if eye_dist < min_eye_dist:
        return False
    
    # Check 2: Mouth below nose (usually true for frontal faces)
    if not (left_mouth[1] > nose[1] and right_mouth[1] > nose[1]):
        return False
    
    return True


# ============================================================================
# Main Detector Class
# ============================================================================

class Haar5ptDetector:
    """
    Combined Haar Cascade + MediaPipe FaceMesh detector.
    
    Pipeline:
        1. Haar detects face box (fast)
        2. MediaPipe extracts 468 landmarks (accurate)
        3. Extract 5 keypoints from landmarks
        4. Validate geometry
        5. Rebuild bounding box from keypoints
        6. Apply temporal smoothing
    
    Attributes:
        face_cascade: Haar Cascade classifier
        mp_face_mesh: MediaPipe FaceMesh model
        min_size: Minimum face size for detection
        smooth_alpha: EMA smoothing factor [0, 1]
        debug: Print debug messages
    
    Example:
        detector = Haar5ptDetector(min_size=(70, 70), smooth_alpha=0.80)
        faces = detector.detect(frame, max_faces=1)
        
        if faces:
            face = faces[0]
            print(f"Box: ({face.x1}, {face.y1}) to ({face.x2}, {face.y2})")
            print(f"Keypoints: {face.kps}")
    """
    
    def __init__(
        self,
        haar_xml: Optional[str] = None,
        min_size: Tuple[int, int] = (60, 60),
        smooth_alpha: float = 0.80,
        debug: bool = True,
    ):
        """
        Initialize detector.
        
        Args:
            haar_xml: Path to Haar Cascade XML (None = use default)
            min_size: Minimum face size (width, height)
            smooth_alpha: Temporal smoothing factor [0, 1]
            debug: Print debug messages
        """
        self.debug = bool(debug)
        self.min_size = tuple(map(int, min_size))
        self.smooth_alpha = float(smooth_alpha)
        
        # Load Haar Cascade
        if haar_xml is None:
            haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        
        self.face_cascade = cv2.CascadeClassifier(haar_xml)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {haar_xml}")
        
        # Initialize MediaPipe FaceMesh
        if mp is None:
            raise RuntimeError(
                f"MediaPipe import failed: {_MP_IMPORT_ERROR}\n"
                f"Install: pip install mediapipe"
            )
        
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # MediaPipe landmark indices for 5 keypoints
        self.IDX_LEFT_EYE = 33
        self.IDX_RIGHT_EYE = 263
        self.IDX_NOSE_TIP = 1
        self.IDX_MOUTH_LEFT = 61
        self.IDX_MOUTH_RIGHT = 291
        
        # Temporal smoothing state
        self._prev_box: Optional[np.ndarray] = None
        self._prev_kps: Optional[np.ndarray] = None
        
        if self.debug:
            print("[Haar5ptDetector] Initialized")
            print(f"  Haar min_size: {self.min_size}")
            print(f"  Smoothing alpha: {self.smooth_alpha}")
    
    def _haar_faces(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect faces using Haar Cascade.
        
        Returns:
            Array of faces in (x, y, w, h) format
        """
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=self.min_size,
        )
        
        if faces is None or len(faces) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        
        return faces.astype(np.int32)
    
    def _facemesh_5pt(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 5 keypoints using MediaPipe FaceMesh.
        
        Args:
            frame_bgr: Input image (BGR)
        
        Returns:
            5 keypoints as (5, 2) array, or None if no face detected
        """
        H, W = frame_bgr.shape[:2]
        
        # MediaPipe requires RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.mp_face_mesh.process(rgb)
        
        if not result.multi_face_landmarks:
            return None
        
        # Get first face's landmarks
        landmarks = result.multi_face_landmarks[0].landmark
        
        # Extract 5 keypoints
        indices = [
            self.IDX_LEFT_EYE,
            self.IDX_RIGHT_EYE,
            self.IDX_NOSE_TIP,
            self.IDX_MOUTH_LEFT,
            self.IDX_MOUTH_RIGHT,
        ]
        
        pts = []
        for idx in indices:
            lm = landmarks[idx]
            # Convert normalized [0, 1] coords to pixel coords
            pts.append([lm.x * W, lm.y * H])
        
        kps = np.array(pts, dtype=np.float32)  # Shape: (5, 2)
        
        # Enforce left/right ordering
        if kps[0, 0] > kps[1, 0]:  # Left eye should be to the left
            kps[[0, 1]] = kps[[1, 0]]
        
        if kps[3, 0] > kps[4, 0]:  # Left mouth should be to the left
            kps[[3, 4]] = kps[[4, 3]]
        
        return kps
    
    def detect(self, frame_bgr: np.ndarray, max_faces: int = 1) -> List[FaceKpsBox]:
        """
        Detect faces with keypoints.
        
        Args:
            frame_bgr: Input image (BGR format)
            max_faces: Maximum number of faces to return
        
        Returns:
            List of FaceKpsBox objects (sorted by area, largest first)
        
        Pipeline:
            1. Haar detects faces
            2. Pick largest face
            3. MediaPipe extracts 5 keypoints
            4. Validate geometry
            5. Rebuild bounding box from keypoints
            6. Apply temporal smoothing
        """
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Haar detection
        faces = self._haar_faces(gray)
        if faces.shape[0] == 0:
            return []
        
        # Step 2: Pick largest Haar face
        areas = faces[:, 2] * faces[:, 3]
        i = int(np.argmax(areas))
        x, y, w, h = faces[i].tolist()
        
        # Step 3: MediaPipe 5-point extraction
        kps = self._facemesh_5pt(frame_bgr)
        if kps is None:
            if self.debug:
                print("[Haar5ptDetector] Haar found face but MediaPipe returned None")
            return []
        
        # Step 4: Validate keypoint geometry
        if not _kps_span_ok(kps, min_eye_dist=max(10.0, 0.18 * w)):
            if self.debug:
                print("[Haar5ptDetector] Keypoint geometry validation failed")
            return []
        
        # Step 5: Rebuild bounding box from keypoints (more accurate + centered)
        box = _bbox_from_5pt(kps, pad_x=0.55, pad_y_top=0.85, pad_y_bot=1.15)
        box = _clip_box_xyxy(box, W, H)
        
        # Step 6: Temporal smoothing
        box_smooth = _ema(self._prev_box, box, self.smooth_alpha)
        kps_smooth = _ema(self._prev_kps, kps, self.smooth_alpha)
        
        self._prev_box = box_smooth.copy()
        self._prev_kps = kps_smooth.copy()
        
        # Build result
        x1, y1, x2, y2 = box_smooth.tolist()
        score = 1.0  # Haar doesn't provide confidence, use placeholder
        
        return [
            FaceKpsBox(
                x1=int(round(x1)),
                y1=int(round(y1)),
                x2=int(round(x2)),
                y2=int(round(y2)),
                score=float(score),
                kps=kps_smooth.astype(np.float32),
            )
        ][:max_faces]


# ============================================================================
# Demo / Test
# ============================================================================

def main():
    """Test the detector with live camera."""
    cap = cv2.VideoCapture(0)
    detector = Haar5ptDetector(
        min_size=(70, 70),
        smooth_alpha=0.80,
        debug=True,
    )
    
    print("Haar + 5pt detector test. Press 'q' to quit.")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        faces = detector.detect(frame, max_faces=1)
        vis = frame.copy()
        
        if faces:
            f = faces[0]
            # Draw box
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
            
            # Draw keypoints
            for (x, y) in f.kps.astype(int):
                cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
            
            cv2.putText(vis, "OK", (f.x1, max(0, f.y1 - 8)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "no face", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        cv2.imshow("haar_5pt test", vis)
        
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()