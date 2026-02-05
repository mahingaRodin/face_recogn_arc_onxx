from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import time

import cv2
import numpy as np
import onnxruntime as ort

from .haar_5pt import Haar5ptDetector, align_face_5pt


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class EmbeddingResult:
    """
    Result from embedding extraction.
    
    Attributes:
        embedding: L2-normalized embedding vector (512,)
        norm_before: Vector norm before normalization (typically 15-30)
        dim: Embedding dimensionality (512 for ArcFace)
    """
    embedding: np.ndarray  # Shape: (512,), dtype: float32, L2-normalized
    norm_before: float     # Original norm before normalization
    dim: int               # Embedding dimension (512)


# ============================================================================
# ArcFace ONNX Embedder
# ============================================================================

class ArcFaceEmbedderONNX:
    """
    ArcFace face embedding extractor using ONNX Runtime.
    
    Model: ArcFace (ResNet-50 backbone, trained on WebFace600K)
    Input: 112×112 BGR image (aligned face)
    Output: 512-dimensional L2-normalized embedding
    
    Pipeline:
        1. BGR → RGB conversion
        2. Normalize to [-1, 1]: (pixel - 127.5) / 128.0
        3. HWC → CHW format: (H, W, C) → (C, H, W)
        4. Add batch dimension: (C, H, W) → (1, C, H, W)
        5. Run ONNX model
        6. L2 normalize output
    
    Why This Preprocessing?
        - ArcFace was trained with these exact steps
        - Different preprocessing = poor results
        - Mean 127.5, std 128.0 centers data around 0
    
    Example:
        embedder = ArcFaceEmbedderONNX("models/embedder_arcface.onnx")
        
        # Embed one face
        aligned = cv2.imread("aligned_face.jpg")  # 112×112 BGR
        result = embedder.embed(aligned)
        
        print(result.embedding.shape)     # (512,)
        print(result.dim)                 # 512
        print(result.norm_before)         # ~21.5 (before normalization)
        print(np.linalg.norm(result.embedding))  # 1.0 (after normalization)
    """
    
    def __init__(
        self,
        model_path: str = "models/embedder_arcface.onnx",
        input_size: Tuple[int, int] = (112, 112),
        debug: bool = False,
    ):
        """
        Initialize ArcFace embedder.
        
        Args:
            model_path: Path to ArcFace ONNX model
            input_size: Expected input size (width, height)
            debug: Print debug information
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If ONNX model can't be loaded
        """
        self.model_path = model_path
        self.in_w, self.in_h = int(input_size[0]), int(input_size[1])
        self.debug = bool(debug)
        
        # Initialize ONNX Runtime session (CPU only)
        # Using CPU provider for consistency and reproducibility
        self.sess = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        
        # Get input/output names
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        
        if self.debug:
            print("[ArcFaceEmbedderONNX] Initialized")
            print(f"  Model: {model_path}")
            print(f"  Input: {self.sess.get_inputs()[0].name} "
                  f"{self.sess.get_inputs()[0].shape} "
                  f"{self.sess.get_inputs()[0].type}")
            print(f"  Output: {self.sess.get_outputs()[0].name} "
                  f"{self.sess.get_outputs()[0].shape} "
                  f"{self.sess.get_outputs()[0].type}")
    
    def _preprocess(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess aligned face for ArcFace model.
        
        Steps:
            1. Resize to 112×112 (if needed)
            2. BGR → RGB
            3. Normalize: (pixel - 127.5) / 128.0
            4. HWC → CHW: (H,W,C) → (C,H,W)
            5. Add batch: (C,H,W) → (1,C,H,W)
        
        Args:
            aligned_bgr: Aligned face image (BGR, any size)
        
        Returns:
            Preprocessed tensor: (1, 3, 112, 112), float32, range [-1, 1]
        """
        img = aligned_bgr
        
        # Resize if needed
        if img.shape[1] != self.in_w or img.shape[0] != self.in_h:
            img = cv2.resize(img, (self.in_w, self.in_h), 
                           interpolation=cv2.INTER_LINEAR)
        
        # BGR → RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Normalize to [-1, 1]
        # This matches ArcFace training preprocessing
        rgb = (rgb - 127.5) / 128.0
        
        # HWC → CHW (channels-first format)
        # OpenCV uses (height, width, channels)
        # PyTorch/ONNX uses (channels, height, width)
        chw = np.transpose(rgb, (2, 0, 1))
        
        # Add batch dimension: (C,H,W) → (1,C,H,W)
        batch = chw[None, ...]
        
        return batch.astype(np.float32)
    
    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
        """
        L2-normalize a vector.
        
        Formula: v_norm = v / ||v||
        
        Why eps=1e-12?
            Prevents division by zero if vector is all zeros
        
        Args:
            v: Input vector (any shape)
            eps: Small epsilon for numerical stability
        
        Returns:
            Tuple of (normalized_vector, original_norm)
        
        Example:
            v = np.array([3, 4])
            v_norm, norm = _l2_normalize(v)
            # v_norm = [0.6, 0.8]
            # norm = 5.0
            # np.linalg.norm(v_norm) = 1.0
        """
        v = v.astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(v) + eps)
        v_normalized = (v / norm).astype(np.float32)
        return v_normalized, norm
    
    def embed(self, aligned_bgr: np.ndarray) -> EmbeddingResult:
        """
        Extract L2-normalized embedding from aligned face.
        
        Args:
            aligned_bgr: Aligned face image (BGR format, 112×112 preferred)
        
        Returns:
            EmbeddingResult containing:
                - embedding: (512,) L2-normalized vector
                - norm_before: Original norm before normalization
                - dim: 512
        
        Example:
            aligned = cv2.imread("face.jpg")
            result = embedder.embed(aligned)
            
            # Use embedding for matching
            similarity = np.dot(result.embedding, other_embedding)
            distance = 1 - similarity
        """
        # Preprocess
        x = self._preprocess(aligned_bgr)
        
        # Run ONNX model
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        
        # Flatten output
        v = y.reshape(-1).astype(np.float32)
        
        # L2 normalize
        v_norm, norm_before = self._l2_normalize(v)
        
        return EmbeddingResult(
            embedding=v_norm,
            norm_before=norm_before,
            dim=v_norm.size
        )


# ============================================================================
# Visualization Helpers
# ============================================================================

def draw_text_block(img, lines, origin=(10, 30), scale=0.7, color=(0, 255, 0)):
    """Draw multiple lines of text with consistent spacing."""
    x, y = origin
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   scale, color, 2, cv2.LINE_AA)
        y += int(28 * scale)


def draw_embedding_matrix(
    img: np.ndarray,
    emb: np.ndarray,
    top_left=(10, 220),
    cell_scale: int = 6,
    title: str = "embedding"
) -> Tuple[int, int]:
    """
    Visualize embedding as a heatmap matrix.
    
    Takes 512-dim vector and reshapes to 2D matrix for visualization.
    
    Args:
        img: Image to draw on
        emb: Embedding vector (512,)
        top_left: Top-left corner position
        cell_scale: Pixel size of each cell
        title: Title text
    
    Returns:
        (width, height) of drawn heatmap
    """
    D = emb.size
    
    # Reshape to approximately square matrix
    cols = int(np.ceil(np.sqrt(D)))
    rows = int(np.ceil(D / cols))
    
    # Create matrix and fill with embedding values
    mat = np.zeros((rows, cols), dtype=np.float32)
    mat.flat[:D] = emb
    
    # Normalize to [0, 1]
    mat_norm = (mat - mat.min()) / (mat.max() - mat.min() + 1e-6)
    
    # Convert to grayscale [0, 255]
    gray = (mat_norm * 255).astype(np.uint8)
    
    # Apply colormap (jet = blue → green → yellow → red)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    # Resize for visibility
    heat = cv2.resize(
        heat,
        (cols * cell_scale, rows * cell_scale),
        interpolation=cv2.INTER_NEAREST,
    )
    
    # Draw on image
    x, y = top_left
    h, w = heat.shape[:2]
    ih, iw = img.shape[:2]
    
    if x + w > iw or y + h > ih:
        return 0, 0
    
    img[y:y+h, x:x+w] = heat
    
    # Draw title
    cv2.putText(
        img,
        title,
        (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2,
    )
    
    return w, h


def emb_preview_str(emb: np.ndarray, n: int = 8) -> str:
    """
    Format first N embedding values as string.
    
    Example: "vec[0:8]: +0.123 -0.456 +0.789 ..."
    """
    vals = " ".join(f"{v:+.3f}" for v in emb[:n])
    return f"vec[0:{n}]: {vals} ..."


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalized embeddings.
    
    Since embeddings are normalized: cos_sim = dot(a, b)
    
    Returns:
        Similarity in [-1, 1] where 1 = identical, 0 = orthogonal
    """
    return float(np.dot(a.reshape(-1), b.reshape(-1)))


# ============================================================================
# Demo / Test
# ============================================================================

def main():
    """
    Demo: Visualize embeddings from live camera.
    
    Shows:
        - Live camera feed with face detection
        - Aligned face preview (top-right)
        - Embedding statistics
        - Embedding heatmap visualization
        - Frame-to-frame similarity
    """
    cap = cv2.VideoCapture(0)
    
    det = Haar5ptDetector(
        min_size=(70, 70),
        smooth_alpha=0.80,
        debug=False,
    )
    
    emb_model = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        debug=True,  # Print model info on startup
    )
    
    prev_emb: Optional[np.ndarray] = None
    
    print("\n" + "="*60)
    print("Embedding Demo")
    print("="*60)
    print("Controls:")
    print("  q - Quit")
    print("  p - Print embedding to terminal")
    print("\nLook at camera to see your embedding visualization...")
    print("="*60 + "\n")
    
    t0 = time.time()
    frames = 0
    fps = 0.0
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        vis = frame.copy()
        faces = det.detect(frame, max_faces=1)
        
        info = []
        
        if faces:
            f = faces[0]
            
            # Draw detection
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
            for (x, y) in f.kps.astype(int):
                cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
            
            # Align + embed
            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            res = emb_model.embed(aligned)
            
            # Display info
            info.append(f"embedding dim: {res.dim}")
            info.append(f"norm(before L2): {res.norm_before:.2f}")
            
            if prev_emb is not None:
                sim = cosine_similarity(prev_emb, res.embedding)
                info.append(f"cos(prev,this): {sim:.3f}")
            
            prev_emb = res.embedding
            
            # Show aligned face preview (top-right corner)
            aligned_small = cv2.resize(aligned, (160, 160))
            h, w = vis.shape[:2]
            vis[10:170, w-170:w-10] = aligned_small
            
            # Draw text info
            draw_text_block(vis, info, origin=(10, 30))
            
            # Draw embedding heatmap
            HEAT_X, HEAT_Y = 10, 220
            CELL_SCALE = 6
            ww, hh = draw_embedding_matrix(
                vis,
                res.embedding,
                top_left=(HEAT_X, HEAT_Y),
                cell_scale=CELL_SCALE,
                title="embedding heatmap",
            )
            
            # Show first 8 values below heatmap
            if ww > 0:
                cv2.putText(
                    vis,
                    emb_preview_str(res.embedding),
                    (HEAT_X, HEAT_Y + hh + 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    2,
                )
        else:
            draw_text_block(vis, ["no face"], origin=(10, 30), color=(0, 0, 255))
        
        # FPS counter
        frames += 1
        dt = time.time() - t0
        if dt >= 1.0:
            fps = frames / dt
            frames = 0
            t0 = time.time()
        
        cv2.putText(vis, f"fps: {fps:.1f}", (10, vis.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Face Embedding Demo - Press 'q' to quit", vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n✓ Demo completed")
            break
        elif key == ord("p") and prev_emb is not None:
            print("\n[Embedding Stats]")
            print(f"  Dimension: {prev_emb.size}")
            print(f"  Min value: {prev_emb.min():.6f}")
            print(f"  Max value: {prev_emb.max():.6f}")
            print(f"  Mean: {prev_emb.mean():.6f}")
            print(f"  Std: {prev_emb.std():.6f}")
            print(f"  Norm: {np.linalg.norm(prev_emb):.6f}")
            print(f"  First 10: {prev_emb[:10]}\n")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
