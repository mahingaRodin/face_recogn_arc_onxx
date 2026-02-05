from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .haar_5pt import Haar5ptDetector, align_face_5pt
from .embed import ArcFaceEmbedderONNX


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EnrollConfig:
    """Enrollment configuration parameters."""
    
    # Database paths
    out_db_npz: Path = Path("data/db/face_db.npz")      # Embeddings storage
    out_db_json: Path = Path("data/db/face_db.json")    # Metadata storage
    
    # Crop storage
    save_crops: bool = True                              # Save aligned crops?
    crops_dir: Path = Path("data/enroll")                # Crop directory
    
    # Capture requirements
    samples_needed: int = 15                             # Minimum samples
    auto_capture_every_s: float = 0.25                   # Auto-capture interval
    max_existing_crops: int = 300                        # Max crops to load
    
    # UI window names
    window_main: str = "enroll"
    window_aligned: str = "aligned_112"


# ============================================================================
# Database Management
# ============================================================================

def ensure_dirs(cfg: EnrollConfig) -> None:
    """Create necessary directories if they don't exist."""
    cfg.out_db_npz.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_db_json.parent.mkdir(parents=True, exist_ok=True)
    if cfg.save_crops:
        cfg.crops_dir.mkdir(parents=True, exist_ok=True)


def load_db(cfg: EnrollConfig) -> Dict[str, np.ndarray]:
    """
    Load existing face database.
    
    Returns:
        Dictionary mapping name → embedding vector
        Empty dict if database doesn't exist
    """
    if cfg.out_db_npz.exists():
        data = np.load(cfg.out_db_npz, allow_pickle=True)
        return {k: data[k].astype(np.float32) for k in data.files}
    return {}


def save_db(cfg: EnrollConfig, db: Dict[str, np.ndarray], meta: dict) -> None:
    """
    Save face database and metadata.
    
    Args:
        cfg: Configuration
        db: Dictionary of {name: embedding}
        meta: Metadata dictionary
    """
    ensure_dirs(cfg)
    
    # Save embeddings (binary, efficient)
    np.savez(cfg.out_db_npz, **{k: v.astype(np.float32) for k, v in db.items()})
    
    # Save metadata (JSON, human-readable)
    cfg.out_db_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def mean_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Compute mean embedding and L2-normalize.
    
    Why average embeddings?
        - Single sample may be noisy (bad lighting, weird angle)
        - Average captures "typical" appearance
        - More robust to variations
    
    Args:
        embeddings: List of embedding vectors
    
    Returns:
        Mean embedding (L2-normalized)
    
    Example:
        samples = [emb1, emb2, emb3, ...]  # 15 samples
        template = mean_embedding(samples)  # One template
    """
    # Stack embeddings into matrix
    E = np.stack([e.reshape(-1) for e in embeddings], axis=0).astype(np.float32)
    
    # Compute mean
    m = E.mean(axis=0)
    
    # L2 normalize
    m = m / (np.linalg.norm(m) + 1e-12)
    
    return m.astype(np.float32)


# ============================================================================
# Existing Crops Loader (Re-enrollment Support)
# ============================================================================

def _list_existing_crops(person_dir: Path, max_count: int) -> List[Path]:
    """
    List existing aligned crop images for a person.
    
    Args:
        person_dir: Directory like data/enroll/Alice/
        max_count: Maximum number of crops to load
    
    Returns:
        List of image paths (sorted, most recent last)
    """
    if not person_dir.exists():
        return []
    
    files = sorted([p for p in person_dir.glob("*.jpg") if p.is_file()])
    
    # Limit to prevent memory issues
    if len(files) > max_count:
        files = files[-max_count:]
    
    return files


def load_existing_samples_from_crops(
    cfg: EnrollConfig,
    emb: ArcFaceEmbedderONNX,
    person_dir: Path,
) -> List[np.ndarray]:
    """
    Load and re-embed existing aligned crops from disk.
    
    This enables re-enrollment:
        - Previous session saved 10 crops
        - This session adds 10 more
        - Total: 20 samples for template
    
    Args:
        cfg: Configuration
        emb: Embedder model
        person_dir: Directory containing crops
    
    Returns:
        List of embeddings from existing crops
    """
    if not cfg.save_crops:
        return []
    
    crops = _list_existing_crops(person_dir, cfg.max_existing_crops)
    base: List[np.ndarray] = []
    
    for p in crops:
        img = cv2.imread(str(p))
        if img is None:
            continue
        
        try:
            r = emb.embed(img)
            base.append(r.embedding)
        except Exception:
            # Skip corrupted images
            continue
    
    return base


# ============================================================================
# UI Helpers
# ============================================================================

def draw_status(
    frame: np.ndarray,
    name: str,
    base_count: int,
    new_count: int,
    needed: int,
    auto: bool,
    msg: str = "",
) -> None:
    """
    Draw enrollment status overlay on frame.
    
    Args:
        frame: Image to draw on (modified in-place)
        name: Person's name
        base_count: Existing samples from disk
        new_count: New samples this session
        needed: Required sample count
        auto: Auto-capture mode enabled?
        msg: Status message
    """
    total = base_count + new_count
    
    lines = [
        f"ENROLL: {name}",
        f"Existing: {base_count} | New: {new_count} | Total: {total} / {needed}",
        f"Auto: {'ON' if auto else 'OFF'} (toggle: a)",
        "SPACE=capture | s=save | r=reset NEW | q=quit",
    ]
    
    if msg:
        lines.insert(0, msg)
    
    # Draw with shadow for readability
    y = 30
    for line in lines:
        # Shadow (black)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.62, (0, 0, 0), 4, cv2.LINE_AA)
        # Text (white)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 26


# ============================================================================
# Main Enrollment Function
# ============================================================================

def main():
    """
    Main enrollment process.
    
    Steps:
        1. Prompt for person's name
        2. Load existing samples (if re-enrolling)
        3. Open camera
        4. Capture samples (manual or auto)
        5. Save template to database
    """
    cfg = EnrollConfig()
    ensure_dirs(cfg)
    
    # Step 1: Get person's name
    name = input("Enter person name to enroll (e.g., Alice): ").strip()
    if not name:
        print("❌ No name provided. Exiting.")
        return
    
    # Step 2: Initialize pipeline
    print("\n⏳ Initializing detector and embedder...")
    
    det = Haar5ptDetector(
        min_size=(70, 70),
        smooth_alpha=0.80,
        debug=False
    )
    
    emb = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False
    )
    
    # Step 3: Load existing database
    db = load_db(cfg)
    
    # Step 4: Setup person directory
    person_dir = cfg.crops_dir / name
    if cfg.save_crops:
        person_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 5: Load existing samples (re-enrollment support)
    base_samples: List[np.ndarray] = load_existing_samples_from_crops(cfg, emb, person_dir)
    new_samples: List[np.ndarray] = []
    
    status_msg = ""
    if base_samples:
        status_msg = f"✓ Loaded {len(base_samples)} existing samples from disk"
        print(status_msg)
    
    # Step 6: Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Failed to open camera")
    
    # Step 7: Setup UI
    cv2.namedWindow(cfg.window_main, cv2.WINDOW_NORMAL)
    cv2.namedWindow(cfg.window_aligned, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cfg.window_aligned, 240, 240)
    
    print("\n" + "="*60)
    print("✓ Enrollment started")
    if base_samples:
        print(f"✓ Re-enrollment mode: {len(base_samples)} existing samples found")
    print("\nTips:")
    print("  - Move slightly left/right")
    print("  - Try different expressions")
    print("  - Stable lighting works best")
    print("\nControls:")
    print("  SPACE - Capture sample")
    print("  a     - Toggle auto-capture")
    print("  s     - Save enrollment")
    print("  r     - Reset new samples")
    print("  q     - Quit")
    print("="*60 + "\n")
    
    # Auto-capture state
    auto = False
    last_auto = 0.0
    
    # FPS tracking
    t0 = time.time()
    frames = 0
    fps: Optional[float] = None
    
    try:
        while True:
            # Read frame
            ok, frame = cap.read()
            if not ok:
                break
            
            vis = frame.copy()
            
            # Detect faces
            faces = det.detect(frame, max_faces=1)
            
            aligned: Optional[np.ndarray] = None
            
            if faces:
                f = faces[0]
                
                # Draw detection
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
                for (x, y) in f.kps.astype(int):
                    cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
                
                # Align face
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                cv2.imshow(cfg.window_aligned, aligned)
            else:
                # Show blank if no face
                cv2.imshow(cfg.window_aligned, np.zeros((112, 112, 3), dtype=np.uint8))
            
            # Auto-capture logic
            now = time.time()
            if auto and aligned is not None and (now - last_auto) >= cfg.auto_capture_every_s:
                # Embed and store
                r = emb.embed(aligned)
                new_samples.append(r.embedding)
                last_auto = now
                status_msg = f"Auto captured NEW ({len(new_samples)})"
                
                # Save crop
                if cfg.save_crops:
                    fn = person_dir / f"{int(now * 1000)}.jpg"
                    cv2.imwrite(str(fn), aligned)
            
            # Calculate FPS
            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()
            
            # Display FPS
            if fps is not None:
                cv2.putText(vis, f"FPS: {fps:.1f}", (10, vis.shape[0] - 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Draw status overlay
            draw_status(
                vis,
                name=name,
                base_count=len(base_samples),
                new_count=len(new_samples),
                needed=cfg.samples_needed,
                auto=auto,
                msg=status_msg,
            )
            
            cv2.imshow(cfg.window_main, vis)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):
                print("\n⚠ Quit without saving")
                break
            
            if key == ord("a"):
                auto = not auto
                status_msg = f"Auto mode {'ON' if auto else 'OFF'}"
                print(f"  {status_msg}")
            
            if key == ord("r"):
                new_samples.clear()
                status_msg = "NEW samples reset (existing kept)"
                print(f"  {status_msg}")
            
            if key == ord(" "):  # SPACE
                if aligned is None:
                    status_msg = "No face detected. Not captured."
                    print(f"  {status_msg}")
                else:
                    # Embed and store
                    r = emb.embed(aligned)
                    new_samples.append(r.embedding)
                    status_msg = f"Captured NEW ({len(new_samples)})"
                    print(f"  ✓ Sample {len(new_samples)} captured")
                    
                    # Save crop
                    if cfg.save_crops:
                        fn = person_dir / f"{int(time.time() * 1000)}.jpg"
                        cv2.imwrite(str(fn), aligned)
            
            if key == ord("s"):
                total = len(base_samples) + len(new_samples)
                
                # Check minimum requirement
                if total < max(3, cfg.samples_needed // 2):
                    status_msg = f"Not enough samples ({total}). Need at least {cfg.samples_needed // 2}"
                    print(f"  ⚠ {status_msg}")
                    continue
                
                # Compute template
                print(f"\n⏳ Computing template from {total} samples...")
                all_samples = base_samples + new_samples
                template = mean_embedding(all_samples)
                
                # Update database
                db[name] = template
                
                # Save
                meta = {
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "embedding_dim": int(template.size),
                    "names": sorted(db.keys()),
                    "samples_existing_used": int(len(base_samples)),
                    "samples_new_used": int(len(new_samples)),
                    "samples_total_used": int(len(all_samples)),
                    "note": "Embeddings are L2-normalized vectors. Matching uses cosine similarity.",
                }
                
                save_db(cfg, db, meta)
                
                status_msg = f"✓ Saved '{name}' to DB. Total identities: {len(db)}"
                print(f"\n{status_msg}")
                print(f"  Database: {cfg.out_db_npz}")
                print(f"  Metadata: {cfg.out_db_json}")
                
                # Reload base samples so UI matches reality
                base_samples = load_existing_samples_from_crops(cfg, emb, person_dir)
                new_samples.clear()
                
                print(f"\n✓ Enrollment complete! Press 'q' to quit or continue capturing.\n")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()