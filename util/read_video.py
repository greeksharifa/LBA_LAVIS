import cv2
import numpy as np
from typing import Tuple, Optional


def read_video(
    video_path: str,
    num_frames: Optional[int] = None,
    do_sample_frames: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Read video data and metadata from a given path.
    
    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to extract. If None, extracts all frames.
        do_sample_frames: If True, uniformly sample frames; if False, take first N frames.
    
    Returns:
        video: numpy array of shape (num_frames, height, width, 3) in RGB format.
        metadata: dict containing video metadata.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0.0
    
    # Determine which frames to extract
    if num_frames is None:
        num_frames = total_frames
    
    num_frames = min(num_frames, total_frames)
    
    if do_sample_frames and num_frames < total_frames:
        # Uniformly sample frames
        frames_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    else:
        # Take first N frames
        frames_indices = list(range(num_frames))
    
    # Read frames
    frames = []
    for frame_idx in frames_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR (OpenCV) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        else:
            # Handle missing frame by duplicating last valid frame or using zeros
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((height, width, 3), dtype=np.uint8))
    
    cap.release()
    
    # Stack frames into numpy array: (num_frames, height, width, 3)
    video = np.stack(frames, axis=0)
    
    # Build metadata (matching vLLM's VideoAsset format)
    metadata = {
        'total_num_frames': len(frames),
        'fps': round(fps / (total_frames / num_frames), 4) if do_sample_frames else fps,
        'duration': duration,
        'video_backend': 'opencv',
        'frames_indices': frames_indices,
        'do_sample_frames': do_sample_frames,
    }
    
    return video, metadata


# Example usage
if __name__ == "__main__":
    video_path = "/data/DramaQA/AnotherMissOh_videos/total/AnotherMissOh14_001_0000.mp4"
    
    # Read 16 frames from video
    video, metadata = read_video(video_path, num_frames=32, do_sample_frames=True)
    
    print(f"video.shape: {video.shape}")  
    print(f"metadata: {metadata}")
    '''
    video.shape: (32, 768, 1024, 3)
    metadata: {
        'total_num_frames': 32, 
        'fps': 0.1486, 
        'duration': 215.36, 
        'video_backend': 'opencv', 
        'frames_indices': [0, 173, 347, 520, 694, 868, 1041, 1215, 1389, 1562, 1736, 1910, 2083, 2257, 2431, 2604, 2778, 2951, 3125, 3299, 3472, 3646, 3820, 3993, 4167, 4341, 4514, 4688, 4862, 5035, 5209, 5383], 
        'do_sample_frames': True
    }
    '''