import cv2
from PIL import Image
import numpy as np


def load_video_to_sampled_frames(video_path, n_frms):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the indices of the frames to sample uniformly
    sample_indices = np.linspace(0, total_frames - 1, n_frms, dtype=int)
    
    def _remove_duplicates(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]
    
    sample_indices = _remove_duplicates(sample_indices)
    
    # List to store the frames
    frames = []
    
    for idx in sample_indices:
        # Set the current frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame from BGR (OpenCV format) to RGB (PIL format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the frame to PIL image
        pil_image = Image.fromarray(frame_rgb)
        
        # Append the PIL image to the list
        frames.append(pil_image)
    
    # Release the video capture object
    cap.release()
    
    return frames


def demo():
    # Example usage
    video_path = 'path/to/your/video.mp4'
    frames = load_video_to_sampled_frames(video_path)
    print(f"Loaded {len(frames)} frames from the video.")
