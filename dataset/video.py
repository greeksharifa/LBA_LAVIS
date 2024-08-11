import numpy as np

import av
import cv2


# Example usage:
# video_frames = process_video_cv2("path/to/your/video.mp4", n_frms=10, start_time=5, end_time=15)
def process_video_cv2(video_path, n_frms, start_time=0, end_time=None):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Handle end_time
    if end_time is None or end_time > duration:
        end_time = duration

    # Convert times to frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Calculate frames to sample
    frames_to_sample = np.linspace(start_frame, end_frame - 1, n_frms, dtype=int)

    frames = []
    current_frame = 0

    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame >= start_frame and current_frame in frames_to_sample:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        current_frame += 1

    # Release the video capture object
    cap.release()

    # Ensure we have exactly n_frms
    if len(frames) < n_frms:
        # Pad with zeros if we don't have enough frames
        last_frame = frames[-1] if frames else np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 
                                                        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                                        3), dtype=np.uint8)
        frames.extend([np.zeros_like(last_frame) for _ in range(n_frms - len(frames))])
    elif len(frames) > n_frms:
        # Truncate if we somehow got too many frames
        frames = frames[:n_frms]

    return frames



# Example usage:
# video_frames = process_video("path/to/your/video.mp4", n_frms=10, start_time=5, end_time=15)
def read_video_pyav(video_path, n_frms, start_time=0, end_time=None):
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    
    # Get video properties
    fps = video_stream.average_rate
    total_frames = video_stream.frames
    duration = total_frames / fps

    # Handle end_time
    if end_time is None or end_time > duration:
        end_time = duration

    # Convert times to frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Calculate frames to sample
    frames_to_sample = np.linspace(start_frame, end_frame - 1, n_frms, dtype=int)
    SUPPLE_N = 3
    try:    
        frames_to_sample_supple = [sorted(np.random.choice(range(start_frame, end_frame - 1), n_frms, replace=(end_frame-start_frame) <= n_frms)) for _ in range(SUPPLE_N)]
    except:
        frames_to_sample_supple = [frames_to_sample for _ in range(SUPPLE_N)]

    # Seek to start_frame
    container.seek(int(start_frame * video_stream.time_base * 1000000))  # Seek in microseconds

    frames = []
    frames_supple = [[] for _ in range(SUPPLE_N)]
    for frame_idx, frame in enumerate(container.decode(video=0)):
        if frame_idx + start_frame >= end_frame:
            break
        if frame_idx + start_frame in frames_to_sample:
            frames.append(frame.to_ndarray(format="rgb24"))
        for i in range(SUPPLE_N):
            if frame_idx + start_frame in frames_to_sample_supple[i]:
                frames_supple[i].append(frame.to_ndarray(format="rgb24"))

    # Ensure we have exactly n_frms
    if len(frames) < n_frms:
        # Pad with zeros if we don't have enough frames
        last_frame = frames[-1] if frames else np.zeros((video_stream.height, video_stream.width, 3), dtype=np.uint8)
        frames.extend([np.zeros_like(last_frame) for _ in range(n_frms - len(frames))])
    elif len(frames) > n_frms:
        # Truncate if we somehow got too many frames
        frames = frames[:n_frms]

    return frames, frames_supple





def backup_read_video_pyav(video_path, n_frms):
    container = av.open(video_path)
    
    # sample uniformly 'n_frms' frames from the video
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / n_frms).astype(int)
    
    frames = []
    
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    
    
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
            
    frms = np.stack([x.to_ndarray(format="rgb24") for x in frames])
    
    return frms


