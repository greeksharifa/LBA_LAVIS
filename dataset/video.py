import av
import numpy as np


def read_video_pyav(video_path, n_frms):
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
            
    clip = np.stack([x.to_ndarray(format="rgb24") for x in frames])
    
    return clip

