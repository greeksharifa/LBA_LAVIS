import requests
import base64
import numpy as np
import cv2


def ndarrays_to_base64(frame_list, add_prefix=False):
    base64_frames = []
    
    for frame in frame_list:
        # Ensure the frame is in uint8 format
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        
        # Convert the buffer to base64
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        if add_prefix:
            jpg_as_text = "data:image/jpeg;base64," + jpg_as_text
            # "data:image;base64," + 
        
        base64_frames.append(jpg_as_text)
    
    return base64_frames
