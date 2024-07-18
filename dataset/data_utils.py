"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import gzip
import logging
import os
import random as rnd
import tarfile
import zipfile
import cv2
import sys

import decord
import webdataset as wds
import numpy as np
import torch
from decord import VideoReader
from tqdm import tqdm

decord.bridge.set_bridge("torch")
MAX_INT = sys.maxsize


def load_video(video_path, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform"):
    vr = VideoReader(uri=video_path, height=height, width=width)

    vlen = len(vr)
    start, end = 0, vlen

    n_frms = min(n_frms, vlen)

    if sampling == "uniform":
        indices = np.arange(start, end, vlen / n_frms).astype(int)
    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    frms = vr.get_batch(indices).permute(3, 0, 1, 2).float()  # (C, T, H, W)

    return frms

def head_tail_frame_sampling(video_path, num_frames, target_height, target_width, start_time=None, end_time=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = total_frames / frame_rate

    start_frame = int(start_time * frame_rate)
    end_frame = int(end_time * frame_rate)
    frame_indices = [start_frame] + [start_frame + (end_frame - start_frame) // (num_frames - 1) * i for i in range(1, num_frames - 1)] + [end_frame]

    frames = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (target_width, target_height))
        frames.append(frame)

    cap.release()
    if len(frames) == 0:
        return None
    return torch.stack([torch.tensor(f).permute(2,0,1).float() for f in frames], dim=1)

def uniform_frame_sampling(video_path, num_frames, target_height, target_width, start_time=None, end_time=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = total_frames / frame_rate

    start_frame = int(start_time * frame_rate)
    end_frame = int(end_time * frame_rate)
    frame_indices = list(range(start_frame, end_frame + 1, (end_frame - start_frame + 1) // num_frames))

    frames = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (target_width, target_height))
        frames.append(frame)

    cap.release()
    return frames


def load_clip(video_path, num_frames, target_height, target_width, start_time=None, end_time=None, sampling="headtail"):
    if sampling == "headtail":
        return head_tail_frame_sampling(video_path, num_frames, target_height, target_width, start_time, end_time)
    elif sampling == "uniform":
        return uniform_frame_sampling(video_path, num_frames, target_height, target_width, start_time, end_time)
    else:
        raise NotImplementedError