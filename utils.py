import os
import shutil

import cv2
import numpy as np
import tensorflow as tf


def save_generated_videos(videos, ids, save_dir, step, prefix='', start_index=0):
    assert np.ndim(
        videos) == 5, f'Shape error: videos shape {videos.shape} has a rank of {np.ndim(videos)} not expected 5.'
    assert ids.shape[0] == videos.shape[
        0], f'Shape error: ids shape {ids.shape} and videos shape {videos.shape} are not compatible.'

    videos = videos[:, :, :, :, ::-1]
    videos = ((videos + 1) * 127.5).astype(np.uint8)
    for video, id in zip(videos, ids):
        video_dir = os.path.join(save_dir, f'{step}_{id}')
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        tf.logging.debug(f'{video.shape}')
        for frame_index, frame in enumerate(video):
            cv2.imwrite(os.path.join(video_dir, f'{prefix}_{frame_index+start_index}.png'), frame)

        tf.logging.debug(f'{id} Saved.')
