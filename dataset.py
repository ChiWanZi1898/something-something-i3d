from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np
import tensorflow as tf


class Dataset:
    def __init__(self,
                 data_dir,
                 tfrecords_dir,
                 tfrecords_pattern,
                 height,
                 width,
                 length,
                 batch_size,
                 epoch=None,
                 thread_num=4):
        self.data_dir = data_dir
        self.tfrecords_dir = tfrecords_dir
        self.tfrecords_pattern = tfrecords_pattern
        self.height = height
        self.width = width
        self.length = length
        self.batch_size = batch_size
        self.epoch = epoch
        self.thread_num = thread_num

    def _get_frames(self, video_id):

        video_path = os.path.join(self.data_dir, '{}.webm'.format(video_id))
        cap = cv2.VideoCapture(video_path)

        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        frames_np = np.array(frames, dtype=np.uint8)
        tf.logging.debug('CLAIM FRAME NUM: {} REAL FRAME NUM: {}'.format(num_frames, frames_np.shape[0]))

        # bgr to rgb
        frames_np = frames_np[:, :, :, ::-1]

        # fps
        tf.logging.debug('FPS: {}'.format(fps))

        # select frames
        frames_np = np.stack([frames_np[i % frames_np.shape[0]] for i in range(self.length)], axis=0)

        # fixed aspect ratio resize
        cur_frame_num, cur_height, cur_width, cur_channels = frames_np.shape
        min_height = max(self.height, int(cur_height / cur_width * self.width))
        min_width = max(self.width, int(cur_width / cur_height * self.height))
        tmp_frames_np = np.zeros([cur_frame_num, min_height, min_width, cur_channels], dtype=frames_np.dtype)
        for i, frame in enumerate(frames_np):
            tmp_frames_np[i] = cv2.resize(frame, dsize=(min_width, min_height), interpolation=cv2.INTER_LINEAR)
        frames_np = tmp_frames_np

        # center crop
        cur_frame_num, cur_height, cur_width, cur_channels = frames_np.shape
        height_offset = (cur_height - self.height) // 2
        width_offset = (cur_width - self.width) // 2
        assert height_offset >= 0, 'Error: current height {} < target height {}.'.format(cur_height, self.height)
        assert width_offset >= 0, 'Error: current width {} < target width {}.'.format(cur_width, self.width)
        frames_np = frames_np[:, height_offset:height_offset + self.height, width_offset:width_offset + self.width]

        return frames_np

    def _parse_example(self, serial_example):
        features = {
            'id': tf.FixedLenFeature((), tf.int64, default_value=-1),
            'class_index': tf.FixedLenFeature((), tf.int64, default_value=-1),
            'subclass_index': tf.FixedLenFeature((), tf.int64, default_value=-1),
        }
        parsed_features = tf.parse_single_example(serial_example, features)
        frames = tf.py_func(self._get_frames, [parsed_features['id']], tf.uint8)
        frames = tf.cast(frames, dtype=tf.float32)
        frames = frames / 127.5 - 1
        frames = tf.reshape(frames, (self.length, self.height, self.width, 3))
        return frames, parsed_features['class_index'], parsed_features['subclass_index']

    def _get_tfrecords_files(self):
        full_pattern = os.path.join(self.tfrecords_dir, self.tfrecords_pattern)
        tfrecords_files = tf.matching_files(full_pattern)
        return tfrecords_files

    def _get_dataset(self):
        tfrecords_files = self._get_tfrecords_files()
        dataset = tf.data.TFRecordDataset(tfrecords_files)
        dataset = dataset.repeat(self.epoch)
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            map_func=self._parse_example, batch_size=self.batch_size, num_parallel_calls=self.thread_num))
        dataset = dataset.prefetch(self.batch_size)
        return dataset

    def input_fn(self):
        dataset = self._get_dataset()
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
