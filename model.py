import os

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim import nets

from dataset import Dataset
from i3d import InceptionI3d
from rhn import recurrent_highway_network
from utils import save_generated_videos


class Model:
    def __init__(self,
                 sess,
                 data_dir,
                 tfrecords_dir,
                 batch_size,
                 height,
                 width,
                 length,
                 learning_rate,
                 train_step_per_epoch,
                 valid_step_per_epoch,
                 valid_interval,
                 dropout,
                 class_num,
                 generated_videos_dir):
        self.sess = sess
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.length = length
        self.class_num = class_num
        self.generated_videos_dir = generated_videos_dir
        self.train_step_per_epoch = train_step_per_epoch
        self.valid_step_per_epoch = valid_step_per_epoch
        self.valid_interval = valid_interval

        train_dataset = Dataset(
            data_dir,
            os.path.join(tfrecords_dir, 'train'),
            'train*.tfrecords',
            height,
            width,
            length,
            batch_size
        )
        self.train_inputs, train_meta = train_dataset.input_fn()
        self.train_target = train_meta['class_index']
        self.train_ids = train_meta['id']
        self.train_inputs = tf.reshape(self.train_inputs, [batch_size] + self.train_inputs.get_shape().as_list()[1:])
        self.train_target_onehot = tf.one_hot(self.train_target, self.class_num)

        valid_dataset = Dataset(
            data_dir,
            os.path.join(tfrecords_dir, 'valid'),
            'valid*.tfrecords',
            height,
            width,
            length,
            batch_size
        )
        self.valid_inputs, valid_meta = valid_dataset.input_fn()
        self.valid_target = valid_meta['class_index']
        self.valid_ids = valid_meta['id']
        self.valid_inputs = tf.reshape(self.valid_inputs, [batch_size] + self.valid_inputs.get_shape().as_list()[1:])

        self.train_outputs, self.train_l2_loss = self._build_rhn(self.train_inputs, reuse=False, is_training=True,
                                                                 dropout_keep_prob=dropout)
        self.valid_outputs, self.valid_l2_loss = self._build_rhn(self.valid_inputs, reuse=True, is_training=False,
                                                                 dropout_keep_prob=1.)

        for variable in tf.global_variables():
            print(variable.name, variable.shape)

        tf.logging.debug(
            'LABEL SHAPE: {} LOGIT SHAPE: {}'.format(self.train_target_onehot.shape, self.train_outputs.shape))

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(
            learning_rate,
            self.global_step,
            10000,
            0.95,
            staircase=True,
        )

        # factor = tf.cast(tf.reduce_min([self.global_step / 10000, 1]), dtype=tf.float32)
        factor = 0.
        # ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.train_target_onehot, logits=self.train_outputs))
        ce_loss = 0.
        self.loss = factor * ce_loss + (1. - factor) * self.train_l2_loss

        # tvars = tf.trainable_variables()
        # train_var_list = [var for var in tvars if 'Inception' not in var.name]
        # self.op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=train_var_list)
        self.op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        # self.top1, self.top5, = self._top_1_and_5(self.valid_outputs)

    def _build_i3d(self, inputs, reuse=False, is_training=True, dropout_keep_prob=0.8):
        with tf.variable_scope('RGB', reuse=reuse):
            encoder_model = InceptionI3d(
                self.class_num,
                spatial_squeeze=True,
                final_endpoint='Predictions'
            )
            logits, endpoints = encoder_model(
                inputs,
                is_training=is_training,
                dropout_keep_prob=dropout_keep_prob
            )
            return logits, endpoints

    def _build_inception_v1(self, inputs, reuse, is_training, dropout_keep_prob):
        with slim.arg_scope(nets.inception.inception_v1_arg_scope()):
            logits, endpoints = nets.inception.inception_v1(
                inputs, self.class_num, is_training=is_training, reuse=reuse,
                dropout_keep_prob=dropout_keep_prob)
        return logits, endpoints

    def _build_rhn(self, inputs, reuse, is_training, dropout_keep_prob):
        logits, hidden_outputs, l2_loss = recurrent_highway_network(inputs, 2, [32, 32], 2, stride=2,
                                                                    layer_to_extract=0)
        return logits, l2_loss

    def _build_conv3d_classification_module(self, hidden_outputs):

        predicted_class = tf.layers.conv3d(hidden_outputs, 41, 4)

        return predicted_class

    def _top_1_and_5(self, outputs):
        top5_values, top5 = tf.nn.top_k(outputs, 5)
        top1 = top5[:, 0]
        return top1, top5

    def _load_from_kinetics_pretrain(self):
        rgb_variable_map = {}
        for variable in tf.global_variables():
            tf.logging.debug('{}, {}'.format(variable.name, variable.shape))
            if variable.name.split('/')[0] == 'RGB' and 'Logits' not in variable.name:
                rgb_variable_map[variable.name.replace(':0', '')] = variable
        saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
        saver.restore(self.sess, './data/checkpoints/rgb_imagenet/model.ckpt')

    def _load_from_inception_v1_pretrain(self):
        rgb_variable_map = {}
        for variable in tf.global_variables():
            variable_name = variable.name
            variable_name = variable_name[10:]
            if variable_name.startswith('InceptionV1') and 'Logits' not in variable_name:
                rgb_variable_map[variable_name.replace(':0', '')] = variable
        saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
        saver.restore(self.sess, './data/checkpoints/inception_v1/inception_v1.ckpt')

    def _validate_batch(self):
        top1, top5, target, logits = self.sess.run([self.top1, self.top5, self.valid_target, self.valid_outputs])
        results = {
            'top1': top1,
            'top5': top5,
            'target': target,
            'logits': logits,
        }
        return results

    def valid(self):
        tf.logging.debug('VALIDATING...')
        for i in range(1, self.valid_step_per_epoch + 1):
            results = self._validate_batch()
            tf.logging.info(
                'VALID: step - {} top1 - {} logits - {} target - {}'.format(i, results['top1'], results['logits'],
                                                                            results['target']))

    def _train_batch(self):
        _, loss, global_step, train_inputs, train_outputs, ids = self.sess.run([self.op, self.loss, self.global_step, self.train_inputs, self.train_outputs, self.train_ids])
        results = {
            'loss': loss,
            'global_step': global_step,
        }
        if global_step % 100 == 1:
            save_generated_videos(train_inputs, ids, self.generated_videos_dir, global_step, 'gt', 0)
            save_generated_videos(train_outputs, ids, self.generated_videos_dir, global_step, 'pred', 1)
        return results

    def train(self):
        tf.logging.debug('TRAINING...')
        for _ in range(self.train_step_per_epoch):
            results = self._train_batch()
            global_step = results['global_step']
            tf.logging.info('TRAIN: step - {} loss - {}'.format(global_step, results['loss']))
            # if (global_step + 1) % self.valid_interval == 0:
            #     self.valid()
