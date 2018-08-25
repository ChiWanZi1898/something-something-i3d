import tensorflow as tf

from dataset import Dataset
from i3d import InceptionI3d

import os


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
                 dropout=0.9,
                 class_num=41):
        self.sess = sess
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.length = length
        self.class_num = class_num
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
        self.train_inputs, _, self.train_target = train_dataset.input_fn()
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
        self.valid_inputs, _, self.valid_target = valid_dataset.input_fn()

        self.train_outputs, _ = self._build(self.train_inputs, reuse=False, is_training=True, dropout=dropout)
        self.valid_outputs, _ = self._build(self.valid_inputs, reuse=True, is_training=False, dropout=1.)
        self._load_from_pretrain()

        tf.logging.debug('LABEL SHAPE: {} LOGIT SHAPE: {}'.format(self.train_target_onehot.shape, self.train_outputs.shape))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.train_target_onehot, logits=self.train_outputs))

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(
            learning_rate,
            self.global_step,
            10000,
            0.95,
            staircase=True,
        )
        self.op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.top1, self.top5 = self._top_1_and_5(self.valid_outputs)

    def _build(self, inputs, reuse, is_training, dropout):
        with tf.variable_scope('RGB', reuse=reuse):
            encoder_model = InceptionI3d(
                self.class_num,
                spatial_squeeze=True,
                final_endpoint='Predictions'
            )
            logits, endpoints = encoder_model(
                inputs,
                is_training=is_training,
                dropout_keep_prob=dropout
            )
            return logits, endpoints

    def _train_batch(self):
        _, loss, global_step = self.sess.run([self.op, self.loss, self.global_step])
        results = {
            'loss': loss,
            'global_step': global_step,
        }
        return results

    def _top_1_and_5(self, outputs):
        top5_values, top5 = tf.nn.top_k(outputs, 5)
        top1 = top5[:, 0]
        return top1, top5

    def _validate_batch(self):
        top1, top5, target = self.sess.run([self.top1, self.top5, self.valid_target])
        results = {
            'top1': top1,
            'top5': top5,
            'target': target
        }
        return results

    def _load_from_pretrain(self):
        rgb_variable_map = {}
        for variable in tf.global_variables():
            tf.logging.debug('{}, {}'.format(variable.name, variable.shape))
            if variable.name.split('/')[0] == 'RGB' and 'Logits' not in variable.name:
                rgb_variable_map[variable.name.replace(':0', '')] = variable
        saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
        saver.restore(self.sess, './data/checkpoints/rgb_imagenet/model.ckpt')

    def valid(self):
        tf.logging.debug('VALIDATING...')
        for i in range(self.valid_step_per_epoch):
            results = self._validate_batch()
            tf.logging.info('VALID: step - {} loss - {}'.format(i, results['top1']))

    def train(self):
        tf.logging.debug('TRAINING...')
        for _ in range(self.train_step_per_epoch):
            results = self._train_batch()
            global_step = results['global_step']
            tf.logging.info('TRAIN: step - {} loss - {}'.format(global_step, results['loss']))
            if global_step % self.valid_interval == 0:
                self.valid()

