import os

import tensorflow as tf

from model import Model

tf.flags.DEFINE_string('gpu', '', 'The devices visible.')
tf.flags.DEFINE_string('data_dir', '/Users/aglax/Downloads/20bn-something-something-v2', 'The directory of data.')
tf.flags.DEFINE_string('tfrecords_dir', '/Users/aglax/OneDrive/Projects/something-something-i3d/data/selected-10', 'Path to tfreocrods directory.')
tf.flags.DEFINE_integer('batch_size', 8, 'Batch size.')
tf.flags.DEFINE_integer('height', 224, 'Height of input video.')
tf.flags.DEFINE_integer('width', 224, 'Width of input video.')
tf.flags.DEFINE_integer('length', 20, 'The length of input video.')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
tf.flags.DEFINE_integer('train_step_per_epoch', 1000, 'The number of steps in one training epoch.')
tf.flags.DEFINE_integer('valid_step_per_epoch', 10, 'The number of steps in one validation epoch.')
tf.flags.DEFINE_integer('epoch', 40, 'The number of epochs.')
tf.flags.DEFINE_integer('valid_interval', 1000, 'The interval between two validation.')
tf.flags.DEFINE_float('dropout', 0.9, 'The keep probability of dropout.')

FLAGS = tf.flags.FLAGS


def train():
    with tf.Session() as sess:
        model = Model(
            sess,
            FLAGS.data_dir,
            FLAGS.tfrecords_dir,
            FLAGS.batch_size,
            FLAGS.height,
            FLAGS.width,
            FLAGS.length,
            FLAGS.learning_rate,
            FLAGS.train_step_per_epoch,
            FLAGS.valid_step_per_epoch,
            FLAGS.valid_interval,
            FLAGS.dropout
        )
        sess.run(tf.global_variables_initializer())
        model.valid()
        for _ in range(FLAGS.epoch):
            model.train()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    tf.logging.set_verbosity(tf.logging.INFO)
    train()


if __name__ == '__main__':
    main()
