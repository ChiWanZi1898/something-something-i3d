__author__ = 'yunbo'

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim import nets

from StrideHighwayCell import HighwayCell


def recurrent_highway_network(images, num_layers,
                              num_hidden, filter_size, stride=1,
                              total_length=20, input_length=10, tln=True, is_training=True, dropout_keep_prob=1.0, reuse=False):
    gen_images = []
    abstract_features = []
    rhn_layer = []
    shape = images.get_shape().as_list()
    output_channels = shape[-1]

    for layer in range(num_layers // 2):
        new_rhn = HighwayCell(
            'rhn_layer_{}'.format(layer),
            filter_size,
            num_hidden[layer],
            stride,
            tln=tln)
        rhn_layer.append(new_rhn)

    for layer in range(num_layers // 2, num_layers):
        new_rhn = HighwayCell(
            'rhn_layer_{}'.format(layer),
            filter_size,
            num_hidden[layer],
            stride,
            deconv=True,
            tln=tln)
        rhn_layer.append(new_rhn)

    with tf.variable_scope('generator'):
        net = None
        local_reuse = reuse

        for time_step in range(total_length - 1):

            inputs = images[:, time_step]

            with slim.arg_scope(nets.inception.inception_v1_arg_scope()):
                _, endpoints = nets.inception.inception_v1(
                    inputs, 41, is_training=is_training, reuse=local_reuse,
                    dropout_keep_prob=dropout_keep_prob)

            inputs = endpoints['Mixed_4f']
            abstract_features.append(inputs)

            with tf.variable_scope('g_rhn', reuse=local_reuse):
                inputs = tf.reshape(inputs, (8, 14, 14, 832))
                net = rhn_layer[0](inputs, net)
                for layer in range(1, num_layers):
                    net = rhn_layer[layer](None, net)

            if time_step < total_length - 2:
                x_gen = tf.layers.conv2d(net, 832, 1, 1, 'same',
                                             activation=tf.nn.sigmoid,
                                             name='g_reduce_conv', reuse=local_reuse)
                gen_images.append(x_gen)
            else:
                y = tf.layers.average_pooling2d(net, 7, 7)
                logits = tf.layers.conv2d(
                        y,
                        41, [2, 2], padding='valid', reuse=reuse)
                logits = tf.squeeze(logits, (1, 2))
            local_reuse = True

    gen_images = tf.stack(gen_images)
    abstract_features = tf.stack(abstract_features)
    # [batch_size, total_length, height, width, channels]
    # gen_images = tf.transpose(gen_images, [1, 0, 2, 3, 4])
    loss = tf.nn.l2_loss(gen_images - abstract_features[1:])

    return logits, loss
