__author__ = 'yunbo'

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim import nets

from StrideHighwayCell import HighwayCell


def recurrent_highway_network(
        images,
        num_layers,
        num_hidden,
        filter_size,
        stride=1,
        total_length=20,
        input_length=10,
        tln=True,
        layer_to_extract=3):

    rhn_layer = []
    shape = images.get_shape().as_list()
    output_channels = shape[-1]

    # The num of layer should be even.
    assert num_layers % 2 == 0
    # The index of layer for extraction should be within the set of all possible choices.
    assert 0 <= layer_to_extract <= num_layers

    if not isinstance(filter_size, (list, tuple)):
        filter_size = [filter_size] * num_layers

    for layer in range(num_layers // 2):
        new_rhn = HighwayCell(
            'rhn_layer_{}'.format(layer),
            filter_size[layer],
            num_hidden[layer],
            stride,
            tln=tln)
        rhn_layer.append(new_rhn)

    for layer in range(num_layers // 2, num_layers):
        new_rhn = HighwayCell(
            'rhn_layer_{}'.format(layer),
            filter_size[layer],
            num_hidden[layer],
            stride,
            deconv=True,
            tln=tln)
        rhn_layer.append(new_rhn)

    with tf.variable_scope('generator'):
        net = None
        gen_images = []
        hidden_outputs = []
        for time_step in range(total_length - 1):
            with tf.variable_scope('g_rhn', reuse=tf.AUTO_REUSE):
                # if time_step < input_length:
                #     inputs = images[:, time_step]
                # else:
                #     inputs = x_gen
                inputs = images[:, time_step]

                for layer in range(0, num_layers):
                    net = rhn_layer[layer](inputs, net)
                    if layer == layer_to_extract:
                        hidden_outputs.append(net)
                    inputs = None

                x_gen = tf.layers.conv2d(net, output_channels, 1, 1, 'same',
                                         activation=tf.nn.sigmoid,
                                         name='g_reduce_conv')
                gen_images.append(x_gen)

    gen_images = tf.stack(gen_images, axis=1)
    hidden_outputs = tf.stack(hidden_outputs, axis=1)
    l2_loss = tf.reduce_mean((gen_images - images[:, 1:]) ** 2)
    return gen_images, hidden_outputs, l2_loss

