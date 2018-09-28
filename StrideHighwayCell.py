import tensorflow as tf

from TensorLayerNorm import tensor_layer_norm


class HighwayCell():
    def __init__(self, layer_name, filter_size, num_features, stride,
                 deconv=False, tln=False, initializer=0.0001):
        """
        Args:
            layer_name: layer names for different convlstm layers.
            filter_size: int tuple thats the height and width of the filter
            num_features: int thats the depth of the cell 
        """
        self.layer_name = layer_name
        self.filter_size = filter_size
        self.num_features = num_features
        self.layer_norm = tln
        self.stride = stride
        self.deconv = deconv
        if initializer == -1:
            self.initializer = None
        else:
            self.initializer = tf.random_uniform_initializer(-initializer, initializer)

    def init_state(self, inputs, num_features):
        dims = inputs.get_shape().ndims
        if dims == 4:
            batch = inputs.get_shape()[0]
            height = inputs.get_shape()[1]
            width = inputs.get_shape()[2]
        else:
            raise ValueError('input tensor should be rank 4.')
        return tf.zeros([batch, height, width, num_features], dtype=tf.float32)

    def __call__(self, x, s):
        if s is None:
            s = self.init_state(x, self.num_features)
        with tf.variable_scope(self.layer_name):
            if self.deconv == False:
                s_concat = tf.layers.conv2d(
                    s, self.num_features * 4,
                    self.filter_size, self.stride, padding='same',
                    kernel_initializer=self.initializer,
                    name='state_to_state')
            else:
                s_concat = tf.layers.conv2d_transpose(
                    s, self.num_features * 4,
                    self.filter_size, self.stride, padding='same',
                    kernel_initializer=self.initializer,
                    name='state_to_state')
            if self.layer_norm:
                s_concat = tensor_layer_norm(s_concat, 'state_to_state_tln')
            new_s, h, t, c = tf.split(s_concat, 4, 3)
            if x != None:
                if self.deconv == False:
                    x_concat = tf.layers.conv2d(
                        x, self.num_features * 3,
                        self.filter_size, self.stride, padding='same',
                        kernel_initializer=self.initializer,
                        name='input_to_state')
                else:
                    x_concat = tf.layers.conv2d_transpose(
                        x, self.num_features * 3,
                        self.filter_size, self.stride, padding='same',
                        kernel_initializer=self.initializer,
                        name='input_to_state')
                if self.layer_norm:
                    x_concat = tensor_layer_norm(x_concat, 'input_to_state_tln')
                h_x, t_x, c_x = tf.split(x_concat, 3, 3)
                h = tf.add(h, h_x)
                t = tf.add(t, t_x)
                c = tf.add(c, c_x)
            h = tf.nn.tanh(h)
            t = tf.nn.sigmoid(t)
            c = tf.nn.sigmoid(c)
            new_s = h * t + tf.tanh(new_s) * c
            return new_s
