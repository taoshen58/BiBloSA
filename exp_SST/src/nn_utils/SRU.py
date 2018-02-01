"""
Author: Anonymity
Github: https://github.com/Anonymity
Email: Anonymity@Anonymity
"""


import tensorflow as tf
from src.nn_utils.nn import bn_dense_layer
from src.nn_utils.rnn import dynamic_rnn, bw_dynamic_rnn
from src.nn_utils.rnn_cell import SwitchableDropoutWrapper


def bi_sru_recurrent_network(
        rep_tensor, rep_mask, is_train=None, keep_prob=1., wd=0.,
        scope=None):
    """

    :param rep_tensor: [Tensor/tf.float32] rank is 3 with shape [batch_size/bs, max_sent_len/sl, vec]
    :param rep_mask: [Tensor/tf.bool]rank is 2 with shape [bs,sl]
    :param is_train: [Scalar Tensor/tf.bool]scalar tensor to indicate whether the mode is training or not
    :param keep_prob: [float] dropout keep probability in the range of (0,1)
    :param wd: [float]for L2 regularization, if !=0, add tensors to tf collection "reg_vars"
    :param scope: [str]variable scope name
    :return: [Tensor/tf.float32] with shape [bs, sl, 2vec] for forward and backward
    """
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]
    with tf.variable_scope(scope or 'bi_sru_recurrent_network'):
        U_d = bn_dense_layer([rep_tensor], 6 * ivec, False, 0., 'get_frc', 'linear',
                           False, wd, keep_prob, is_train)  # bs, sl, 6vec
        U_d_fw, U_d_bw = tf.split(U_d, 2, 2)
        with tf.variable_scope('forward'):
            U_fw = tf.concat([rep_tensor, U_d_fw], -1)
            fw_SRUCell = SwitchableDropoutWrapper(SRUCell(ivec, tf.nn.tanh), is_train, keep_prob)
            fw_output, _ = dynamic_rnn(
                fw_SRUCell, U_fw, tf.reduce_sum(tf.cast(rep_mask, tf.int32), -1),
                dtype=tf.float32, scope='forward_sru')  # bs, sl, vec

        with tf.variable_scope('backward'):
            U_bw = tf.concat([rep_tensor, U_d_bw], -1)
            bw_SRUCell = SwitchableDropoutWrapper(SRUCell(ivec, tf.nn.tanh), is_train, keep_prob)
            bw_output, _ = bw_dynamic_rnn(
                bw_SRUCell, U_bw, tf.reduce_sum(tf.cast(rep_mask, tf.int32), -1),
                dtype=tf.float32, scope='backward_sru')  # bs, sl, vec

        all_output = tf.concat([fw_output, bw_output], -1)  # bs, sl, 2vec
        return all_output


class SRUCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, activation=None, reuse=None):
        super(SRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or tf.nn.tanh

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """

        :param inputs: [bs,4*vec]
        :param state: [bs, vec]
        :return:
        """
        b_f = tf.get_variable('b_f', [self._num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(0))
        b_r = tf.get_variable('b_r', [self._num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(0))

        x_t, x_dt, f_t, r_t = tf.split(inputs, 4, 1)
        f_t = tf.nn.sigmoid(f_t + b_f)
        r_t = tf.nn.sigmoid(r_t + b_r)
        c_t = f_t * state + (1 - f_t) * x_dt
        h_t = r_t * self._activation(c_t) + (1 - r_t) * x_t
        return h_t, c_t


class NormalSRUCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, activation=None, reuse=None):
        super(NormalSRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or tf.nn.tanh

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """

        :param inputs: [bs, vec]
        :param state:
        :param scope:
        :return:
        """
        with tf.variable_scope(scope or "SRU_cell"):
            b_f = tf.get_variable('b_f', [self._num_units], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0))
            b_r = tf.get_variable('b_r', [self._num_units], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0))
            U_d = bn_dense_layer(inputs, 3 * self._num_units, False, 0., 'get_frc', 'linear')  # bs, 3vec
            x_t = tf.identity(inputs, 'x_t')
            x_dt, f_t, r_t = tf.split(U_d, 3, 1)
            f_t = tf.nn.sigmoid(f_t + b_f)
            r_t = tf.nn.sigmoid(r_t + b_r)
            c_t = f_t * state + (1 - f_t) * x_dt
            h_t = r_t * self._activation(c_t) + (1 - r_t) * x_t
            return h_t, c_t



