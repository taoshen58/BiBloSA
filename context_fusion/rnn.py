import tensorflow as tf
from context_fusion.general import flatten, reconstruct
from context_fusion.nn import bn_dense_layer
from tensorflow.contrib.rnn import DropoutWrapper
from context_fusion.general import get_last_state, add_reg_without_bias


def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):
    assert not time_major  # TODO : to be implemented later!
    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    flat_outputs, final_state = tf.nn.dynamic_rnn(cell, flat_inputs, sequence_length=flat_len,
                                             initial_state=initial_state, dtype=dtype,
                                             parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                             time_major=time_major, scope=scope)

    outputs = reconstruct(flat_outputs, inputs, 2)
    return outputs, final_state


def bw_dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                   dtype=None, parallel_iterations=None, swap_memory=False,
                   time_major=False, scope=None):
    assert not time_major  # TODO : to be implemented later!

    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    flat_inputs = tf.reverse(flat_inputs, [1]) if sequence_length is None \
        else tf.reverse_sequence(flat_inputs, sequence_length, 1)
    flat_outputs, final_state = tf.nn.dynamic_rnn(cell, flat_inputs, sequence_length=flat_len,
                                             initial_state=initial_state, dtype=dtype,
                                             parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                             time_major=time_major, scope=scope)
    flat_outputs = tf.reverse(flat_outputs, [1]) if sequence_length is None \
        else tf.reverse_sequence(flat_outputs, sequence_length, 1)

    outputs = reconstruct(flat_outputs, inputs, 2)
    return outputs, final_state


def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):
    assert not time_major

    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    (flat_fw_outputs, flat_bw_outputs), final_state = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, flat_inputs, sequence_length=flat_len,
                                   initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw,
                                   dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                   time_major=time_major, scope=scope)

    fw_outputs = reconstruct(flat_fw_outputs, inputs, 2)
    bw_outputs = reconstruct(flat_bw_outputs, inputs, 2)
    # FIXME : final state is not reshaped!
    return (fw_outputs, bw_outputs), final_state


# ---------------- RNN Cell ----------------
class SwitchableDropoutWrapper(DropoutWrapper):
    def __init__(self, cell, is_train, input_keep_prob=1.0, output_keep_prob=1.0,
                 seed=None):
        super(SwitchableDropoutWrapper, self).__init__(cell,
                                                       input_keep_prob=input_keep_prob,
                                                       output_keep_prob=output_keep_prob,
                                                       seed=seed)
        self.is_train = is_train

    def __call__(self, inputs, state, scope=None):
        outputs_do, new_state_do = super(SwitchableDropoutWrapper, self).__call__(inputs, state, scope=scope)
        tf.get_variable_scope().reuse_variables()
        outputs, new_state = self._cell(inputs, state, scope)
        outputs = tf.cond(self.is_train, lambda: outputs_do, lambda: outputs)
        if isinstance(state, tf.contrib.rnn.LSTMStateTuple):
            new_state = state.__class__(*[tf.cond(self.is_train, lambda: new_state_do_i, lambda: new_state_i)
                                          for new_state_do_i, new_state_i in zip(new_state_do, new_state)])
        elif isinstance(state, tuple):
            new_state = state.__class__([tf.cond(self.is_train, lambda: new_state_do_i, lambda: new_state_i)
                                         for new_state_do_i, new_state_i in zip(new_state_do, new_state)])
        else:
            new_state = tf.cond(self.is_train, lambda: new_state_do, lambda: new_state)
        return outputs, new_state


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


# ---------- accelerated SRU --------------
def bi_sru_recurrent_network(
        rep_tensor, rep_mask, is_train=None, keep_prob=1., wd=0.,
        scope=None, hn=None, reuse=None):
    """

    :param rep_tensor: [Tensor/tf.float32] rank is 3 with shape [batch_size/bs, max_sent_len/sl, vec]
    :param rep_mask: [Tensor/tf.bool]rank is 2 with shape [bs,sl]
    :param is_train: [Scalar Tensor/tf.bool]scalar tensor to indicate whether the mode is training or not
    :param keep_prob: [float] dropout keep probability in the range of (0,1)
    :param wd: [float]for L2 regularization, if !=0, add tensors to tf collection "reg_vars"
    :param scope: [str]variable scope name
    :param hn:
    :param
    :return: [Tensor/tf.float32] with shape [bs, sl, 2vec] for forward and backward
    """
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]
    ivec = hn or ivec

    with tf.variable_scope(scope or 'bi_sru_recurrent_network'):
        # U_d = bn_dense_layer([rep_tensor], 6 * ivec, False, 0., 'get_frc', 'linear',
        #                    False, wd, keep_prob, is_train)  # bs, sl, 6vec
        # U_d_fw, U_d_bw = tf.split(U_d, 2, 2)
        with tf.variable_scope('forward'):
            U_d_fw = bn_dense_layer([rep_tensor], 3 * ivec, False, 0., 'get_frc_fw', 'linear',
                                    False, wd, keep_prob, is_train)  # bs, sl, 6vec
            U_fw = tf.concat([rep_tensor, U_d_fw], -1)
            fw_SRUCell = SwitchableDropoutWrapper(SRUCell(ivec, tf.nn.tanh, reuse), is_train, keep_prob)
            fw_output, _ = dynamic_rnn(
                fw_SRUCell, U_fw, tf.reduce_sum(tf.cast(rep_mask, tf.int32), -1),
                dtype=tf.float32, scope='forward_sru')  # bs, sl, vec

        with tf.variable_scope('backward'):
            U_d_bw = bn_dense_layer([rep_tensor], 3 * ivec, False, 0., 'get_frc_bw', 'linear',
                                    False, wd, keep_prob, is_train)  # bs, sl, 6vec
            U_bw = tf.concat([rep_tensor, U_d_bw], -1)
            bw_SRUCell = SwitchableDropoutWrapper(SRUCell(ivec, tf.nn.tanh, reuse), is_train, keep_prob)
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

# ----------------- END --------------------


# ----------------- RNN integration -------
def contextual_bi_rnn(tensor_rep, mask_rep, hn, cell_type, only_final=False,
                      wd=0., keep_prob=1.,is_train=None, scope=None):
    """
    fusing contextual information using bi-direction rnn
    :param tensor_rep: [..., sl, vec]
    :param mask_rep: [..., sl]
    :param hn:
    :param cell_type: 'gru', 'lstm', basic_lstm' and 'basic_rnn'
    :param only_final: True or False
    :param wd:
    :param keep_prob:
    :param is_train:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope or 'contextual_bi_rnn'): # correct
        reuse = None if not tf.get_variable_scope().reuse else True

        if cell_type == 'sru':
            rnn_outputs = bi_sru_recurrent_network(
                tensor_rep, mask_rep, is_train, keep_prob, wd, 'bi_sru_recurrent_network', hn, reuse)
        else:
            if cell_type == 'gru':
                cell_fw = tf.contrib.rnn.GRUCell(hn, reuse=reuse)
                cell_bw = tf.contrib.rnn.GRUCell(hn, reuse=reuse)
            elif cell_type == 'lstm':
                cell_fw = tf.contrib.rnn.LSTMCell(hn, reuse=reuse)
                cell_bw = tf.contrib.rnn.LSTMCell(hn, reuse=reuse)
            elif cell_type == 'basic_lstm':
                cell_fw = tf.contrib.rnn.BasicLSTMCell(hn, reuse=reuse)
                cell_bw = tf.contrib.rnn.BasicLSTMCell(hn, reuse=reuse)
            elif cell_type == 'basic_rnn':
                cell_fw = tf.contrib.rnn.BasicRNNCell(hn, reuse=reuse)
                cell_bw = tf.contrib.rnn.BasicRNNCell(hn, reuse=reuse)
            elif cell_type == 'sru_normal':
                cell_fw = NormalSRUCell(hn, reuse=reuse)
                cell_bw = NormalSRUCell(hn, reuse=reuse)
            else:
                raise AttributeError('no cell type \'%s\'' % cell_type)
            cell_dp_fw = SwitchableDropoutWrapper(cell_fw,is_train,keep_prob)
            cell_dp_bw = SwitchableDropoutWrapper(cell_bw,is_train,keep_prob)

            tensor_len = tf.reduce_sum(tf.cast(mask_rep, tf.int32), -1)  # [bs]

            (outputs_fw, output_bw), _=bidirectional_dynamic_rnn(
                cell_dp_fw, cell_dp_bw, tensor_rep, tensor_len,
                dtype=tf.float32)
            rnn_outputs = tf.concat([outputs_fw,output_bw],-1)  # [...,sl,2hn]

        if wd > 0:
            add_reg_without_bias()
        if not only_final:
            return rnn_outputs  # [....,sl, 2hn]
        else:
            return get_last_state(rnn_outputs, mask_rep)  # [...., 2hn]






