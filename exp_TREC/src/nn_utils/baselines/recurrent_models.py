from src.nn_utils.rnn_cell import SwitchableDropoutWrapper
from src.nn_utils.rnn import bidirectional_dynamic_rnn
import tensorflow as tf
from src.nn_utils.general import get_last_state, add_reg_without_bias
from src.nn_utils.baselines.SRU import NormalSRUCell


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
        #print(reuse)
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