import tensorflow as tf
from src.nn_utils.rnn_cell import SwitchableDropoutWrapper
from src.nn_utils.nn import linear,highway_network,softsel,get_logits
from src.nn_utils.general import get_last_state, exp_mask, add_reg_without_bias
from src.nn_utils.general import mask as normal_mask
from src.nn_utils.attention import self_align_attention, self_choose_attention
from tensorflow.contrib.rnn import RNNCell
from abc import ABCMeta, abstractmethod


def build_tree_structure(normal_data, op_lists, reduce_mats, method='dy_tree_lstm.v1', hn=None,
                         wd=0., is_train=None, keep_prob=1.,swap_memory=False, scope=None):
    """
    get shift reduce stacked mat from data and tree info
    :param normal_data: rank is 3 with shape [bs,sl,vec]
    :param op_lists: rank is 2 with shape [bs,ol], 1 for shift, 2 for reduce and 3 for padding
    :param reduce_mats: rank is 3 with shape [bs,ol,mc], indicate the reduce indices in stack matrix, -1 for padding
    :param method: 'concat' 'mean' 'merge' 'lstm'
    :param hn: hn for some func
    :param wd: weight decay
    :param is_train: 
    :param keep_prob: 
    :param swap_memory: use physical memory
    :param scope: 
    :return: [bs,ol,hn]
    """
    # todo: add new generate method
    method_class_list = [GeneBiLSTM, GeneBTTreeLSTM, GeneBTMerge, GeneDyTreeLSTMv0,
                         GeneDyTreeLSTMv1]
    with tf.variable_scope(scope or 'build_tree_structure', reuse=None):
        # tanspose
        op_lists = tf.transpose(op_lists, [1, 0])  # [ol,bs]
        reduce_mats = tf.transpose(reduce_mats, [1, 0, 2])  # [ol,bs,mc]

        # len parameters
        bs, sl, d = tf.shape(normal_data)[0], tf.shape(normal_data)[1], tf.shape(normal_data)[2]
        ol = tf.shape(op_lists)[0]
        mc = tf.shape(reduce_mats)[2]

        gene = None
        for gene_class in method_class_list:
            if gene_class.method_type == method:
                gene = gene_class(hn, keep_prob, is_train, wd)
                break
        assert gene is not None, 'no shift reduce method %s' % method

        hn = gene.update_tree_hn()

        # elems for scan
        elems_tensors = [op_lists, reduce_mats]

        # non-sequence
        batch_indices = tf.range(0, bs, dtype=tf.int32)  # bs
        batch_indices_mat = tf.tile(tf.expand_dims(batch_indices, 1), [1, mc])  # bs,mc
        data_extend = tf.concat(
            [normal_data, tf.zeros(shape=[bs, 1, d], dtype=tf.float32)], axis=1)  # pointer will be 'data_len+1' at last
        # scan variable init
        t_init = tf.constant(0, tf.int32)  # indicate the stack mat index
        data_pointer_init = tf.zeros([bs], tf.int32)  # indicate the stack which data should be shifted next
        stack_mat_init = tf.zeros([ol, bs, hn], tf.float32)
        scan_init = (t_init, data_pointer_init, stack_mat_init)

        def main_scan_body(iter_vars, elems_vars):
            # get tensors
            # # iter: 1.t 2. data_pointer 3. stack_mat
            t = iter_vars[0]
            data_pointer = iter_vars[1]
            stack_mat = iter_vars[2]  # ol,bs,d
            # # elems: 1.op_list 2.reduce mat
            op_list = elems_vars[0]  # bs
            reduce_mat = elems_vars[1]  # bs mc

            # for shift
            shift_data_coordinates = tf.stack([batch_indices, data_pointer], axis=1)  # bs,2
            data_for_shift = tf.gather_nd(data_extend, shift_data_coordinates)  # coord:[bs,2]  data: [bs,sl,d]->bs,d
            # # TODO: add processing for shifted data
            processed_shifted_data = gene.do_shift(data_for_shift)
            assert processed_shifted_data is not None
            # # mask shifted data for change un-shifted data into zero ==> need to add
            masked_shifted_data = tf.where(tf.equal(op_list, tf.ones_like(op_list, tf.int32)),
                                           processed_shifted_data, tf.zeros_like(processed_shifted_data))  # bs,d
            # # data_pointer update
            data_pointer = tf.where(tf.equal(op_list, tf.ones_like(op_list, tf.int32)),
                                    data_pointer + 1, data_pointer)

            # for reduce
            # # mask generation
            reduce_data_coordinates = tf.stack([reduce_mat, batch_indices_mat], axis=2)  # bs,mc,2
            data_for_reduce = tf.gather_nd(stack_mat, reduce_data_coordinates)  # bs,mc,d
            mask_for_reduce = tf.not_equal(reduce_mat,
                                           tf.ones_like(reduce_mat) * -1)  # (reduce_mats[t] != -1)  # [bs,mc]
            # TODO: add processing for reduced data
            processed_reduced_data = gene.do_reduce(data_for_reduce, mask_for_reduce)

            masked_reduced_data = tf.where(tf.equal(op_list, tf.ones_like(op_list, tf.int32) * 2),
                                           processed_reduced_data, tf.zeros_like(processed_reduced_data))  # bs,d
            sr_data = masked_shifted_data + masked_reduced_data  # bs,d

            # new update method for shift and reduce result
            sr_data = tf.scatter_nd(indices=[[t]],updates=[sr_data],shape=[ol,bs,hn])
            stack_mat = stack_mat + sr_data

            return t+1,data_pointer,stack_mat

        output = tf.scan(main_scan_body, elems_tensors,scan_init,
                         parallel_iterations=1, swap_memory=swap_memory)

        output_stack_mats = output[2] # ol,ol,bs,v
        output_stack_mat = tf.transpose(output_stack_mats[-1],[1,0,2]) # bs,ol,hn
        output_stack_mat = gene.fetch_output(output_stack_mat)
        if wd > 0:
            add_reg_without_bias()
        return output_stack_mat


class GeneTemplate(metaclass=ABCMeta):
    method_type = 'template'

    def __init__(self, method_type, hn, dropout, is_train, wd=0.):
        self.method_type = method_type
        self.hn = hn
        self.dropout = dropout
        self.is_train = is_train
        self.wd = wd

    def update_tree_hn(self):
        return self.hn

    def do_shift(self, data_for_shift):
        with tf.variable_scope('sr_%s' % self.method_type):
            shifted_value = tf.nn.relu(linear([data_for_shift], self.hn, True, 0., 'shift_linear', False,
                                              input_keep_prob= self.dropout, is_train=self.is_train))
            return shifted_value

    @abstractmethod
    def do_reduce(self, data_for_reduce, mask_for_reduce):
        pass

    def fetch_output(self, output):
        with tf.variable_scope('sr_%s' % self.method_type):
            return tf.identity(output)


class GeneBiLSTM(GeneTemplate):
    method_type = 'bi_lstm'

    def __init__(self, hn, dropout, is_train, wd=0.):
        super(GeneBiLSTM, self).__init__(GeneBiLSTM.method_type, hn, dropout, is_train, wd)
        with tf.variable_scope('sr_%s' % self.method_type):
            assert hn % 2 == 0
            self.cell_fw = tf.contrib.rnn.LSTMCell(hn / 2)  # tf.contrib.rnn.GRUCell(data.get_shape()[2])
            self.cell_bw = tf.contrib.rnn.LSTMCell(hn / 2)  # tf.contrib.rnn.GRUCell(data.get_shape()[2])
            self.cell_dp_fw = SwitchableDropoutWrapper(self.cell_fw, is_train, dropout)
            self.cell_dp_bw = SwitchableDropoutWrapper(self.cell_bw, is_train, dropout)

    def do_reduce(self, data_for_reduce, mask_for_reduce):
        with tf.variable_scope('sr_%s' % self.method_type):
            seq_len = tf.reduce_sum(tf.cast(mask_for_reduce, tf.int32), -1)
            (fw, bw), _ = tf.nn.bidirectional_dynamic_rnn(self.cell_dp_fw, self.cell_dp_bw,
                                                          data_for_reduce, seq_len,
                                                          dtype=tf.float32,
                                                          scope='shift_reduce_bilstm_loop')
            value = tf.concat([fw, bw], -1)
            processed_reduced_data = get_last_state(value, mask_for_reduce)
            return processed_reduced_data


class GeneBTMerge(GeneTemplate):
    method_type = 'bt.merge'

    def __init__(self, hn, dropout, is_train, wd=0.):
        super(GeneBTMerge, self).__init__(GeneBTMerge.method_type, hn, dropout, is_train, wd)
        with tf.variable_scope('sr_%s' % self.method_type):
            pass

    def do_reduce(self, data_for_reduce, mask_for_reduce):
        with tf.variable_scope('sr_%s' % self.method_type):
            data_for_reduce_re = tf.reshape(data_for_reduce, [-1, 2 * self.hn])
            reduced_value = tf.nn.relu(linear([data_for_reduce_re], self.hn, True, 0., 'reduce_linear', False,
                                              input_keep_prob=self.dropout, is_train=self.is_train))
            return reduced_value


class GeneBTTreeLSTM(GeneTemplate):
    method_type = 'bt.tree_lstm'

    def __init__(self, hn, dropout, is_train, wd=0.):
        super(GeneBTTreeLSTM, self).__init__(GeneBTTreeLSTM.method_type, hn, dropout, is_train, wd)
        with tf.variable_scope('sr_%s' % self.method_type):
            pass

    def update_tree_hn(self):
        return self.hn * 2

    def do_shift(self, data_for_shift):
        hn, dropout, is_train, wd = self.hn, self.dropout, self.is_train, self.wd
        with tf.variable_scope('sr_%s' % self.method_type):
            I = tf.nn.sigmoid(linear([data_for_shift], hn, True, 0., 'W_i_0',
                                     False, 0., dropout, is_train))

            O = tf.nn.sigmoid(linear([data_for_shift], hn, True, 0., 'W_o_0',
                                     False, 0., dropout, is_train))

            U = tf.nn.tanh(linear([data_for_shift], hn, True, 0., 'W_u_0',
                                  False, 0., dropout, is_train))

            C = I * U  # bs, hn

            H = O * tf.nn.tanh(C)  # bs, hn

            return tf.concat([H, C], -1) # bs, 2hn

    def do_reduce(self, data_for_reduce, mask_for_reduce):
        hn, dropout, is_train, wd = self.hn, self.dropout, self.is_train, self.wd
        with tf.variable_scope('sr_%s' % self.method_type):
            left_child_hid = data_for_reduce[:, 0, :hn]
            left_child_cell = data_for_reduce[:, 0, hn:]

            right_child_hid = data_for_reduce[:, 1, :hn]
            right_child_cell = data_for_reduce[:, 1, hn:]

            # LSTM update
            I = tf.nn.sigmoid(
                linear([left_child_hid], hn, False, 0., 'W_i_l',
                       False, 0., dropout, is_train) +
                linear([right_child_hid], hn, True, 0., 'W_i_r',
                       False, 0., dropout, is_train),
            )

            F_l = tf.nn.sigmoid(
                linear([left_child_hid], hn, False, 0., 'W_f_l_l',
                       False, 0., dropout, is_train) +
                linear([right_child_hid], hn, True, 0., 'W_f_l_r',
                       False, 0., dropout, is_train)
            )

            F_r = tf.nn.sigmoid(
                linear([left_child_hid], hn, False, 0., 'W_f_r_l',
                       False, 0., dropout, is_train) +
                linear([right_child_hid], hn, True, 0., 'W_f_r_r',
                       False, 0., dropout, is_train)
            )

            O = tf.nn.sigmoid(
                linear([left_child_hid], hn, False, 0., 'W_o_l',
                       False, 0., dropout, is_train) +
                linear([right_child_hid], hn, True, 0., 'W_o_r',
                       False, 0., dropout, is_train)
            )

            U = tf.nn.tanh(
                linear([left_child_hid], hn, False, 0., 'W_u_l',
                       False, 0., dropout, is_train) +
                linear([right_child_hid], hn, True, 0., 'W_u_r',
                       False, 0., dropout, is_train)
            )

            C = I * U + F_l * left_child_cell + F_r * right_child_cell
            H = O * tf.nn.tanh(C)
            return tf.concat([H, C], -1)

    def fetch_output(self, output):
        with tf.variable_scope('sr_%s' % self.method_type):
            return output[:, :, :self.hn]


class GeneDyTreeLSTMv0(GeneTemplate):
    method_type = 'dy_tree_lstm.v0'

    def __init__(self, hn, dropout, is_train, wd=0.):
        super(GeneDyTreeLSTMv0, self).__init__(GeneDyTreeLSTMv0.method_type, hn, dropout, is_train, wd)
        with tf.variable_scope('sr_%s' % self.method_type):
            pass

    def update_tree_hn(self):
        return self.hn * 2

    def do_shift(self, data_for_shift):
        hn, dropout, is_train, wd = self.hn, self.dropout, self.is_train, self.wd
        with tf.variable_scope('sr_%s' % self.method_type):
            I = tf.nn.sigmoid(linear([data_for_shift], hn, True, 0., 'W_i_0',
                                     False, 0., dropout, is_train))
            O = tf.nn.sigmoid(linear([data_for_shift], hn, True, 0., 'W_o_0',
                                     False, 0., dropout, is_train))
            U = tf.nn.tanh(linear([data_for_shift], hn, True, 0., 'W_u_0',
                                  False, 0., dropout, is_train))
            C = I * U  # bs, hn
            H = O * tf.nn.tanh(C)  # bs, hn
            return tf.concat([H, C], -1)  # bs, hn*2

    def do_reduce(self, data_for_reduce, mask_for_reduce):
        hn, dropout, is_train, wd = self.hn, self.dropout, self.is_train, self.wd
        mc = tf.shape(data_for_reduce)[1]
        with tf.variable_scope('sr_%s' % self.method_type):
            self_choose_attention(data_for_reduce, mask_for_reduce, hn, dropout, is_train, 'change_me')
            children_hid = data_for_reduce[:, :, :hn]
            children_cell = data_for_reduce[:, :, hn:]

            I = tf.nn.sigmoid(
                linear([self_choose_attention(children_hid, mask_for_reduce,
                                              hn, dropout, is_train, 'self_ch_i')],
                       hn, True, 0., 'linear_i', False, 0., dropout, is_train))

            # bs,mc,hn/ -> bs,1,mc,hn/2 -> bs,mc,mc,hn/2
            children_hid_tile_1 = tf.tile(tf.expand_dims(children_hid, 1), [1, mc, 1, 1])  #
            children_hid_tile_2 = tf.tile(tf.expand_dims(children_hid, 2), [1, 1, mc, 1])  #
            children_hid_tile = tf.concat([children_hid_tile_1, children_hid_tile_2], -1)  # bs,mc,mc,2* hn
            children_hid_tile_re = tf.reshape(children_hid_tile, [-1, mc, 2 * hn])  # bs*mc,mc,2* hn
            # # mask
            mask_tile_1 = tf.tile(tf.expand_dims(mask_for_reduce, 1), [1, mc, 1])
            mask_tile_2 = tf.tile(tf.expand_dims(mask_for_reduce, 2), [1, 1, mc])
            mask_tile = tf.logical_and(mask_tile_1, mask_tile_2)
            mask_tile_re = tf.reshape(mask_tile, [-1, mc])

            # bs*mc, 2* hn -linear-> bs*mc,hn -re-> bs,mc,hn
            F = tf.nn.sigmoid(tf.reshape(linear(
                [self_choose_attention(children_hid_tile_re, mask_tile_re,
                                       2 * hn, dropout, is_train, 'self_ch_f')],
                hn, True, 0., 'linear_f', False, 0., dropout, is_train), [-1, mc, hn]))

            O = tf.nn.sigmoid(
                linear([self_choose_attention(children_hid, mask_for_reduce,
                                              hn, dropout, is_train, 'self_ch_o')],
                       hn, True, 0., 'linear_o', False, 0., dropout, is_train))

            U = tf.nn.tanh(
                linear([self_choose_attention(children_hid, mask_for_reduce,
                                              hn, dropout, is_train, 'self_ch_u')],
                       hn, True, 0., 'linear_u', False, 0., dropout, is_train))

            # children_cell * F--[bs,mc,hn]   mask_for_reduce [bs,mc]->[bs,mc,1]
            C = I * U + tf.reduce_sum(
                normal_mask(children_cell * F,
                            tf.expand_dims(mask_for_reduce, -1)), 1
            )
            H = O * tf.nn.tanh(C)

            return tf.concat([H, C], -1)

    def fetch_output(self, output):
        with tf.variable_scope('sr_%s' % self.method_type):
            return output[:, :, :self.hn]


class GeneDyTreeLSTMv1(GeneTemplate):
    method_type = 'dy_tree_lstm.v1'

    def __init__(self, hn, dropout, is_train, wd=0.):
        super(GeneDyTreeLSTMv1, self).__init__(GeneDyTreeLSTMv1.method_type, hn, dropout, is_train, wd)
        with tf.variable_scope('sr_%s' % self.method_type):
            self.bias_I = tf.get_variable('bias_I', [self.hn], tf.float32, tf.constant_initializer(0.))
            self.bias_O = tf.get_variable('bias_O', [self.hn], tf.float32, tf.constant_initializer(0.))
            self.bias_U = tf.get_variable('bias_U', [self.hn], tf.float32, tf.constant_initializer(0.))

    def update_tree_hn(self):
        return self.hn * 2

    def do_shift(self, data_for_shift):
        hn, dropout, is_train, wd = self.hn, self.dropout, self.is_train, self.wd
        with tf.variable_scope('sr_%s' % self.method_type):
            print('var num in (2.1) :', len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))

            I = tf.nn.sigmoid(linear([data_for_shift], hn, False, 0., 'W_i_0',
                                     False, 0., dropout, is_train) + self.bias_I)
            O = tf.nn.sigmoid(linear([data_for_shift], hn, False, 0., 'W_o_0',
                                     False, 0., dropout, is_train) + self.bias_O)
            U = tf.nn.tanh(linear([data_for_shift], hn, False, 0., 'W_u_0',
                                  False, 0., dropout, is_train) + self.bias_U)
            print('var num in (2.2) :', len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))

            C = I * U  # bs, hn
            H = O * tf.nn.tanh(C)  # bs, hn
            return tf.concat([H, C], -1)  # bs, 2*hn

    def do_reduce(self, data_for_reduce, mask_for_reduce):
        hn, dropout, is_train, wd = self.hn, self.dropout, self.is_train, self.wd
        mc = tf.shape(data_for_reduce)[1]
        with tf.variable_scope('sr_%s' % self.method_type):
            print('var num in (2.3) :', len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
            # bs, mc, hn
            children_hid_un = data_for_reduce[:, :, :hn]
            children_cell = data_for_reduce[:, :, hn:]

            # bs, mc, hn
            children_hid = tf.concat([children_hid_un,
                                      self_align_attention(children_hid_un, mask_for_reduce),],
                                     -1)

            I = tf.nn.sigmoid(
                linear([self_choose_attention(children_hid, mask_for_reduce,
                                              hn, dropout, is_train, 'self_ch_i', True)],
                       hn, False, 0., 'linear_i', False, 0., dropout, is_train) + self.bias_I)

            print('var num in (2.4) :', len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))

            # bs*mc, 2* hn -linear-> bs*mc,hn  -re-> bs,mc,hn
            F = tf.nn.sigmoid(linear(
                [children_hid], hn, True, 0., 'linear_f', False, 0., dropout, is_train))

            print('var num in (2.5) :', len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))

            O = tf.nn.sigmoid(
                linear([self_choose_attention(children_hid, mask_for_reduce,
                                              hn, dropout, is_train, 'self_ch_o', True)],
                       hn, False, 0., 'linear_o', False, 0., dropout, is_train) + self.bias_O)

            print('var num in (2.6) :', len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))

            U = tf.nn.tanh(
                linear([self_choose_attention(children_hid, mask_for_reduce,
                                              hn, dropout, is_train, 'self_ch_u', True)],
                       hn, False, 0., 'linear_u', False, 0., dropout, is_train) + self.bias_U)

            print('var num in (2.7) :', len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))

            # children_cell * F--[bs,mc,hn]   mask_for_reduce [bs,mc]->[bs,mc,1]
            C = I * U + tf.reduce_sum(
                normal_mask(children_cell * F,
                            tf.expand_dims(mask_for_reduce, -1)), 1
            )
            H = O * tf.nn.tanh(C)

            return tf.concat([H, C], -1)

    def fetch_output(self, output):
        with tf.variable_scope('sr_%s' % self.method_type):
            return output[:, :, :self.hn]
