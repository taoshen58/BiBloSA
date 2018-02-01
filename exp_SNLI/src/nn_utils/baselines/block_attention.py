import tensorflow as tf

from src.nn_utils.general import exp_mask_for_high_rank, mask_for_high_rank
from src.nn_utils.integration_func import directional_attention_with_dense
from src.nn_utils.nn import bn_dense_layer, linear


def bi_directional_simple_block_attention(
        rep_tensor, rep_mask, block_len=5, scope=None,
        keep_prob=1., is_train=None, wd=0., activation='elu'):
    with tf.variable_scope(scope or 'bi_directional_simple_block_attn'):

        fw_attn_res = simple_block_attention(
            rep_tensor, rep_mask, block_len, "forward_attn", "forward",
            keep_prob, is_train, wd, activation)
        bw_attn_res = simple_block_attention(
            rep_tensor, rep_mask, block_len, "backward_attn", "backward",
            keep_prob, is_train, wd, activation)
        attn_res = tf.concat([fw_attn_res, bw_attn_res], -1)
        return attn_res


def simple_block_attention(
        rep_tensor, rep_mask, block_len=5, scope=None, direction=None,
        keep_prob=1., is_train=None, wd=0., activation='elu'):
    assert direction is not None

    def scaled_tanh(x, scale=5.):
        return scale * tf.nn.tanh(1. / scale * x)

    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]
    with tf.variable_scope(scope or 'block_simple'):
        # @1. split sequence
        with tf.variable_scope('split_seq'):
            block_num = tf.cast(tf.ceil(tf.divide(tf.cast(sl, tf.float32), tf.cast(block_len, tf.float32))), tf.int32)
            comp_len = block_num * block_len - sl

            rep_tensor_comp = tf.concat([rep_tensor, tf.zeros([bs, comp_len, ivec], tf.float32)], 1)
            rep_mask_comp = tf.concat([rep_mask, tf.cast(tf.zeros([bs, comp_len], tf.int32), tf.bool)], 1)

            rep_tensor_split = tf.reshape(rep_tensor_comp, [bs, block_num, block_len, ivec])  # bs,bn,bl,d
            rep_mask_split = tf.reshape(rep_mask_comp, [bs, block_num, block_len])  # bs,bn,bl

            # non-linear
            rep_map = bn_dense_layer(rep_tensor_split, ivec, True, 0., 'bn_dense_map', activation,
                                     False, wd, keep_prob, is_train)  # bs,bn,bl,vec
            rep_map_tile = tf.tile(tf.expand_dims(rep_map, 2), [1, 1, block_len, 1, 1])  # bs,bn,bl,bl,vec
            # rep_map_dp = dropout(rep_map, keep_prob, is_train)
            bn = block_num
            bl = block_len

        with tf.variable_scope('self_attention'):
            # @2.self-attention in block
            # mask generation
            sl_indices = tf.range(block_len, dtype=tf.int32)
            sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)
            if direction == 'forward':
                direct_mask = tf.greater(sl_row, sl_col)  # bl,bl
            else:
                direct_mask = tf.greater(sl_col, sl_row)  # bl,bl
            direct_mask_tile = tf.tile(
                tf.expand_dims(tf.expand_dims(direct_mask, 0), 0), [bs, bn, 1, 1])  # bs,bn,bl,bl
            rep_mask_tile_1 = tf.tile(tf.expand_dims(rep_mask_split, 2), [1, 1, bl, 1])  # bs,bn,bl,bl
            rep_mask_tile_2 = tf.tile(tf.expand_dims(rep_mask_split, 3), [1, 1, 1, bl])  # bs,bn,bl,bl
            rep_mask_tile = tf.logical_and(rep_mask_tile_1, rep_mask_tile_2)
            attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile, name='attn_mask')  # bs,bn,bl,bl

            # attention
            f_bias = tf.get_variable('f_bias', [ivec], tf.float32, tf.constant_initializer(0.))
            dependent_head = linear(
                rep_map, 2 * ivec, False, 0., 'linear_dependent_head', False, wd, keep_prob, is_train)  # bs,bn,bl,2vec
            dependent, head = tf.split(dependent_head, 2, 3)
            dependent_etd = tf.expand_dims(dependent, 2)  # bs,bn,1,bl,vec
            head_etd = tf.expand_dims(head, 3)  # bs,bn,bl,1,vec
            logits = scaled_tanh(dependent_etd + head_etd + f_bias, 5.0)  # bs,bn,bl,bl,vec
            logits_masked = exp_mask_for_high_rank(logits, attn_mask)
            attn_score = tf.nn.softmax(logits_masked, 3)  # bs,bn,bl,bl,vec
            attn_score = mask_for_high_rank(attn_score, attn_mask)  # bs,bn,bl,bl,vec
            self_attn_result = tf.reduce_sum(attn_score * rep_map_tile, 3)  # bs,bn,bl,vec

        with tf.variable_scope('source2token_self_attn'):
            inter_block_logits = bn_dense_layer(self_attn_result, ivec, True, 0., 'bn_dense_map', 'linear',
                                                False, wd, keep_prob, is_train)  # bs,bn,bl,vec
            inter_block_logits_masked = exp_mask_for_high_rank(inter_block_logits, rep_mask_split)  # bs,bn,bl,vec
            inter_block_soft = tf.nn.softmax(inter_block_logits_masked, 2)  # bs,bn,bl,vec
            inter_block_attn_output = tf.reduce_sum(self_attn_result * inter_block_soft, 2)  # bs,bn,vec

        with tf.variable_scope('self_attn_inter_block'):
            inter_block_attn_output_mask = tf.cast(tf.ones([bs, bn], tf.int32), tf.bool)
            block_ct_res = directional_attention_with_dense(
                inter_block_attn_output, inter_block_attn_output_mask, direction, 'disa',
                keep_prob, is_train, wd, activation
            )  # [bs,bn,vec]

            block_ct_res_tile = tf.tile(tf.expand_dims(block_ct_res, 2), [1, 1, bl, 1])#[bs,bn,vec]->[bs,bn,bl,vec]

        with tf.variable_scope('combination'):
            # input:1.rep_map[bs,bn,bl,vec]; 2.self_attn_result[bs,bn,bl,vec]; 3.rnn_res_tile[bs,bn,bl,vec]
            rep_tensor_with_ct = tf.concat([rep_map, self_attn_result, block_ct_res_tile], -1)  # [bs,bn,bl,3vec]
            new_context_and_gate = linear(rep_tensor_with_ct, 2 * ivec, True, 0., 'linear_new_context_and_gate',
                                          False, wd, keep_prob, is_train)  # [bs,bn,bl,2vec]
            new_context, gate = tf.split(new_context_and_gate, 2, 3)  # bs,bn,bl,vec
            if activation == "relu":
                new_context_act = tf.nn.relu(new_context)
            elif activation == "elu":
                new_context_act = tf.nn.elu(new_context)
            elif activation == "linear":
                new_context_act = tf.identity(new_context)
            else:
                raise RuntimeError
            gate_sig = tf.nn.sigmoid(gate)
            combination_res = gate_sig * new_context_act + (1 - gate_sig) * rep_map  # bs,bn,bl,vec

        with tf.variable_scope('restore_original_length'):
            combination_res_reshape = tf.reshape(combination_res, [bs, bn * bl, ivec])  # bs,bn*bl,vec
            output = combination_res_reshape[:, :sl, :]
            return output