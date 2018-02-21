from context_fusion.general import get_last_state, exp_mask_for_high_rank, mask_for_high_rank
from context_fusion.nn import linear, get_logits, pooling_with_mask, softsel, feature_combination, dropout,\
    bn_dense_layer
import tensorflow as tf
import math
from context_fusion.general import add_reg_without_bias, exp_mask


def traditional_attention(rep_tensor, rep_mask, scope=None,
                          keep_prob=1., is_train=None, wd=0., activation='elu',
                          tensor_dict=None, name=None):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'traditional_attention'):
        rep_tensor_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                        False, wd, keep_prob, is_train)

        rep_tensor_logits = get_logits([rep_tensor_map], None, False, scope='self_attn_logits',
                                       mask=rep_mask, input_keep_prob=keep_prob, is_train=is_train)  # bs,sl
        attn_res = softsel(rep_tensor, rep_tensor_logits, rep_mask)  # bs,vec

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name] = tf.nn.softmax(rep_tensor_logits)

        return attn_res


def multi_dimensional_attention(rep_tensor, rep_mask, scope=None,
                                keep_prob=1., is_train=None, wd=0., activation='elu',
                                tensor_dict=None, name=None):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'multi_dimensional_attention'):
        map1 = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map1', activation,
                              False, wd, keep_prob, is_train)
        map2 = bn_dense_layer(map1, ivec, True, 0., 'bn_dense_map2', 'linear',
                              False, wd, keep_prob, is_train)
        map2_masked = exp_mask_for_high_rank(map2, rep_mask)

        soft = tf.nn.softmax(map2_masked, 1)  # bs,sl,vec
        attn_output = tf.reduce_sum(soft * rep_tensor, 1)  # bs, vec

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name] = soft

        return attn_output


def directional_attention_with_dense(
        rep_tensor, rep_mask, direction=None, scope=None,
        keep_prob=1., is_train=None, wd=0., activation='elu',
        tensor_dict=None, name=None, hn=None):

    def scaled_tanh(x, scale=5.):
        return scale * tf.nn.tanh(1./scale * x)

    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]
    ivec = hn or ivec
    with tf.variable_scope(scope or 'directional_attention_%s' % direction or 'diag'):
        # mask generation
        sl_indices = tf.range(sl, dtype=tf.int32)
        sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)
        if direction is None:
            direct_mask = tf.cast(tf.diag(- tf.ones([sl], tf.int32)) + 1, tf.bool)
        else:
            if direction == 'forward':
                direct_mask = tf.greater(sl_row, sl_col)
            else:
                direct_mask = tf.greater(sl_col, sl_row)
        direct_mask_tile = tf.tile(tf.expand_dims(direct_mask, 0), [bs, 1, 1])  # bs,sl,sl
        rep_mask_tile = tf.tile(tf.expand_dims(rep_mask, 1), [1, sl, 1])  # bs,sl,sl
        attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile)  # bs,sl,sl

        # non-linear
        rep_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                 False, wd, keep_prob, is_train)
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, sl, 1, 1])  # bs,sl,sl,vec
        rep_map_dp = dropout(rep_map, keep_prob, is_train)

        # attention
        with tf.variable_scope('attention'):  # bs,sl,sl,vec
            f_bias = tf.get_variable('f_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            dependent = linear(rep_map_dp, ivec, False, scope='linear_dependent')  # bs,sl,vec
            dependent_etd = tf.expand_dims(dependent, 1)  # bs,1,sl,vec
            head = linear(rep_map_dp, ivec, False, scope='linear_head') # bs,sl,vec
            head_etd = tf.expand_dims(head, 2)  # bs,sl,1,vec

            logits = scaled_tanh(dependent_etd + head_etd + f_bias, 5.0)  # bs,sl,sl,vec

            logits_masked = exp_mask_for_high_rank(logits, attn_mask)
            attn_score = tf.nn.softmax(logits_masked, 2)  # bs,sl,sl,vec
            attn_score = mask_for_high_rank(attn_score, attn_mask)

            attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)  # bs,sl,vec

        with tf.variable_scope('output'):
            o_bias = tf.get_variable('o_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            # input gate
            fusion_gate = tf.nn.sigmoid(
                linear(rep_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
                linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) +
                o_bias)
            output = fusion_gate * rep_map + (1-fusion_gate) * attn_result
            output = mask_for_high_rank(output, rep_mask)

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name + '_dependent'] = dependent
            tensor_dict[name + '_head'] = head
            tensor_dict[name] = attn_score
            tensor_dict[name + '_gate'] = fusion_gate
        return output


# ----------------- bi-blosan -----------
def bi_directional_simple_block_attention(
        rep_tensor, rep_mask, block_len=5, scope=None,
        keep_prob=1., is_train=None, wd=0., activation='elu', hn=None):
    with tf.variable_scope(scope or 'bi_directional_simple_block_attn'):

        fw_attn_res = simple_block_attention(
            rep_tensor, rep_mask, block_len, "forward_attn", "forward",
            keep_prob, is_train, wd, activation, hn)
        bw_attn_res = simple_block_attention(
            rep_tensor, rep_mask, block_len, "backward_attn", "backward",
            keep_prob, is_train, wd, activation, hn)
        attn_res = tf.concat([fw_attn_res, bw_attn_res], -1)
        return attn_res


def simple_block_attention(
        rep_tensor, rep_mask, block_len=5, scope=None, direction=None,
        keep_prob=1., is_train=None, wd=0., activation='elu', hn=None):
    assert direction is not None

    def scaled_tanh(x, scale=5.):
        return scale * tf.nn.tanh(1. / scale * x)

    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    org_ivec = rep_tensor.get_shape().as_list()[2]
    ivec = hn or org_ivec
    with tf.variable_scope(scope or 'block_simple'):
        # @1. split sequence
        with tf.variable_scope('split_seq'):
            block_num = tf.cast(tf.ceil(tf.divide(tf.cast(sl, tf.float32), tf.cast(block_len, tf.float32))), tf.int32)
            comp_len = block_num * block_len - sl

            rep_tensor_comp = tf.concat([rep_tensor, tf.zeros([bs, comp_len, org_ivec], tf.float32)], 1)
            rep_mask_comp = tf.concat([rep_mask, tf.cast(tf.zeros([bs, comp_len], tf.int32), tf.bool)], 1)

            rep_tensor_split = tf.reshape(rep_tensor_comp, [bs, block_num, block_len, org_ivec])  # bs,bn,bl,d
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


# multi-head attention
# https://github.com/Kyubyong/transformer/blob/master/modules.py#L167
def multi_head_attention_git(rep_tensor, rep_mask, num_heads=8, num_units=64,scope=None,
        is_train=None, keep_prob=1., wd=0.):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    causality = False
    with tf.variable_scope(scope or "multihead_attention"):
        # because of self-attention, queries and keys is equal to rep_tensor
        queries = rep_tensor
        keys = rep_tensor

        # Set the fall back option for num_units
        if num_units is None:  # hn
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = rep_mask  # tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # exp mask
        outputs = tf.where(key_masks, outputs, paddings)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = rep_mask # tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= tf.cast(query_masks, tf.float32)  # broadcasting. (N, T_q, C)

        # Dropouts
        # outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        outputs = dropout(outputs, keep_prob, is_train)

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        # outputs += queries

        # Normalize
        # outputs = normalize(outputs)  # (N, T_q, C)

    return outputs


def multi_head_attention(
        rep_tensor, rep_mask, head_num=8, hidden_units_num=64,scope=None,
        is_train=None, keep_prob=1., wd=0.):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]

    with tf.variable_scope(scope or 'multi_head_attention'):

        with tf.variable_scope('positional_encoding'):
            seq_idxs = tf.tile(tf.expand_dims(tf.range(sl), 1), [1, ivec])  # sl, ivec
            feature_idxs = tf.tile(tf.expand_dims(tf.range(ivec), 0), [sl, 1])  # sl, ivec
            pos_enc = tf.where(
                tf.equal(tf.mod(feature_idxs, 2), 0),
                tf.sin(tf.cast(seq_idxs, tf.float32) /
                       tf.pow(10000., 2.0 * tf.cast(feature_idxs, tf.float32) / (1.0 * ivec))),
                tf.cos(tf.cast(seq_idxs, tf.float32) /
                       tf.pow(10000., 2.0 * tf.cast(feature_idxs - 1, tf.float32) / (1.0 * ivec))),
            )
            rep_tensor_pos = mask_for_high_rank(rep_tensor + pos_enc, rep_mask)  # bs, sl, ivec


        with tf.variable_scope('multi_head_attention'):
            W = tf.get_variable('W', [3, head_num, ivec, hidden_units_num], tf.float32)
            rep_tile = tf.tile(
                tf.expand_dims(tf.expand_dims(rep_tensor_pos, 0), 0),
                [3, head_num, 1, 1, 1])  # 3,head_num,bs,sl,ivec
            rep_tile_reshape = tf.reshape(rep_tile, [3, head_num, bs * sl, ivec])  # head_num,bs*sl,ivec

            maps = tf.reshape( # 3,head_num,bs*sl,hn ->  3,head_num,bs,sl,hn
                tf.matmul(dropout(rep_tile_reshape, keep_prob, is_train), W),
                [3, head_num, bs, sl, hidden_units_num])
            Q_map, K_map, V_map = tf.split(maps, 3, 0)
            Q_map = tf.squeeze(Q_map, [0])  # head_num,bs,sl,hn
            K_map = tf.squeeze(K_map, [0])  # head_num,bs,sl,hn
            V_map = tf.squeeze(V_map, [0])  # head_num,bs,sl,hn

            # head_num,bs,sl,sl
            # similarity_mat = tf.reduce_sum(Q_map_tile * K_map_tile, -1) / math.sqrt(1. * hidden_units_num)
            similarity_mat = tf.matmul(
                Q_map, tf.transpose(K_map, [0,1,3,2])
            ) / math.sqrt(1. * hidden_units_num)

            # mask: bs,sl -> head_num,bs,sl
            multi_mask = tf.tile(tf.expand_dims(rep_mask, 0), [head_num, 1, 1])  # head_num,bs,sl
            multi_mask_tile_1 = tf.expand_dims(multi_mask, 2)  # head_num,bs,1,sl
            multi_mask_tile_2 = tf.expand_dims(multi_mask, 3)  # head_num,bs,sl,1
            multi_mask_tile = tf.logical_and(multi_mask_tile_1, multi_mask_tile_2)  # head_num,bs,sl,sl
            similarity_mat_masked = exp_mask(similarity_mat, multi_mask_tile)  # head_num,bs,sl,sl
            prob_dist = tf.nn.softmax(similarity_mat_masked)  # head_num,bs,sl,sl
            prob_dist_dp = dropout(prob_dist, keep_prob, is_train)

            attn_res = tf.matmul(prob_dist_dp, V_map)  # head_num,bs,sl,hn

            attn_res_tran = tf.transpose(attn_res, [1,2,0,3])
            output = tf.reshape(attn_res_tran, [bs, sl, head_num * hidden_units_num])

            if wd > 0.:
                add_reg_without_bias()

            return output



