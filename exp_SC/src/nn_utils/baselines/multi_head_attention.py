import tensorflow as tf
import math
from src.nn_utils.general import exp_mask, mask_for_high_rank
from src.nn_utils.nn import dropout, add_reg_without_bias


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

            # Q_map = tf.transpose(  # head_num,bs,sl,hn -> bs,sl,head,hn
            #     tf.reshape(tf.matmul(rep_tile_reshape, W_Q), [head_num, bs, sl, hidden_units_num]), [1, 2, 0, 3])
            # K_map = tf.transpose(  # head_num,bs,sl,hn -> bs,sl,head,hn
            #     tf.reshape(tf.matmul(rep_tile_reshape, W_K), [head_num, bs, sl, hidden_units_num]), [1, 2, 0, 3])
            # V_map = tf.transpose(  # head_num,bs,sl,hn -> bs,sl,head,hn
            #     tf.reshape(tf.matmul(rep_tile_reshape, W_V), [head_num, bs, sl, hidden_units_num]), [1, 2, 0, 3])


def generate_positional_encoding(rep_tensor, name=None):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2] or vec
    with tf.name_scope(name or 'generate_positional_encoding'):
        pass



