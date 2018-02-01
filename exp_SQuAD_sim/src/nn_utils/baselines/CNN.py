"""
@Author: Anonymity
Tensorflow implementation for CNN in sentence encoding
"""

import tensorflow as tf
from src.nn_utils.nn import dropout, add_reg_without_bias
from src.nn_utils.general import mask_for_high_rank


def cnn_for_context_fusion(
        rep_tensor, rep_mask, filter_sizes=(3,4,5), num_filters=200, scope=None,
        is_train=None, keep_prob=1., wd=0.):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]

    with tf.variable_scope(scope or 'cnn_for_sentence_encoding'):
        rep_tensor = mask_for_high_rank(rep_tensor, rep_mask)
        rep_tensor_expand = tf.expand_dims(rep_tensor, 3)  # bs, sl,
        rep_tensor_expand_dp = dropout(rep_tensor_expand, keep_prob, is_train)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, ivec, 1, num_filters]
                W = tf.get_variable('W', filter_shape, tf.float32)
                b = tf.get_variable('b', [num_filters], tf.float32)

                # # pading in the sequence
                if filter_size % 2 == 1:
                    padding_front = padding_back = int((filter_size - 1) / 2)
                else:
                    padding_front = (filter_size - 1) // 2
                    padding_back = padding_front + 1
                padding = [[0, 0], [padding_front, padding_back], [0, 0], [0, 0]]
                rep_tensor_expand_dp_pad = tf.pad(rep_tensor_expand_dp, padding)

                conv = tf.nn.conv2d(
                    rep_tensor_expand_dp_pad,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  # bs, sl, 1, fn
                h_squeeze = tf.squeeze(h, [2])  # bs, sl, fn
                pooled_outputs.append(h_squeeze)

        # Combine all the pooled features
        result = tf.concat(pooled_outputs, 2)  # bs, sl, 3 * fn

        if wd > 0.:
            add_reg_without_bias()

        return result


def cnn_for_sentence_encoding(
        rep_tensor, rep_mask, filter_sizes=(3,4,5), num_filters=200, scope=None,
        is_train=None, keep_prob=1., wd=0.):
    """

    :param rep_tensor:
    :param rep_mask:
    :param filter_sizes:
    :param num_filters:
    :param scope:
    :param is_train:
    :param keep_prob:
    :param wd:
    :return:
    """
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]

    with tf.variable_scope(scope or 'cnn_for_sentence_encoding'):
        rep_tensor = mask_for_high_rank(rep_tensor, rep_mask)
        rep_tensor_expand = tf.expand_dims(rep_tensor, 3)
        rep_tensor_expand_dp = dropout(rep_tensor_expand, keep_prob, is_train)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, ivec, 1, num_filters]
                W = tf.get_variable('W', filter_shape, tf.float32)
                b = tf.get_variable('b', [num_filters], tf.float32)

                conv = tf.nn.conv2d(
                    rep_tensor_expand_dp,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  # bs, sl-fs+1, 1, fn
                # Maxpooling over the outputs
                # pooled = tf.nn.max_pool(
                #     h,
                #     ksize=[1, sl - filter_size + 1, 1, 1],
                #     strides=[1, 1, 1, 1],
                #     padding='VALID',
                #     name="pool")
                pooled = tf.reduce_max(h, 1, True)  # bs, 1, 1, fn
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        if wd > 0.:
            add_reg_without_bias()

        return h_pool_flat












