"""
Author: Anonymity
Github: https://github.com/Anonymity
Email: Anonymity@Anonymity
Tensorflow implementation for Hierarchical CNN with "gated linear units (GLU)"[1] and residual connection[2]
[1] Dauphin, Yann N., et al. "Language modeling with gated convolutional networks." arXiv preprint arXiv:1612.08083 (2016).
[2] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
"""



import tensorflow as tf
from context_fusion.nn import dropout, add_reg_without_bias
from context_fusion.general import mask_for_high_rank


def hierarchical_cnn_res_gate(
        rep_tensor, rep_mask, n_gram=5, layer_num=5, hn=None, scope=None,
        is_train=None, keep_prob=1., wd=0.):
    # padding
    if n_gram % 2 == 1:
        padding_front = padding_back = int((n_gram - 1) / 2)
    else:
        padding_front = (n_gram - 1) // 2
        padding_back = padding_front + 1
    padding = [[0, 0], [padding_front, padding_back], [0, 0], [0, 0]]

    # lengths
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    org_ivec = rep_tensor.get_shape().as_list()[2]
    ivec = hn or org_ivec

    with tf.variable_scope(scope or 'cnn_for_sentence_encoding'):
        rep_tensor = mask_for_high_rank(rep_tensor, rep_mask)  # bs, sl, hn

        iter_rep = rep_tensor
        layer_res_list = []

        for layer_idx in range(layer_num):
            with tf.variable_scope("conv_maxpool_%s" % layer_idx):

                iter_rep_etd = tf.expand_dims(iter_rep, 3)  # bs,sl,hn,1
                iter_rep_etd_dp = dropout(iter_rep_etd, keep_prob, is_train)
                # Convolution Layer
                feature_size = org_ivec if layer_idx == 0 else ivec
                filter_shape = [n_gram, feature_size, 1, 2 * ivec]
                W = tf.get_variable('W', filter_shape, tf.float32)
                b = tf.get_variable('b', [2 * ivec], tf.float32)
                iter_rep_etd_pad = tf.pad(iter_rep_etd_dp, padding)
                conv = tf.nn.conv2d(
                    iter_rep_etd_pad,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                map_res = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  # bs,sl,1,2hn
                map_res = tf.squeeze(map_res, [2])  # bs,sl,2*hn
                # gate
                map_res_a, map_res_b = tf.split(map_res, num_or_size_splits=2, axis=2)
                iter_rep = map_res_a * tf.nn.sigmoid(map_res_b)

                # res
                if len(layer_res_list) > 0:
                    iter_rep = iter_rep + layer_res_list[-1]
                layer_res_list.append(iter_rep)

        if wd > 0.:
            add_reg_without_bias()
        return iter_rep





