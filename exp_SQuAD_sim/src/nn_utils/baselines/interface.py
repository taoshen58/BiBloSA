"""
This is the baseline layers of context fusion layers and sentence-encoding models
"""
import tensorflow as tf
from src.nn_utils.baselines.recurrent_models import contextual_bi_rnn
from src.nn_utils.baselines.SRU import bi_sru_recurrent_network
from src.nn_utils.baselines.CNN import cnn_for_context_fusion, cnn_for_sentence_encoding
from src.nn_utils.baselines.multi_head_attention import multi_head_attention, multi_head_attention_git
from src.nn_utils.integration_func import directional_attention_with_dense
from src.nn_utils.baselines.block_attention import bi_directional_simple_block_attention

from src.nn_utils.integration_func import multi_dimensional_attention


def context_fusion_layers(
        rep_tensor, rep_mask, method, activation_function,
        scope=None, wd=0., is_train=None, keep_prob=1., hn=None, **kwargs):
    method_name_list = [
        'lstm', 'gru', 'sru', 'sru_normal',  # rnn
        'cnn',
        'multi_head', 'multi_head_git', 'disa',
        'block'
    ]
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]
    hn = hn or ivec
    context_fusion_output = None
    with tf.variable_scope(scope or 'context_fusion_layers'):
        if method in ['lstm', 'gru', 'sru_normal']:
            context_fusion_output = contextual_bi_rnn(
                rep_tensor, rep_mask, hn, method,
                False, wd, keep_prob, is_train, 'ct_bi_%s' % method)
        elif method == 'sru':
            context_fusion_output = bi_sru_recurrent_network(
                rep_tensor, rep_mask, is_train, keep_prob, wd, 'ct_bi_sru', hn)
        elif method == 'cnn':
            assert 2*hn % 4 == 0
            sub_hn = 2*hn // 4
            context_fusion_output = cnn_for_context_fusion(
                rep_tensor, rep_mask, (3, 4, 5, 6), sub_hn, 'ct_cnn', is_train, keep_prob, wd)
        elif method == 'multi_head':
            assert 2*hn % 4 == 0
            sub_hn = 2*hn // 4
            context_fusion_output = multi_head_attention(
                rep_tensor, rep_mask, 4, sub_hn, 'ct_multi_head', is_train, keep_prob, wd)
        elif method == 'multi_head_git':
            assert 2*hn % 4 == 0
            sub_hn = 2*hn // 4
            context_fusion_output = multi_head_attention_git(
                rep_tensor, rep_mask, 4, sub_hn, 'ct_multi_head', is_train, keep_prob, wd)
        elif method == 'disa':
            with tf.variable_scope('ct_disa'):
                disa_fw = directional_attention_with_dense(
                    rep_tensor, rep_mask,'forward', 'fw_disa',
                    keep_prob, is_train, wd, activation_function, hn=hn)
                disa_bw = directional_attention_with_dense(
                    rep_tensor, rep_mask, 'backward', 'bw_disa',
                    keep_prob, is_train, wd, activation_function, hn=hn)
                context_fusion_output = tf.concat([disa_fw, disa_bw], -1)
        elif method == 'block':
            if 'block_len' in kwargs.keys():
                block_len = kwargs['block_len']
            else:
                block_len = None
            if block_len is None:
                block_len = tf.cast(tf.ceil(tf.pow(tf.cast(2 * sl, tf.float32), 1.0 / 3)), tf.int32)
            context_fusion_output = bi_directional_simple_block_attention(
                rep_tensor, rep_mask, block_len, 'ct_block_attn',
                keep_prob, is_train, wd, activation_function, hn=hn)
        else:
            raise RuntimeError

        return context_fusion_output


def sentence_encoding_models(
        rep_tensor, rep_mask, method, activation_function,
        scope=None, wd=0., is_train=None, keep_prob=1., hn=None, **kwargs):
    method_name_list = [
        'cnn_kim',
        'no_ct',
        'lstm', 'gru', 'sru', 'sru_normal',  # rnn
        'cnn',
        'multi_head', 'multi_head_git', 'disa',
        'block'
    ]
    with tf.variable_scope(scope or 'sentence_encoding_models'):
        if method == 'cnn_kim':
            assert 2*hn % 4 == 0
            sub_hn = 2*hn // 4
            sent_coding = cnn_for_sentence_encoding(
                rep_tensor, rep_mask, (3,4,5,6), sub_hn, 'sent_encoding_cnn_kim', is_train, keep_prob, wd)
        else:
            ct_rep = None
            if method == 'no_ct':
                ct_rep = tf.identity(rep_tensor)
            else:
                ct_rep = context_fusion_layers(
                    rep_tensor, rep_mask, method, activation_function,
                    None, wd, is_train, keep_prob, hn, **kwargs)

            sent_coding = multi_dimensional_attention(
                ct_rep, rep_mask, 'multi_dim_attn_for_%s' % method,
                keep_prob, is_train, wd, activation_function)

        return sent_coding

