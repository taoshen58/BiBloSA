from configs import cfg
from src.utils.record_log import _logger
import tensorflow as tf
from src.model.model_template import ModelTemplate
from src.nn_utils.nn import exp_mask, get_logits, softsel, bn_dense_layer
from src.nn_utils.integration_func import generate_embedding_mat, multi_dimensional_attention
from src.nn_utils.baselines.interface import context_fusion_layers, sentence_encoding_models


class ModelContextFusion(ModelTemplate):
    def __init__(self, token_emb_mat, glove_emb_mat, tds, tel, hn, scope):
        super(ModelContextFusion, self).__init__(token_emb_mat, glove_emb_mat, tds, tel, hn, scope)
        self.update_tensor_add_ema_and_opt()

    def build_network(self):
        tds, tel, hn = self.tds, self.tel, self.hn
        bs, sn, sl, ql = self.bs, self.sn, self.sl, self.ql

        with tf.variable_scope('emb'):
            token_emb_mat = generate_embedding_mat(
                tds, tel, init_mat=self.token_emb_mat, extra_mat=self.glove_emb_mat, scope='gene_token_emb_mat')
            c_emb = tf.nn.embedding_lookup(token_emb_mat, self.context_token)  # bs,sn,sl,tel
            q_emb = tf.nn.embedding_lookup(token_emb_mat, self.question_token)  # s,ql,tel

        with tf.variable_scope('prepro'):
            q_rep = multi_dimensional_attention(
                q_emb, self.question_token_mask, 'q2coding',cfg.dropout, self.is_train, cfg.wd, 'relu')  # bs, hn
            q_rep_map = bn_dense_layer(
                q_rep, hn, True, 0., 'q_rep_map', 'relu', False, cfg.wd, cfg.dropout, self.is_train)  # bs, hn

        with tf.variable_scope('sent_emb'):
            c_emb_rshp = tf.reshape(c_emb, [bs*sn, sl, tel], 'c_emb_rshp')  # bs*sn,sl,tel
            c_mask_rshp = tf.reshape(self.context_token_mask, [bs*sn, sl], 'c_mask_rshp')  # bs*sn,sl,tel
            sent_enc_rshp = sentence_encoding_models(
                c_emb_rshp, c_mask_rshp, cfg.context_fusion_method, 'relu', 'sent2enc', cfg.wd,
                self.is_train, cfg.dropout, hn, block_len=cfg.block_len)  # bs*sn, 2*hn
            sent_enc = tf.reshape(sent_enc_rshp, [bs, sn, 2*hn])  # bs,sn, 2*hn
            sent_enc_map = bn_dense_layer(
                sent_enc, hn, True, 0., 'sent_enc_map', 'relu', False, cfg.wd, cfg.dropout, self.is_train)

        with tf.variable_scope('fusion'):
            q_rep_map_ex = tf.tile(tf.expand_dims(q_rep_map, 1), [1, sn, 1]) # bs, sn, hn
            fusion_rep = tf.concat(
                [sent_enc_map, q_rep_map_ex, sent_enc_map - q_rep_map_ex, sent_enc_map * q_rep_map_ex], -1)  # bs,sn,4hn

        with tf.variable_scope('output'):
            out_cf = context_fusion_layers(
                fusion_rep, self.context_sent_mask, cfg.context_fusion_method, 'relu', 'out_cf', cfg.wd,
                self.is_train, cfg.dropout, hn, block_len=4)
            pre_output = bn_dense_layer(
                out_cf, hn, True, 0., 'pre_output', 'relu', False, cfg.wd, cfg.dropout, self.is_train)

        logits = get_logits(  # exp masked
            pre_output, None, True, 0., 'logits', self.context_sent_mask, cfg.wd, cfg.dropout, self.is_train, 'linear')
        return logits














