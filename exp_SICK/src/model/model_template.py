from configs import cfg
from src.utils.record_log import _logger
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod


class ModelTemplate(metaclass=ABCMeta):
    def __init__(self, token_emb_mat, glove_emb_mat, tds, cds, tl, scope):
        self.scope = scope
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)


        self.token_emb_mat, self.glove_emb_mat = token_emb_mat, glove_emb_mat

        # ---- place holder -----
        self.sent1_token = tf.placeholder(tf.int32, [None, None], name='sent1_token')
        self.sent1_char = tf.placeholder(tf.int32, [None, None, tl], name='sent1_char')

        self.sent2_token = tf.placeholder(tf.int32, [None, None], name='sent2_token')
        self.sent2_char = tf.placeholder(tf.int32, [None, None, tl], name='sent2_char')

        self.target_distribution = tf.placeholder(tf.float32, [None, 5], name='target_distribution')
        self.gold_label = tf.placeholder(tf.float32, [None], name='gold_label')
        self.is_train = tf.placeholder(tf.bool, [], name='is_train')

        # ----------- parameters -------------
        self.tds, self.cds = tds, cds
        self.tl = tl
        self.tel = cfg.word_embedding_length
        self.cel = cfg.char_embedding_length
        self.cos = cfg.char_out_size
        self.ocd = list(map(int, cfg.out_channel_dims.split(',')))
        self.fh = list(map(int, cfg.filter_heights.split(',')))
        self.hn = cfg.hidden_units_num
        self.finetune_emb = cfg.fine_tune

        self.output_class = 5  # 0 for contradiction, 1 for neural and 2 for entailment
        self.bs = tf.shape(self.sent1_token)[0]
        self.sl1 = tf.shape(self.sent1_token)[1]
        self.sl2 = tf.shape(self.sent2_token)[1]

        # ------------ other ---------
        self.sent1_token_mask = tf.cast(self.sent1_token, tf.bool)
        self.sent1_char_mask = tf.cast(self.sent1_char, tf.bool)
        self.sent1_token_len = tf.reduce_sum(tf.cast(self.sent1_token_mask, tf.int32), -1)
        self.sent1_char_len = tf.reduce_sum(tf.cast(self.sent1_char_mask, tf.int32), -1)

        self.sent2_token_mask = tf.cast(self.sent2_token, tf.bool)
        self.sent2_char_mask = tf.cast(self.sent2_char, tf.bool)
        self.sent2_token_len = tf.reduce_sum(tf.cast(self.sent2_token_mask, tf.int32), -1)
        self.sent2_char_len = tf.reduce_sum(tf.cast(self.sent2_char_mask, tf.int32), -1)

        self.tensor_dict = {}

        # ------ start ------
        self.logits = None
        self.loss = None

        self.mse = None
        self.predicted_score = None

        self.var_ema = None
        self.ema = None
        self.summary = None
        self.opt = None
        self.train_op = None

    @abstractmethod
    def build_network(self):
        pass

    def build_loss(self):
        # weight_decay
        with tf.name_scope("weight_decay"):
            for var in set(tf.get_collection('reg_vars', self.scope)):
                weight_decay = tf.multiply(tf.nn.l2_loss(var), cfg.wd,
                                           name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
                tf.add_to_collection('losses', weight_decay)
        reg_vars = tf.get_collection('losses', self.scope)
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        _logger.add('regularization var num: %d' % len(reg_vars))
        _logger.add('trainable var num: %d' % len(trainable_vars))

        target_dist = tf.clip_by_value(self.target_distribution, 1e-10, 1.)
        predicted_dist = tf.clip_by_value(tf.nn.softmax(self.logits), 1e-10, 1.)

        kl_batch = tf.reduce_sum(target_dist * tf.log(target_dist / predicted_dist), -1)
        # kl_batch = tf.reduce_sum((target_dist - predicted_dist) ** 2, -1)

        tf.add_to_collection('losses', tf.reduce_mean(kl_batch, name='kl_divergence_mean'))
        loss = tf.add_n(tf.get_collection('losses', self.scope), name='loss')
        tf.summary.scalar(loss.op.name, loss)
        tf.add_to_collection('ema/scalar', loss)
        return loss

    def build_mse(self):
        predicted_dist = tf.nn.softmax(self.logits)
        mask = tf.tile(tf.expand_dims(tf.range(1, self.output_class+1), 0), [self.bs, 1])
        predicted_score = tf.reduce_sum(predicted_dist * tf.cast(mask, tf.float32), -1)  # bs

        mse = (predicted_dist - self.gold_label) ** 2
        return mse, predicted_score


    def update_tensor_add_ema_and_opt(self):
        self.logits = self.build_network()
        self.loss = self.build_loss()
        self.mse, self.predicted_score = self.build_mse()

        # ------------ema-------------
        if True:
            self.var_ema = tf.train.ExponentialMovingAverage(cfg.var_decay)
            self.build_var_ema()

        if cfg.mode == 'train':
            self.ema = tf.train.ExponentialMovingAverage(cfg.decay)
            self.build_ema()
        self.summary = tf.summary.merge_all()

        # ---------- optimization ---------
        if cfg.optimizer.lower() == 'adadelta':
            assert cfg.learning_rate > 0.1 and cfg.learning_rate < 1.
            self.opt = tf.train.AdadeltaOptimizer(cfg.learning_rate)
        elif cfg.optimizer.lower() == 'adam':
            assert cfg.learning_rate < 0.1
            self.opt = tf.train.AdamOptimizer(cfg.learning_rate)
        elif cfg.optimizer.lower() == 'rmsprop':
            assert cfg.learning_rate < 0.1
            self.opt = tf.train.RMSPropOptimizer(cfg.learning_rate)

        elif cfg.optimizer.lower() == 'test':
            self.opt = tf.train.RMSPropOptimizer(0.001, 0.75)
            # self.opt = tf.contrib.keras.optimizers.Nadam()
        else:
            raise AttributeError('no optimizer named as \'%s\'' % cfg.optimizer)

        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        # trainable param num:
        # print params num
        all_params_num = 0
        for elem in trainable_vars:
            # elem.name
            var_name = elem.name.split(':')[0]
            if var_name.endswith('emb_mat'): continue
            params_num = 1
            for l in elem.get_shape().as_list(): params_num *= l
            all_params_num += params_num
        _logger.add('Trainable Parameters Number: %d' % all_params_num)

        self.train_op = self.opt.minimize(self.loss, self.global_step,
                                          var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope))

    def build_var_ema(self):
        ema_op = self.var_ema.apply(tf.trainable_variables(),)
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def build_ema(self):
        tensors = tf.get_collection("ema/scalar", scope=self.scope) + \
                  tf.get_collection("ema/vector", scope=self.scope)
        ema_op = self.ema.apply(tensors)
        for var in tf.get_collection("ema/scalar", scope=self.scope):
            ema_var = self.ema.average(var)
            tf.summary.scalar(ema_var.op.name, ema_var)
        for var in tf.get_collection("ema/vector", scope=self.scope):
            ema_var = self.ema.average(var)
            tf.summary.histogram(ema_var.op.name, ema_var)

        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def get_feed_dict(self, sample_batch, data_type='train'):
        # max lens
        sl1, sl2 = 0, 0

        for sample in sample_batch:
            sl1 = max(sl1, len(sample['sentence1_token_digital']))
            sl2 = max(sl2, len(sample['sentence2_token_digital']))


        # token and char
        sent1_token_b = []
        sent1_char_b = []
        sent2_token_b = []
        sent2_char_b = []
        for sample in sample_batch:
            sent1_token = np.zeros([sl1], cfg.intX)
            sent1_char = np.zeros([sl1, self.tl], cfg.intX)
            for idx_t, (token, char_seq_v) in enumerate(zip(sample['sentence1_token_digital'],
                                                            sample['sentence1_char_digital'])):
                sent1_token[idx_t] = token
                for idx_c, char in enumerate(char_seq_v):
                    if idx_c < self.tl:
                        sent1_char[idx_t, idx_c] = char

            sent2_token = np.zeros([sl2], cfg.intX)
            sent2_char = np.zeros([sl2, self.tl], cfg.intX)

            for idx_t, (token, char_seq_v) in enumerate(zip(sample['sentence2_token_digital'],
                                                            sample['sentence2_char_digital'])):
                sent2_token[idx_t] = token
                for idx_c, char in enumerate(char_seq_v):
                    if idx_c < self.tl:
                        sent2_char[idx_t, idx_c] = char
            sent1_token_b.append(sent1_token)
            sent1_char_b.append(sent1_char)
            sent2_token_b.append(sent2_token)
            sent2_char_b.append(sent2_char)
        sent1_token_b = np.stack(sent1_token_b)
        sent1_char_b = np.stack(sent1_char_b)
        sent2_token_b = np.stack(sent2_token_b)
        sent2_char_b = np.stack(sent2_char_b)

        # label
        target_distribution_b = []
        gold_label_b = []
        for sample in sample_batch:
            target_distribution_b.append(sample['distribution'])
            gold_label_b.append(sample['relatedness_score'])
        target_distribution_b = np.array(target_distribution_b, cfg.floatX)
        gold_label_b = np.stack(gold_label_b).astype(cfg.floatX)

        feed_dict = {
            self.sent1_token: sent1_token_b, self.sent1_char: sent1_char_b,
            self.sent2_token: sent2_token_b, self.sent2_char: sent2_char_b,
            self.target_distribution: target_distribution_b,
            self.gold_label: gold_label_b,
            self.is_train: True if data_type == 'train' else False
        }

        return feed_dict

    def step(self, sess, batch_samples, get_summary=False):
        assert isinstance(sess, tf.Session)
        feed_dict = self.get_feed_dict(batch_samples, 'train')
        cfg.time_counter.add_start()
        if get_summary:
            loss, summary, train_op = sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)

        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        cfg.time_counter.add_stop()

        return loss, summary, train_op

