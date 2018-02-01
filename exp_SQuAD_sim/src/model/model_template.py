from configs import cfg
from src.utils.record_log import _logger
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod


class ModelTemplate(metaclass=ABCMeta):
    def __init__(self, token_emb_mat, glove_emb_mat, tds, tel, hn, scope):
        self.scope = scope
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.token_emb_mat, self.glove_emb_mat = token_emb_mat, glove_emb_mat

        # ---------place holders-------------
        self.context_token = tf.placeholder(tf.int32, [None, None, None], name='context_token')
        self.question_token = tf.placeholder(tf.int32, [None, None], name='question_token')
        self.sent_label = tf.placeholder(tf.int32, [None], 'sent_label')
        self.is_train = tf.placeholder(tf.bool, [], name='is_train')

        # -------- Lengths -------
        self.tds, self.tel = tds, tel
        self.hn = hn
        self.bs = tf.shape(self.context_token)[0]
        self.sn, self.sl = tf.shape(self.context_token)[1], tf.shape(self.context_token)[2]
        self.ql = tf.shape(self.question_token)[1]

        # ------other ------
        self.context_token_mask = tf.cast(self.context_token, tf.bool)
        self.question_token_mask = tf.cast(self.question_token, tf.bool)
        self.context_token_len = tf.reduce_sum(tf.cast(self.context_token_mask, tf.int32), -1)
        self.question_token_len = tf.reduce_sum(tf.cast(self.question_token_mask, tf.int32), -1)

        self.context_sent_mask = tf.cast(tf.reduce_sum(tf.cast(self.context_token_mask, tf.int32), -1), tf.bool)
        self.context_sent_len = tf.reduce_sum(tf.cast(self.context_sent_mask, tf.int32), -1)

        self.tensor_dict = {}

        # ----- start ------
        self.logits = None
        self.loss = None
        self.accuracy = None
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
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.sent_label,
            logits=self.logits
        )
        tf.add_to_collection('losses', tf.reduce_mean(losses, name='xentropy_loss_mean'))
        loss = tf.add_n(tf.get_collection('losses', self.scope), name='loss')
        tf.summary.scalar(loss.op.name, loss)
        tf.add_to_collection('ema/scalar', loss)
        return loss

    def build_accuracy(self):
        correct = tf.equal(
            tf.cast(tf.argmax(self.logits, -1), tf.int32),
            self.sent_label
        )  # [bs]
        return tf.cast(correct, tf.float32)

    def update_tensor_add_ema_and_opt(self):
        self.logits = self.build_network()
        self.loss = self.build_loss()
        self.accuracy = self.build_accuracy()

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
            assert cfg.learning_rate > 0.1 and cfg.learning_rate <= 1.
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
        max_sn, max_sl, max_ql = 0, 0, 0

        for sample in sample_batch:
            max_ql = max(max_ql, len(sample['question_token_digital']))
            max_sn = max(max_sn, len(sample['context_token_digital']))
            for sent_token in sample['context_token_digital']:
                max_sl = max(max_sl, len(sent_token))

        # -----------

        context_token_b = []
        question_token_b = []

        # tokens
        for sample in sample_batch:
            context_token = np.zeros([max_sn, max_sl], cfg.intX)
            for idx_s, sent_token in enumerate(sample['context_token_digital']):
                for idx_t, token in enumerate(sent_token):
                    context_token[idx_s, idx_t] = token
            context_token_b.append(context_token)

            question_token = np.zeros([max_ql], cfg.intX)
            for idx_qt, qtoken in enumerate(sample['question_token_digital']):
                question_token[idx_qt] = qtoken
            question_token_b.append(question_token)

        context_token_b = np.stack(context_token_b)
        question_token_b = np.stack(question_token_b)

        feed_dict = {
            self.context_token: context_token_b, self.question_token: question_token_b,
            self.is_train: True if data_type == 'train' else False
        }
        # labels
        if data_type in ['train', 'dev']:
            sent_label_b = []
            for sample in sample_batch:
                sent_label_b.append(sample['answers'][0]['sent_label'])
            sent_label_b = np.stack(sent_label_b).astype(cfg.intX)
            feed_dict[self.sent_label] = sent_label_b
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



