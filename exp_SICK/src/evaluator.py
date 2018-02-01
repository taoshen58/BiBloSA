from configs import cfg
from src.utils.record_log import _logger
import numpy as np
import tensorflow as tf
import scipy.stats as stats


class Evaluator(object):
    def __init__(self, model):
        self.model = model
        self.global_step = model.global_step

        ## ---- summary----
        self.build_summary()
        self.writer = tf.summary.FileWriter(cfg.summary_dir)

    def get_evaluation(self, sess, dataset_obj, global_step=None):
        _logger.add()
        _logger.add('getting evaluation result for %s' % dataset_obj.data_type)

        logits_list, loss_list = [], []
        target_score_list, predicted_score_list = [], []
        for sample_batch, _, _, _ in dataset_obj.generate_batch_sample_iter():
            feed_dict = self.model.get_feed_dict(sample_batch, 'dev')
            logits, loss, predicted_score = sess.run([self.model.logits, self.model.loss,
                                           self.model.predicted_score], feed_dict)
            logits_list.append(np.argmax(logits, -1))
            loss_list.append(loss)
            predicted_score_list.append(predicted_score)

            for sample in sample_batch:
                target_score_list.append(sample['relatedness_score'])

        logits_array = np.concatenate(logits_list, 0)
        loss_value = np.mean(loss_list)
        target_scores = np.array(target_score_list)
        predicted_scores = np.concatenate(predicted_score_list, 0)
        # pearson, spearman, mse
        pearson_value = stats.pearsonr(target_scores, predicted_scores)[0]
        spearman_value = stats.spearmanr(target_scores, predicted_scores)[0]
        mse_value = np.mean((target_scores - predicted_scores) ** 2)

        # todo: analysis
        # analysis_save_dir = cfg.mkdir(cfg.answer_dir, 'gs_%d' % global_step or 0)
        # OutputAnalysis.do_analysis(dataset_obj, logits_array, accu_array, analysis_save_dir,
        #                            cfg.fine_grained)

        if global_step is not None:
            if dataset_obj.data_type == 'train':
                summary_feed_dict = {
                    self.train_loss: loss_value,
                    self.train_pearson: pearson_value,
                    self.train_spearman: spearman_value,
                    self.train_mse: mse_value,
                }
                summary = sess.run(self.train_summaries, summary_feed_dict)
                self.writer.add_summary(summary, global_step)
            elif dataset_obj.data_type == 'dev':
                summary_feed_dict = {
                    self.dev_loss: loss_value,
                    self.dev_pearson: pearson_value,
                    self.dev_spearman: spearman_value,
                    self.dev_mse: mse_value,
                }
                summary = sess.run(self.dev_summaries, summary_feed_dict)
                self.writer.add_summary(summary, global_step)
            else:
                summary_feed_dict = {
                    self.test_loss: loss_value,
                    self.test_pearson: pearson_value,
                    self.test_spearman: spearman_value,
                    self.test_mse: mse_value,
                }
                summary = sess.run(self.test_summaries, summary_feed_dict)
                self.writer.add_summary(summary, global_step)

        return loss_value, (pearson_value, spearman_value, mse_value)


    # --- internal use ------
    def build_summary(self):
        with tf.name_scope('train_summaries'):
            self.train_loss = tf.placeholder(tf.float32, [], 'train_loss')
            self.train_pearson = tf.placeholder(tf.float32, [], 'train_pearson')
            self.train_spearman = tf.placeholder(tf.float32, [], 'train_spearman')
            self.train_mse = tf.placeholder(tf.float32, [], 'train_mse')
            tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_loss', self.train_loss))
            tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_pearson', self.train_pearson))
            tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_spearman', self.train_spearman))
            tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_mse', self.train_mse))
            self.train_summaries = tf.summary.merge_all('train_summaries_collection')

        with tf.name_scope('dev_summaries'):
            self.dev_loss = tf.placeholder(tf.float32, [], 'dev_loss')
            self.dev_pearson = tf.placeholder(tf.float32, [], 'dev_pearson')
            self.dev_spearman = tf.placeholder(tf.float32, [], 'dev_spearman')
            self.dev_mse = tf.placeholder(tf.float32, [], 'dev_mse')
            tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_loss',self.dev_loss))
            tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_pearson', self.dev_pearson))
            tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_spearman', self.dev_spearman))
            tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_mse', self.dev_mse))
            self.dev_summaries = tf.summary.merge_all('dev_summaries_collection')

        with tf.name_scope('test_summaries'):
            self.test_loss = tf.placeholder(tf.float32, [], 'test_loss')
            self.test_pearson = tf.placeholder(tf.float32, [], 'test_pearson')
            self.test_spearman = tf.placeholder(tf.float32, [], 'test_spearman')
            self.test_mse = tf.placeholder(tf.float32, [], 'test_mse')
            tf.add_to_collection('test_summaries_collection', tf.summary.scalar('test_loss',self.test_loss))
            tf.add_to_collection('test_summaries_collection', tf.summary.scalar('test_pearson', self.test_pearson))
            tf.add_to_collection('test_summaries_collection', tf.summary.scalar('test_spearman', self.test_spearman))
            tf.add_to_collection('test_summaries_collection', tf.summary.scalar('test_mse', self.test_mse))
            self.test_summaries = tf.summary.merge_all('test_summaries_collection')




