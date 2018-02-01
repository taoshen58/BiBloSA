import math

import tensorflow as tf
import numpy as np

from configs import cfg
from src.dataset import Dataset
from src.evaluator import Evaluator
from src.graph_handler import GraphHandler
from src.perform_recorder import PerformRecoder
from src.utils.file import load_file, save_file
from src.utils.record_log import _logger

from src.time_accu_recorder import TimeAccuRecorder


# choose network
network_type = cfg.network_type

if network_type == 'exp_context_fusion':
    from src.model.model_context_fusion import ModelContextFusion as Model
model_set =['exp_context_fusion']


def train():
    n_fold_val = 10
    output_model_params()

    loadFile = True
    ifLoad, data = False, None
    if loadFile:
        ifLoad, data = load_file(cfg.processed_path, 'processed data', 'pickle')
    if not ifLoad or not loadFile:
        data_obj = Dataset(cfg.data_path, cfg.dataset_type)
        save_file({'data_obj': data_obj},
                  cfg.processed_path)
    else:
        data_obj = data['data_obj']

    data_obj.split_dataset_to_blocks(n_fold_val)

    # for block len
    if cfg.block_len is None and cfg.context_fusion_method == 'block':
        _logger.add()
        _logger.add('calculating block length for dataset')
        statistic = data_obj.get_statistic()
        expected_n = statistic['mean'] + statistic['std'] * math.sqrt(2. * math.log(1. * cfg.train_batch_size))
        dy_block_len = math.ceil(math.pow(2 * expected_n, 1.0 / 3)) + 1  # fixme: change length
        cfg.block_len = dy_block_len
        _logger.add('block length is %d' % dy_block_len)

    emb_mat_token, emb_mat_glove = data_obj.emb_mat_token, data_obj.emb_mat_glove
    output_cls_num = data_obj.class_num
    steps_per_epoch = int(math.ceil(1.0 * data_obj.sample_num / cfg.train_batch_size))
    num_steps = cfg.num_steps or steps_per_epoch * cfg.max_epoch

    dev_performance_list = []
    for n_th_fold in range(n_fold_val):
        time_accu_recorder = TimeAccuRecorder(data_obj.dataset_type, n_th_fold, cfg.answer_dir)

        g = tf.Graph()
        with g.as_default():
            with tf.variable_scope("%s_%s" % (cfg.dataset_type, network_type)) as scope:
                if network_type in model_set:
                    model = Model(emb_mat_token, emb_mat_glove, len(data_obj.dicts['token']),
                                  len(data_obj.dicts['char']), data_obj.max_lens['token'], output_cls_num,
                                  scope=scope.name)
                else:
                    assert RuntimeError

                graphHandler = GraphHandler(model)
                evaluator = Evaluator(model)
                performRecoder = PerformRecoder(1)

                if cfg.gpu_mem is None:
                    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.gpu_mem,
                                                allow_growth=True)
                    graph_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

                else:
                    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.gpu_mem)
                    graph_config = tf.ConfigProto(gpu_options=gpu_options)
                # graph_config.gpu_options.allow_growth = True
                sess = tf.Session(config=graph_config)
                graphHandler.initialize(sess)

                global_step = 0
                for sample_batch, batch_num, data_round, idx_b in \
                        data_obj.generate_batch_sample_iter(n_th_fold, num_steps):
                    global_step = sess.run(model.global_step) + 1
                    if_get_summary = global_step % (cfg.log_period or steps_per_epoch) == 0
                    loss, summary, train_op = model.step(sess, sample_batch, get_summary=if_get_summary)
                    # if global_step % 100 == 0:
                    _logger.add('cross validation index: %d' % n_th_fold)
                    _logger.add('data round: %d: %d/%d, global step:%d -- loss: %.4f' %
                                (data_round, idx_b, batch_num, global_step, loss))

                    if if_get_summary:
                        graphHandler.add_summary(summary, global_step)

                    # Occasional evaluation
                    if global_step % (cfg.eval_period or steps_per_epoch) == 0:
                        # ---- dev ----
                        dev_loss, dev_accu = evaluator.get_evaluation(
                            sess, data_obj, n_th_fold, global_step
                        )
                        _logger.add('==> for dev, loss: %.4f, accuracy: %.4f' %
                                    (dev_loss, dev_accu))

                        # record time vs. accuracy
                        time_accu_recorder.add_data(cfg.time_counter.global_training_time, dev_accu)
                        is_in_top, deleted_step = performRecoder.update_top_list(global_step, dev_accu, sess)
                    this_epoch_time, mean_epoch_time = cfg.time_counter.update_data_round(data_round)
                    # if this_epoch_time is not None and mean_epoch_time is not None:
                    #     _logger.add('##> this epoch time: %f, mean epoch time: %f' % (this_epoch_time, mean_epoch_time))
                dev_performance_list.append(performRecoder.best_result)
                _logger.add("%d th x val accuracy is %.4f" % (n_th_fold, performRecoder.best_result))
                time_accu_recorder.save_to_file()

    if len(dev_performance_list) > 0:
        dev_performance_array = np.array(dev_performance_list)
        xval_average = np.mean(dev_performance_array)
        xval_std = np.std(dev_performance_array)
    else:
        xval_average = 0
        xval_std = 0
    dev_performance_list_str = [str(elem) for elem in dev_performance_list]
    _logger.add("all accuracies: %s" % ', '.join(dev_performance_list_str))
    _logger.add('%d fold cross validation average accuracy is %f, standard variance is %f' % \
                (n_fold_val, xval_average, xval_std))
    _logger.writeToFile()

def main(_):
    if cfg.mode == 'train':
        train()
    else:
        raise RuntimeError('no running mode named as %s' % cfg.mode)

def output_model_params():
    _logger.add()
    _logger.add('==>model_title: ' + cfg.model_name[1:])
    _logger.add()
    for key, value in cfg.args.__dict__.items():
        if key not in ['test', 'shuffle']:
            _logger.add('%s: %s' % (key, value))

if __name__ == '__main__':
    tf.app.run()
