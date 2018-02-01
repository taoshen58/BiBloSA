import platform
import argparse
import os
from os.path import join
from src.utils.time_counter import TimeCounter


class Configs(object):
    def __init__(self):
        self.project_dir = os.getcwd()
        self.dataset_dir = join(self.project_dir, 'dataset')

        # ------parsing input arguments"--------
        parser = argparse.ArgumentParser()
        parser.register('type', 'bool', (lambda x: x.lower() in ('True', "yes", "true", "t", "1")))

        # @ ----- control ----
        parser.add_argument('--debug', type='bool', default=False, help='whether run as debug mode')
        parser.add_argument('--mode', type=str, default='train', help='train, dev or test')
        parser.add_argument('--network_type', type=str, default='test', help='None')
        parser.add_argument('--log_period', type=int, default=500, help='log_period')  ###  change for running
        parser.add_argument('--save_period', type=int, default=1500, help='save_period')
        parser.add_argument('--eval_period', type=int, default=1500, help='eval_period')  ###  change for running
        parser.add_argument('--gpu', type=int, default=2, help='eval_period')
        parser.add_argument('--gpu_mem', type=float, default=None, help='gpu memory ratio to employ')

        parser.add_argument('--save_model', type='bool', default=False, help='save model')
        parser.add_argument('--load_model', type='bool', default=False, help='load_model')
        parser.add_argument('--load_step', type=int, default=None, help='load specified step')
        parser.add_argument('--load_path', type=str, default=None, help='load specified step')
        parser.add_argument('--swap_memory', type='bool', default=False, help='help...')
        parser.add_argument('--model_dir_suffix', type=str, default='', help='help...')

        # @ ----------training ------
        parser.add_argument('--max_epoch', type=int, default=40, help='Max Epoch Number')
        parser.add_argument('--num_steps', type=int, default=100000, help='num_steps')
        parser.add_argument('--train_batch_size', type=int, default=32, help='Train Batch Size')
        parser.add_argument('--test_batch_size_gain', type=float, default=1.2, help='Test Batch Size')

        # @ ----- Text Processing ----
        parser.add_argument('--word_embedding_length', type=int, default=100, help='')
        parser.add_argument('--use_glove_unk_token', type='bool', default=True, help='')
        parser.add_argument('--glove_corpus', type=str, default='6B', help='choose glove corpus to employ')
        parser.add_argument('--lower_word', type='bool', default=True, help='help...')
        parser.add_argument('--sent_len_rate', type=float, default=0.97, help='for space-efficiency')

        # @ ------neural network-----
        parser.add_argument('--dropout', type=float, default=0.75, help='')
        parser.add_argument('--wd', type=float, default=5e-5, help='weight decay factor')
        parser.add_argument('--hidden_units_num', type=int, default=100, help='Hidden units number of Neural Network')
        parser.add_argument('--optimizer', type=str, default='adadelta', help='choose an optimizer[adadelta|adam]')
        parser.add_argument('--learning_rate', type=float, default=0.5, help='Learning rate')
        parser.add_argument('--var_decay', type=float, default=0.999, help='Learning rate')
        parser.add_argument('--decay', type=float, default=0.9, help='Learning rate')

        # @ ------------- other ----------
        parser.add_argument(
            '--context_fusion_method', type=str, default='block',
            help='[block|lstm|gru|sru|sru_normal|cnn|cnn_kim|multi_head|multi_head_git|disa|no_ct]')
        parser.add_argument('--block_len', type=int, default=None, help='block_len')

        parser.set_defaults(shuffle=True)
        self.args = parser.parse_args()

        ## ---- to member variables -----
        for key, value in self.args.__dict__.items():
            if key not in ['test', 'shuffle']:
                exec('self.%s = self.args.%s' % (key, key))

        # ----------name----------
        # # ---file name ----
        self.train_dataset_name = 'train-v1.1.json'
        self.dev_dataset_name = 'dev-v1.1.json'

        network_type = self.network_type
        if not network_type == 'test':
            params_name_list = ['network_type', 'dropout', 'wd',
                                'glove_corpus', 'word_embedding_length', 'sent_len_rate',
                                'learning_rate', 'optimizer']
            if network_type.find('context_fusion') >= 0:
                params_name_list.append('context_fusion_method')
                if self.block_len is not None:
                    params_name_list.append('block_len')
            self.model_name = self.get_params_str(params_name_list)
        else:
            # self.model_name = self.get_params_str(['network_type'])
            self.model_name = network_type

        self.processed_name = 'processed' + self.get_params_str(['glove_corpus', 'word_embedding_length',
                                                                 'lower_word', 'use_glove_unk_token',
                                                                 'sent_len_rate']) + '.pickle'
        self.model_ckpt_name = 'modelfile.ckpt'

        # ---------- dir -------------
        self.glove_dir = join(self.dataset_dir, 'glove')
        self.result_dir = self.mkdir(self.project_dir, 'result')
        self.standby_log_dir = self.mkdir(self.result_dir, 'log')
        self.dict_dir = self.mkdir(self.result_dir, 'dict')
        self.processed_dir = self.mkdir(self.result_dir, 'processed_data')
        self.tree_dir = self.mkdir(self.result_dir, 'tree')

        self.log_dir = None
        self.all_model_dir = self.mkdir(self.result_dir, 'model')
        self.model_dir = self.mkdir(self.all_model_dir, self.model_dir_suffix + self.model_name)
        self.log_dir = self.mkdir(self.model_dir, 'log_files')
        self.summary_dir = self.mkdir(self.model_dir, 'summary')
        self.ckpt_dir = self.mkdir(self.model_dir, 'ckpt')
        self.answer_dir = self.mkdir(self.model_dir, 'answer')

        # -------- path --------
        self.train_dataset_path = join(self.dataset_dir, 'SQuAD', self.train_dataset_name)
        self.dev_dataset_path = join(self.dataset_dir, 'SQuAD', self.dev_dataset_name)

        self.processed_path = join(self.processed_dir, self.processed_name)
        self.ckpt_path = join(self.ckpt_dir, self.model_ckpt_name)


        # ---- global------
        self.floatX = 'float32'
        self.intX = 'int32'
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        self.time_counter = TimeCounter()

    def get_params_str(self, params):
        def abbreviation(name):
            words = name.strip().split('_')
            abb = ''
            for word in words:
                abb += word[0]
            return abb

        abbreviations = map(abbreviation, params)
        model_params_str = ''
        for paramsStr, abb in zip(params, abbreviations):
            model_params_str += '_' + abb + '_' + str(eval('self.args.' + paramsStr))
        return model_params_str

    def mkdir(self, *args):
        dirPath = join(*args)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        return dirPath

    def get_file_name_from_path(self, path):
        assert isinstance(path, str)
        fileName = '.'.join((path.split('/')[-1]).split('.')[:-1])
        return fileName

cfg = Configs()