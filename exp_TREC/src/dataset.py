from configs import cfg
from src.utils.record_log import _logger
import json, random, math
import numpy as np
from src.utils.nlp import dynamic_length, dynamic_keep
from src.utils.file import load_glove, save_file


class Dataset(object):
    def __init__(self, data_file_path, data_type, dicts=None):
        self.data_type = data_type
        _logger.add('building data set object for %s' % data_type)
        assert data_type in ['train', 'dev', 'test']
        # check
        if data_type in ['dev', 'test']:
            assert dicts is not None

        raw_data = self.load_question_classification_data(data_file_path, data_type)

        if data_type == 'train':
            self.dicts, self.max_lens = self.count_data_and_build_dict(raw_data)
        else:
            _, self.max_lens = self.count_data_and_build_dict(raw_data, False)
            self.dicts = dicts

        digital_data = self.digitize_data(raw_data, self.dicts, data_type)
        self.nn_data = digital_data
        self.sample_num = len(self.nn_data)
        if data_type == 'train':
            self.emb_mat_token, self.emb_mat_glove = self.generate_index2vec_matrix()

    # external_use
    # how to generate sub tree? : done find all node belonging to the subtree
    def save_dict(self, path):
        save_file(self.dicts, path, 'token and char dict data', 'pickle')

    def get_statistic(self):
        len_list = []
        output = {}
        for sample in self.nn_data:
            len_list.append(len(sample['token']))
        len_array = np.array(len_list).astype('float32')
        output['mean'] = float(np.mean(len_array))
        output['std'] = float(np.std(len_array))
        output['max'] = float(np.max(len_array))
        return output

    def generate_batch_sample_iter(self, max_step=None):
        if max_step is not None:
            batch_size = cfg.train_batch_size

            def data_queue(data, batch_size):
                assert len(data) >= batch_size
                random.shuffle(data)
                data_ptr = 0
                dataRound = 0
                idx_b = 0
                step = 0
                while True:
                    if data_ptr + batch_size <= len(data):
                        yield data[data_ptr:data_ptr + batch_size], dataRound, idx_b
                        data_ptr += batch_size
                        idx_b += 1
                        step += 1
                    elif data_ptr + batch_size > len(data):
                        offset = data_ptr + batch_size - len(data)
                        out = data[data_ptr:]
                        random.shuffle(data)
                        out += data[:offset]
                        data_ptr = offset
                        dataRound += 1
                        yield out, dataRound, 0
                        idx_b = 1
                        step += 1
                    if step >= max_step:
                        break
            batch_num = math.ceil(len(self.nn_data) / batch_size)
            for sample_batch, data_round, idx_b in data_queue(self.nn_data, batch_size):
                yield sample_batch, batch_num, data_round, idx_b
        else:
            batch_size = cfg.test_batch_size
            batch_num = math.ceil(len(self.nn_data) / batch_size)
            idx_b = 0
            sample_batch = []
            for sample in self.nn_data:
                sample_batch.append(sample)
                if len(sample_batch) == batch_size:
                    yield sample_batch, batch_num, 0, idx_b
                    idx_b += 1
                    sample_batch = []
            if len(sample_batch) > 0:
                yield sample_batch, batch_num, 0, idx_b

    # --------- internal use -----------
    def load_question_classification_data(self, data_file_path, data_type):
        _logger.add()
        _logger.add('load file for %s' % data_type)
        dataset = []
        with open(data_file_path, 'r', encoding='latin-1') as file:
            for line in file:
                line_split = line.strip().split(' ')
                cls = line_split[0].split(':')[0]
                sub_cls = line_split[0]
                token = line_split[1:]
                sample = {'token': token, 'cls': cls, 'sub_cls': sub_cls}
                dataset.append(sample)
        _logger.done()
        return dataset

    def count_data_and_build_dict(self, data_list, gene_dicts=True):
        def add_ept_and_unk(a_list):
            a_list.insert(0, '@@@empty')
            a_list.insert(1, '@@@unk')
            return a_list

        _logger.add()
        _logger.add('counting and build dictionaries')
        token_collection = []
        char_collection = []

        sent_len_collection = []
        token_len_collection = []

        for sample in data_list:
            token_collection += sample['token']
            sent_len_collection.append(len(sample['token'] ))
            for token in sample['token']:
                char_collection += list(token)
                token_len_collection.append(len(token))

        max_sent_len = dynamic_length(sent_len_collection, 1, security=False)[0]
        max_token_len = dynamic_length(token_len_collection, 0.99, security=False)[0]

        if gene_dicts:
            # cls dict
            all_cls, all_sub_cls = [], []
            for sample in data_list:
                all_cls.append(sample['cls'])
                all_sub_cls.append(sample['sub_cls'])
            cls_dict = dict([(cls, idx_c) for idx_c, cls in enumerate(set(all_cls))])
            sub_cls_dict = dict([(cls, idx_c) for idx_c, cls in enumerate(set(all_sub_cls))])

            # token & char
            tokenSet = dynamic_keep(token_collection, 1)
            charSet = dynamic_keep(char_collection, 1)
            if cfg.use_glove_unk_token:
                gloveData = load_glove(cfg.word_embedding_length)
                gloveTokenSet = list(gloveData.keys())
                if cfg.lower_word:
                    tokenSet = list(set([token.lower() for token in tokenSet]))  ##!!!
                    gloveTokenSet = list(set([token.lower() for token in gloveTokenSet]))  ##!!!

                # delete token from gloveTokenSet which appears in tokenSet
                for token in tokenSet:
                    try:
                        gloveTokenSet.remove(token)
                    except ValueError:
                        pass
            else:
                if cfg.lower_word:
                    tokenSet = list(set([token.lower() for token in tokenSet]))
                gloveTokenSet = []
            tokenSet = add_ept_and_unk(tokenSet)
            charSet = add_ept_and_unk(charSet)
            dicts = {'token': tokenSet, 'char': charSet, 'glove': gloveTokenSet,
                     'cls': cls_dict, 'sub_cls': sub_cls_dict}
        else:
            dicts = {}
        _logger.done()
        return dicts, {'sent': max_sent_len, 'token': max_token_len}


    def digitize_data(self, data_list, dicts, data_type):
        token2index = dict([(token, idx) for idx, token in enumerate(dicts['token'] + dicts['glove'])])
        char2index = dict([(token, idx) for idx, token in enumerate(dicts['char'])])

        def digitize_token(token):
            token = token if not cfg.lower_word else token.lower()
            try:
                return token2index[token]
            except KeyError:
                return 1

        def digitize_char(char):
            try:
                return char2index[char]
            except KeyError:
                return 1

        _logger.add()
        _logger.add('digitizing data: %s...' % data_type)
        for sample in data_list:
            sample['token_digital'] = [digitize_token(token) for token in sample['token']]
            sample['char_digital'] = [[digitize_char(char) for char in list(token)]
                                      for token in sample['token']]
            sample['cls_digital'] = dicts['cls'][sample['cls']]
            sample['sub_cls_digital'] = dicts['sub_cls'][sample['sub_cls']]
        _logger.done()
        return data_list

    def generate_index2vec_matrix(self):
        _logger.add()
        _logger.add('generate index to vector numpy matrix')

        token2vec = load_glove(cfg.word_embedding_length)
        if cfg.lower_word:
            newToken2vec = {}
            for token, vec in token2vec.items():
                newToken2vec[token.lower()] = vec
            token2vec = newToken2vec

        # prepare data from trainDataset and devDataset
        mat_token = np.random.uniform(-0.05, 0.05, size=(len(self.dicts['token']), cfg.word_embedding_length)).astype(
            cfg.floatX)

        mat_glove = np.zeros((len(self.dicts['glove']), cfg.word_embedding_length), dtype=cfg.floatX)

        for idx, token in enumerate(self.dicts['token']):
            try:
                mat_token[idx] = token2vec[token]
            except KeyError:
                pass
            mat_token[0] = np.zeros(shape=(cfg.word_embedding_length,), dtype=cfg.floatX)

        for idx, token in enumerate(self.dicts['glove']):
            mat_glove[idx] = token2vec[token]

        _logger.add('Done')
        return mat_token, mat_glove


if __name__ == '__main__':
    data = Dataset(cfg.train_data_path, 'train')