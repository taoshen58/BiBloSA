from configs import cfg
from src.utils.record_log import _logger
from src.utils.nlp import dynamic_length, dynamic_keep
from src.utils.file import load_glove, save_file
import numpy as np
import nltk
import re
import os
import math
import random


class Dataset(object):
    def __init__(self, data_file_path, dataset_type=None):
        self.dataset_type = dataset_type

        data_list, self.class_num = self.load_sc_and_process_data(data_file_path)
        self.dicts, self.max_lens = self.count_data_and_build_dict(data_list, gene_dicts=True)
        self.digital_data = self.digitize_data(data_list, self.dicts, self.dataset_type)
        self.sample_num = len(self.digital_data)
        self.emb_mat_token, self.emb_mat_glove = self.generate_index2vec_matrix()

        self.nn_data_blocks = None

    # -------------- external use --------------

    def split_dataset_to_blocks(self, n_fold=10):
        # split data for 10-fold validation
        self.nn_data_blocks = self.split_data_list(self.digital_data, n_fold)


    def save_dict(self, path):
        save_file(self.dicts, path, 'token and char dict data', 'pickle')

    def generate_batch_sample_iter(self, validation_idx, max_step=None):
        if max_step is not None:
            train_data_list = []
            for idx_db, data_block in enumerate(self.nn_data_blocks):
                if idx_db != validation_idx:
                    train_data_list.extend(data_block)
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
            batch_num = math.ceil(len(train_data_list) / batch_size)
            for sample_batch, data_round, idx_b in data_queue(train_data_list, batch_size):
                yield sample_batch, batch_num, data_round, idx_b
        else:
            dev_data_list = self.nn_data_blocks[validation_idx]
            batch_size = cfg.test_batch_size
            batch_num = math.ceil(len(dev_data_list) / batch_size)
            idx_b = 0
            sample_batch = []
            for sample in dev_data_list:
                sample_batch.append(sample)
                if len(sample_batch) == batch_size:
                    yield sample_batch, batch_num, 0, idx_b
                    idx_b += 1
                    sample_batch = []
            if len(sample_batch) > 0:
                yield sample_batch, batch_num, 0, idx_b

    def get_statistic(self):
        len_list = []
        output = {}
        for nn_data_block in self.nn_data_blocks:
            for sample in nn_data_block:
                len_list.append(len(sample['token']))
        len_array = np.array(len_list).astype('float32')
        output['mean'] = float(np.mean(len_array))
        output['std'] = float(np.std(len_array))
        output['max'] = float(np.max(len_array))
        return output

    # -------------- internal use --------------
    def load_sc_and_process_data(self, data_file_path):
        data_list = []
        gold_label_set = set()
        with open(data_file_path, 'r', encoding='latin-1') as file:
            for line in file:
                split_list = line.strip().split(' ')
                gold_label = int(split_list[0])
                sentence = ' '.join(split_list[1:])
                token_list = Dataset.further_tokenize(nltk.word_tokenize(sentence))
                sample = {'sentence':sentence, 'token': token_list, 'gold_label': gold_label}
                data_list.append(sample)
                gold_label_set.add(gold_label)
        return data_list, len(gold_label_set)

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
            sent_len_collection.append(len(sample['token']))
            for token in sample['token']:
                char_collection += list(token)
                token_len_collection.append(len(token))

        max_sent_len = dynamic_length(sent_len_collection, 1, security=False)[0]
        max_token_len = dynamic_length(token_len_collection, 0.99, security=False)[0]

        if gene_dicts:
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
            dicts = {'token': tokenSet, 'char': charSet, 'glove': gloveTokenSet}
        else:
            dicts = {}
        _logger.done()
        return dicts, {'sent': max_sent_len, 'token': max_token_len}

    def digitize_data(self, data_list, dicts, dataset_type):
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
        _logger.add('digitizing data: %s...' % dataset_type)
        for sample in data_list:
            sample['token_digital'] = [digitize_token(token) for token in sample['token']]
            sample['char_digital'] = [[digitize_char(char) for char in list(token)]
                                      for token in sample['token']]
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

    def split_data_list(self, data_list, n=10):
        assert len(data_list) >= n
        random.shuffle(data_list)
        unit_len = len(data_list) * 1. / n
        idxs = [math.floor(idx * unit_len) for idx in range(n + 1)]  # len = n+1
        idxs[-1] = len(data_list)
        nn_data = []
        for i in range(n):
            nn_data.append(data_list[idxs[i]:idxs[i+1]])
        return nn_data


    @staticmethod
    def further_tokenize(temp_tokens):
        tokens = []  # [[(s,e),...],...]
        for token in temp_tokens:
            l = (
                "-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
            tokens.extend(re.split("([{}])".format("".join(l)), token))
        return tokens


if __name__ == '__main__':
    paths = [
        '/Users/xxx/Workspaces/dataset/sentence_classification/custrev.all',
        '/Users/xxx/Workspaces/dataset/sentence_classification/mpqa.all',
        '/Users/xxx/Workspaces/dataset/sentence_classification/rt-polarity.all',
        '/Users/xxx/Workspaces/dataset/sentence_classification/subj.all',
    ]

    data_obj = Dataset(paths[0], cfg.dataset_type)







