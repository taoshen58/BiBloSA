from configs import cfg
from src.utils.record_log import _logger
import json, re, random, math
import numpy as np
from tqdm import tqdm
import nltk
from src.utils.nlp import dynamic_length, dynamic_keep
from src.utils.file import load_glove


class Dataset(object):
    def __init__(self, file_path, data_type, dicts= None):
        assert data_type in ['train', 'dev', 'test']

        if data_type == 'test' or data_type == 'dev':
            assert dicts is not None

        self.data_type = data_type

        raw_data = self.load_squad_dataset(file_path)
        processed_data = self.process_raw_dataset(raw_data, self.data_type)

        if data_type == 'train':
            self.dicts, self.max_lens = self.count_data_and_build_dict(processed_data, cfg.sent_len_rate, True)
            self.dicts, self.max_lens = self.count_data_and_build_dict(processed_data, cfg.sent_len_rate, True)
            self.dicts, self.max_lens = self.count_data_and_build_dict(processed_data, cfg.sent_len_rate, True)
        else:
            self.dicts = dicts
            _, self.max_lens = self.count_data_and_build_dict(processed_data, 1., False)

        digitized_data = self.digitize_dataset(processed_data, self.dicts, data_type)
        self.shared_data, self.nn_data = self.divide_data_into_shared_data(digitized_data, data_type)

        self.emb_mat_token, self.emb_mat_glove = self.generate_index2vec_matrix()

    # ---- external use -------
    def generate_batch_sample_iter(self, max_step=None):
        if max_step is not None:
            batch_size = cfg.train_batch_size

            def data_queue(data, _batch_size, _batch_num):
                assert len(data) >= _batch_size
                random.shuffle(data)
                data_ptr = 0
                dataRound = 0
                idx_b = 0
                step = 0

                while True:
                    if data_ptr+_batch_size <= len(data):
                        yield data[data_ptr:data_ptr+_batch_size], _batch_num, dataRound, idx_b
                        data_ptr += _batch_size
                        idx_b += 1
                        step +=1
                    elif data_ptr+_batch_size > len(data):
                        offset = data_ptr+_batch_size - len(data)
                        out = data[data_ptr:]
                        random.shuffle(data)
                        out += data[:offset]
                        data_ptr = offset
                        dataRound += 1
                        yield out, _batch_num, dataRound, 0
                        idx_b = 1
                        step += 1
                    if max_step is not None and step >= max_step:
                        break

            batch_num = math.ceil(len(self.nn_data) / batch_size)
            for qaBatch, batchNum, dataRound, idx_b in data_queue(self.nn_data, batch_size, batch_num):
                yield self.gene_sample_batch_from_qa_batch(qaBatch), batchNum, dataRound, idx_b
        else:
            batch_size = math.ceil(cfg.train_batch_size * cfg.test_batch_size_gain)
            batch_num = math.ceil(len(self.nn_data) / batch_size)
            idx_b = 0
            qa_batch = []
            for sample in self.nn_data:
                qa_batch.append(sample)
                if len(qa_batch) == batch_size:
                    yield self.gene_sample_batch_from_qa_batch(qa_batch), batch_num, 0, idx_b
                    idx_b += 1
                    qa_batch = []
            if len(qa_batch) > 0:
                yield self.gene_sample_batch_from_qa_batch(qa_batch), batch_num, 0, idx_b

    def gene_sample_batch_from_qa_batch(self, qa_batch):
        sample_batch = []
        for qa in qa_batch:
            sample = {}
            sample.update(qa)
            sample.update(self.shared_data[qa['shared_index']])
            sample_batch.append(sample)
        return sample_batch

    def get_statistic(self):
        len_list = []
        output = {}
        for qa in self.nn_data:
            ct = self.shared_data[qa['shared_index']]
            for sent_token in ct:
                len_list.append(len(sent_token))
        len_array = np.array(len_list).astype('float32')
        output['mean'] = float(np.mean(len_array))
        output['std'] = float(np.std(len_array))
        output['max'] = float(np.max(len_array))
        return output

    def filter_data(self):
        new_nn_data = []

        for qa in self.nn_data:
            context = self.shared_data[qa['shared_index']]
            satisfy_cond = True
            if self.data_type == 'train':
                if len(qa['question_token']) > self.max_lens['question']:
                    satisfy_cond = False
                if len(context['context_token']) > self.max_lens['sent_num']:
                    satisfy_cond = False
                for sent_token in context['context_token']:
                    if len(sent_token) > self.max_lens['sent_len']:
                        satisfy_cond = False
            else:
                pass
            if satisfy_cond:
                new_nn_data.append(qa)

        self.nn_data = new_nn_data

    @property
    def sample_num(self):
        return len(self.nn_data)

    # ---- internal use -------
    @staticmethod
    def load_squad_dataset(file_path):
        with open(file_path, 'r', encoding='utf-8') as data_file:
            line = data_file.readline()
            dataset = json.loads(line)
        return dataset['data']

    @staticmethod
    def process_raw_dataset(raw_data, data_type):
        _logger.add()
        _logger.add('processing raw data: %s...' % data_type)
        for topic in tqdm(raw_data):
            for paragraph in topic['paragraphs']:
                # context
                paragraph['context'] = paragraph['context'].replace("''", '" ').replace("``", '" ')
                paragraph['context_token'] = [[token.replace("''", '"').replace("``", '"')
                                               for token in nltk.word_tokenize(sent)]
                                              for sent in nltk.sent_tokenize(paragraph['context'])]
                paragraph['context_token'] = [Dataset.further_tokenize(sent) for sent in paragraph['context_token']]

                # qas
                for qa in paragraph['qas']:
                    qa['question'] = qa['question'].replace("''", '" ').replace("``", '" ')
                    qa['question_token'] = Dataset.further_tokenize(
                        [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(qa['question'])])
                    # # tag generation
                    for answer in qa['answers']:
                        answer['sent_label'] = Dataset.sentence_label_generation(
                            paragraph['context'], paragraph['context_token'], answer['text'], answer['answer_start'])
        _logger.done()
        return raw_data

    @staticmethod
    def count_data_and_build_dict(dataset, sent_len_rate, gene_dicts=True):
        def add_ept_and_unk(a_list):
            a_list.insert(0, '@@@empty')
            a_list.insert(1, '@@@unk')
            return a_list
        _logger.add()
        _logger.add('counting and build dictionaries')

        token_collection = []
        sent_num_collection = []
        sent_len_collection = []
        question_len_collection = []

        for topic in dataset:
            for paragraph in topic['paragraphs']:
                sent_num_collection.append(len(paragraph['context_token']))
                for sent_token in paragraph['context_token']:
                    sent_len_collection.append(len(sent_token))
                    token_collection += sent_token
                for qa in paragraph['qas']:
                    question_len_collection.append(len(qa['question_token']))
                    token_collection += qa['question_token']

        _logger.done()

        max_sent_num, _ = dynamic_length(sent_num_collection, 1.)
        max_sent_len, _ = dynamic_length(sent_len_collection, sent_len_rate)
        max_question_len, _ = dynamic_length(question_len_collection, 0.995)

        if gene_dicts:
            tokenSet = dynamic_keep(token_collection, 0.995)
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
            dicts = {'token': tokenSet, 'glove': gloveTokenSet}
        else:
            dicts = {}
        _logger.done()
        return dicts, {'sent_num': max_sent_num, 'sent_len': max_sent_len,'question': max_question_len}

    @staticmethod
    def digitize_dataset(dataset, dicts, data_type):
        token2index = dict([(token, idx) for idx, token in enumerate(dicts['token'] + dicts['glove'])])

        def digitize_token(token):
            token = token if not cfg.lower_word else token.lower()
            try:
                return token2index[token]
            except KeyError:
                return 1

        _logger.add()
        _logger.add('digitizing data: %s...' % data_type)

        for topic in tqdm(dataset):
            for paragraph in topic['paragraphs']:
                paragraph['context_token_digital'] = [[digitize_token(token)for token in sent]
                                                      for sent in paragraph['context_token']]
                for qa in paragraph['qas']:
                    qa['question_token_digital'] = [digitize_token(token) for token in qa['question_token']]
        _logger.done()
        return dataset

    @staticmethod
    def divide_data_into_shared_data(dataset, data_type):
        _logger.add()
        _logger.add('dividing data in to shared data: %s' % data_type)

        shared_data = []
        nn_data = []

        shared_idx = 0
        for topic in tqdm(dataset):
            for paragraph in topic['paragraphs']:
                shared_data.append({
                    'context': paragraph['context'], 'context_token': paragraph['context_token'],
                    'context_token_digital': paragraph['context_token_digital'],
                })
                for qa in paragraph['qas']:
                    qa['shared_index'] = shared_idx
                    nn_data.append(qa)
                shared_idx += 1
        _logger.done()
        return shared_data, nn_data

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

    # ------ utilities ------
    @staticmethod
    def further_tokenize(temp_tokens):
        tokens = []  # [[(s,e),...],...]
        for token in temp_tokens:
            l = (
                "-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
            tokens.extend(re.split("([{}])".format("".join(l)), token))
        return tokens

    @staticmethod
    def find_token_boundaries(context_text, context_token):
        c_pointer = 0
        sent_spans = []
        for idx_s, sent in enumerate(context_token):
            token_spans = []
            for idx_t, token in enumerate(sent):
                find_start_idx = context_text.find(token, c_pointer)

                assert find_start_idx >= 0, "CONTEXT:%s, TOKEN: %s, C_POINTER: %d" % (context_text, token, c_pointer)
                end_idx = find_start_idx + len(token)
                token_spans.append((find_start_idx, end_idx))
                c_pointer = end_idx
            sent_spans.append(token_spans)
        return sent_spans

    @staticmethod
    def sentence_label_generation(context_text, context_token, answer_text, char_start):
        sent_spans = Dataset.find_token_boundaries(context_text, context_token)

        sent_label = None
        for idx_t, token_spans in enumerate(sent_spans):
            if char_start < token_spans[-1][1]:
                sent_label = idx_t
                break
        assert sent_label is not None
        return sent_label




if __name__ == '__main__':
    dev_data_path = "/Users/xxx/Workspaces/dataset/SQuAD/dev-v1.1.json"
    data_obj = Dataset(dev_data_path, 'train')








