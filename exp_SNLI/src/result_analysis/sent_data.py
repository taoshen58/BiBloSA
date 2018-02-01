import numpy as np
import os


class SentData(object):
    def __init__(self, dataset_obj):
        self.nn_data = []
        for sample in dataset_obj.nn_data:
            if 1 not in sample['sentence1_token']:
                new_sample1 = {
                    'token': sample['sentence1_token'],
                    'token_digital': sample['sentence1_token_digital']
                }
                self.nn_data.append(new_sample1)

            if 1 not in sample['sentence2_token']:
                new_sample2 = {
                    'token': sample['sentence2_token'],
                    'token_digital': sample['sentence2_token_digital']
                }
                self.nn_data.append(new_sample2)

    def get_one_sample_feed_dict_iter(self, sent1_token_placeholder, is_train_placeholder):
        for idx,sample in enumerate(self.nn_data):
            yield idx, sample['token'],\
                    self.get_one_sample_feed_dict(sample, sent1_token_placeholder, is_train_placeholder)

    def get_one_sample_feed_dict(self, sample, sent1_token_placeholder, is_train_placeholder):
        sent = np.stack([np.array(sample['token_digital'])], 0).astype('int32')
        feed_dict = {
            sent1_token_placeholder: sent,
            is_train_placeholder: False
        }
        return feed_dict


    def filter_sent(self, min_len=5, max_len=10, delete_duplicate=True):
        new_dataset = []

        for sample in self.nn_data:
            if len(sample['token']) >= min_len and len(sample['token']) <= max_len:
                new_dataset.append(sample)

        if delete_duplicate:
            new_dataset_dd = []
            sent_set = set()
            for sample in new_dataset:
                sample_sent = ' '.join(sample['token'])
                if sample_sent not in sent_set:
                    new_dataset_dd.append(sample)
                    sent_set.add(sample_sent)
            new_dataset = new_dataset_dd

        self.nn_data = new_dataset


    # save
    def save_sentence(self, path):
        with open(path, 'w', encoding='utf-8') as file:
            for idx, sample in enumerate(self.nn_data):
                file.write('%d\t%s%s' % (idx, ' '.join(sample['token']), os.linesep))














